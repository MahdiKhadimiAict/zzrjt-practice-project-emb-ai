"""Unit tests for the EmotionDetection package and its Flask front end.

The tests never download an emotion model. Both backends are replaced with
stubs so the suite runs fast, offline and deterministically. The five canonical
emotion checks that need a real model live at the bottom of the file and are
skipped automatically when no backend is reachable.
"""

# Test bodies deliberately assert on private helpers and need no docstrings.
# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

import server
from EmotionDetection import emotion_detection
from EmotionDetection.emotion_detection import (
    EMOTION_LABELS,
    MAX_INPUT_CHARS,
    EmotionDetectionError,
    EmotionModelUnavailableError,
    InvalidInputError,
    backend_status,
    configured_engine,
    emotion_detector,
    validate_text,
)

WATSON_OK_PAYLOAD: Dict[str, Any] = {
    "emotionPredictions": [
        {
            "emotion": {
                "anger": 0.1,
                "disgust": 0.05,
                "fear": 0.2,
                "joy": 0.6,
                "sadness": 0.05,
            }
        }
    ]
}


WATSON_EMOTION_SCORES: Dict[str, float] = WATSON_OK_PAYLOAD["emotionPredictions"][0]["emotion"]


class FakeResponse:
    """Minimal stand in for a ``requests`` response object."""

    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else WATSON_OK_PAYLOAD

    def json(self) -> Any:
        """Return the canned payload.

        :returns: The JSON body this fake response was built with.
        """
        return self._payload


@pytest.fixture(autouse=True)
def fixture_reset_watson_cooldown() -> Any:
    """Clear the Watson failure cooldown so tests never inherit one.

    :yields: Nothing; the fixture only brackets each test.
    """
    emotion_detection._watson_retry_after = 0.0
    yield
    emotion_detection._watson_retry_after = 0.0


@pytest.fixture(autouse=True)
def fixture_reset_local_classifier_cache() -> Any:
    """Clear the memoised local pipeline so no test inherits a loaded model.

    ``_local_classifier`` is ``lru_cache`` decorated, so without this a real
    pipeline built by one test would leak into the next one.

    :yields: Nothing; the fixture only brackets each test.
    """
    emotion_detection._local_classifier.cache_clear()
    yield
    emotion_detection._local_classifier.cache_clear()


@pytest.fixture(name="app_client")
def fixture_app_client() -> Any:
    """Provide a Flask test client bound to the emotion detector app.

    :returns: A ``werkzeug`` test client for ``server.app``.
    """
    server.app.config.update(TESTING=True)
    with server.app.test_client() as client:
        yield client


@pytest.fixture(name="watson_scores")
def fixture_watson_scores() -> Any:
    """Pin the Watson backend to a joy dominant score distribution.

    :yields: The patched ``_watson_scores`` function.
    """
    scores = dict(WATSON_EMOTION_SCORES)
    with patch.object(emotion_detection, "_watson_scores", return_value=scores) as stub:
        yield stub


@pytest.fixture(name="fake_local_pipeline")
def fixture_fake_local_pipeline() -> Any:
    """Pin the local backend to a stubbed transformer pipeline factory.

    The stub classifier answers with a small, deterministic GoEmotions score set
    so the fallback path never downloads a model.

    :yields: The patched ``_transformer_pipeline``.
    """
    with patch.object(emotion_detection, "_transformer_pipeline") as factory:
        factory.return_value = lambda text: [
            {"label": "joy", "score": 0.8},
            {"label": "anger", "score": 0.2},
        ]
        yield factory


class TestValidateText:
    """Tests for the blank input guard behind HTTP 400."""

    def test_trims_surrounding_whitespace(self) -> None:
        assert validate_text("  I love this product  ") == "I love this product"

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t ", "\r\n"])
    def test_rejects_blank_input(self, blank: str) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            validate_text(blank)
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("value", [None, 42, 3.5, ["text"], {"text": "hi"}])
    def test_rejects_non_string_input(self, value: Any) -> None:
        with pytest.raises(InvalidInputError):
            validate_text(value)

    def test_rejects_text_beyond_the_documented_limit(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            validate_text("a" * (MAX_INPUT_CHARS + 1))
        assert "1000" in str(excinfo.value)

    def test_accepts_text_at_the_documented_limit(self) -> None:
        assert validate_text("a" * MAX_INPUT_CHARS) == "a" * MAX_INPUT_CHARS


class TestNormalisation:
    """Tests for the shared score normalisation helper."""

    def test_projects_onto_the_five_watson_labels(self) -> None:
        result = emotion_detection._renormalise(WATSON_EMOTION_SCORES)
        assert result is not None
        assert set(result) == set(EMOTION_LABELS)

    def test_scores_sum_to_one(self) -> None:
        result = emotion_detection._renormalise(WATSON_EMOTION_SCORES)
        assert result is not None
        assert sum(result.values()) == pytest.approx(1.0)

    def test_returns_none_without_signal(self) -> None:
        assert emotion_detection._renormalise({}) is None
        assert emotion_detection._renormalise({"anger": 0.0, "joy": 0.0}) is None


class TestGoEmotionsAggregation:
    """Tests for folding the 28 GoEmotions labels into five Watson labels."""

    def test_maps_known_labels(self) -> None:
        result = emotion_detection._aggregate_goemotions({"joy": 0.7, "anger": 0.3})
        assert result is not None
        assert result["joy"] == pytest.approx(0.7)
        assert result["anger"] == pytest.approx(0.3)

    def test_sums_labels_that_share_a_target(self) -> None:
        result = emotion_detection._aggregate_goemotions(
            {"anger": 0.4, "annoyance": 0.4, "joy": 0.2}
        )
        assert result is not None
        assert result["anger"] == pytest.approx(0.8)
        assert result["joy"] == pytest.approx(0.2)

    def test_renormalises_after_summing(self) -> None:
        result = emotion_detection._aggregate_goemotions({"anger": 0.4, "annoyance": 0.4})
        assert result is not None
        assert result["anger"] == pytest.approx(1.0)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_ignores_unmapped_labels(self) -> None:
        result = emotion_detection._aggregate_goemotions({"neutral": 0.9, "joy": 0.1})
        assert result is not None
        assert result["joy"] == pytest.approx(1.0)

    def test_returns_none_when_everything_is_unmapped(self) -> None:
        assert emotion_detection._aggregate_goemotions({"neutral": 0.9}) is None


class TestFlattenPredictions:
    """Tests for reading the shape variants a transformers pipeline can return."""

    def test_accepts_a_single_mapping(self) -> None:
        result = emotion_detection._flatten_predictions({"label": "joy", "score": 0.9})
        assert result == {"joy": 0.9}

    def test_accepts_a_list_of_mappings(self) -> None:
        result = emotion_detection._flatten_predictions(
            [{"label": "joy", "score": 0.9}, {"label": "anger", "score": 0.1}]
        )
        assert result == {"joy": 0.9, "anger": 0.1}

    def test_unwraps_a_single_item_batch(self) -> None:
        batched = [[{"label": "joy", "score": 0.9}, {"label": "anger", "score": 0.1}]]
        assert emotion_detection._flatten_predictions(batched) == {"joy": 0.9, "anger": 0.1}

    @pytest.mark.parametrize("predictions", ["joy", None, 42, [], [42], [None]])
    def test_ignores_anything_it_cannot_read(self, predictions: Any) -> None:
        assert not emotion_detection._flatten_predictions(predictions)

    def test_skips_entries_missing_a_label_or_a_score(self) -> None:
        result = emotion_detection._flatten_predictions(
            [{"label": "joy", "score": 0.9}, {"label": "anger"}, {"score": 0.5}, "junk"]
        )
        assert result == {"joy": 0.9}

    def test_coerces_string_labels_and_numeric_scores(self) -> None:
        result = emotion_detection._flatten_predictions([{"label": 7, "score": "0.25"}])
        assert result == {"7": 0.25}

    def test_keeps_the_last_score_for_a_repeated_label(self) -> None:
        result = emotion_detection._flatten_predictions(
            [{"label": "joy", "score": 0.9}, {"label": "joy", "score": 0.3}]
        )
        assert result == {"joy": 0.3}


class TestWatsonBackend:
    """Tests for the IBM Watson NLP emotion workflow adapter."""

    def test_parses_the_watson_response(self) -> None:
        with patch.object(emotion_detection.requests, "post", return_value=FakeResponse()):
            result = emotion_detection._watson_scores("I am glad this happened")
        assert result is not None
        assert result["joy"] == pytest.approx(0.6)

    def test_sends_the_model_id_header(self) -> None:
        with patch.object(
            emotion_detection.requests, "post", return_value=FakeResponse()
        ) as mocked_post:
            emotion_detection._watson_scores("text")
        headers = mocked_post.call_args.kwargs["headers"]
        assert headers["grpc-metadata-mm-model-id"] == emotion_detection.WATSON_MODEL_ID

    def test_returns_none_on_non_200(self) -> None:
        with patch.object(
            emotion_detection.requests, "post", return_value=FakeResponse(status_code=500)
        ):
            assert emotion_detection._watson_scores("text") is None

    def test_returns_none_when_the_endpoint_is_unreachable(self) -> None:
        error = emotion_detection.requests.ConnectionError("no route to host")
        with patch.object(emotion_detection.requests, "post", side_effect=error):
            assert emotion_detection._watson_scores("text") is None

    def test_returns_none_on_a_malformed_body(self) -> None:
        with patch.object(
            emotion_detection.requests, "post", return_value=FakeResponse(payload={})
        ):
            assert emotion_detection._watson_scores("text") is None

    def test_opens_a_cooldown_after_a_failure(self) -> None:
        with patch.object(
            emotion_detection.requests, "post", return_value=FakeResponse(status_code=500)
        ):
            emotion_detection._watson_scores("text")
        assert emotion_detection._watson_cooldown_active() is True

    def test_cooldown_skips_the_next_request(self) -> None:
        emotion_detection._start_watson_cooldown()
        with patch.object(emotion_detection.requests, "post") as post_mock:
            assert emotion_detection._watson_scores("text") is None
        post_mock.assert_not_called()

    def test_cooldown_expires_on_its_own(self) -> None:
        emotion_detection._start_watson_cooldown()
        emotion_detection._watson_retry_after = emotion_detection.time.monotonic() - 1
        assert emotion_detection._watson_cooldown_active() is False


class TestEmotionDetector:
    """Tests for the public ``emotion_detector`` entry point."""

    def test_returns_all_five_scores_and_the_dominant_emotion(self) -> None:
        with patch.object(
            emotion_detection, "_watson_scores", return_value=WATSON_EMOTION_SCORES
        ):
            result = emotion_detector("I am glad this happened")
        for label in EMOTION_LABELS:
            assert label in result
        assert result["dominant_emotion"] == "joy"

    def test_blank_input_raises_the_400_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            emotion_detector("")
        assert excinfo.value.status_code == 400

    def test_none_input_raises_the_400_error(self) -> None:
        with pytest.raises(InvalidInputError) as excinfo:
            emotion_detector(None)
        assert excinfo.value.status_code == 400

    def test_falls_back_to_the_local_backend(self) -> None:
        local = {"anger": 0.1, "disgust": 0.0, "fear": 0.0, "joy": 0.8, "sadness": 0.1}
        with patch.object(emotion_detection, "_watson_scores", return_value=None):
            with patch.object(emotion_detection, "_local_scores", return_value=local) as local_mock:
                result = emotion_detector("I am glad this happened")
        local_mock.assert_called_once()
        assert result["dominant_emotion"] == "joy"

    def test_watson_engine_raises_when_unreachable(self) -> None:
        with patch.dict("os.environ", {"EMOTION_ENGINE": "watson"}):
            with patch.object(emotion_detection, "_watson_scores", return_value=None):
                with pytest.raises(EmotionModelUnavailableError) as excinfo:
                    emotion_detector("text")
        assert excinfo.value.status_code == 503

    def test_local_engine_skips_the_watson_call(self) -> None:
        local = {"anger": 0.0, "disgust": 0.0, "fear": 0.1, "joy": 0.0, "sadness": 0.9}
        with patch.dict("os.environ", {"EMOTION_ENGINE": "local"}):
            with patch.object(emotion_detection, "_watson_scores") as watson_mock:
                with patch.object(emotion_detection, "_local_scores", return_value=local):
                    result = emotion_detector("I am so sad about this")
        watson_mock.assert_not_called()
        assert result["dominant_emotion"] == "sadness"

    def test_local_backend_failure_propagates_as_503(self) -> None:
        with patch.object(emotion_detection, "_watson_scores", return_value=None):
            with patch.object(
                emotion_detection,
                "_local_classifier",
                side_effect=EmotionModelUnavailableError("no model"),
            ):
                with pytest.raises(EmotionModelUnavailableError):
                    emotion_detector("text")

    def test_missing_transformers_is_reported_clearly(self) -> None:
        with patch.object(emotion_detection, "_transformer_pipeline", None):
            with pytest.raises(EmotionModelUnavailableError) as excinfo:
                emotion_detection._local_classifier.cache_clear()
                emotion_detection._local_classifier()
        assert "transformers" in str(excinfo.value)


class TestLocalBackend:
    """Tests for building and driving the local GoEmotions fallback."""

    def test_passes_the_documented_pipeline_arguments(self, fake_local_pipeline: Any) -> None:
        emotion_detection._local_classifier()
        kwargs = fake_local_pipeline.call_args.kwargs
        assert kwargs["task"] == "text-classification"
        assert kwargs["model"] == emotion_detection.LOCAL_MODEL_ID
        assert kwargs["top_k"] is None
        assert kwargs["function_to_apply"] == "sigmoid"

    def test_memoises_the_pipeline(self, fake_local_pipeline: Any) -> None:
        assert emotion_detection._local_classifier() is emotion_detection._local_classifier()
        fake_local_pipeline.assert_called_once()

    def test_wraps_a_failing_pipeline_build(self) -> None:
        with patch.object(
            emotion_detection, "_transformer_pipeline", side_effect=OSError("no weights")
        ):
            with pytest.raises(EmotionModelUnavailableError) as excinfo:
                emotion_detection._local_classifier()
        assert "no weights" in str(excinfo.value)

    @pytest.mark.usefixtures("fake_local_pipeline")
    def test_goemotions_scores_reads_the_pipeline_output(self) -> None:
        assert emotion_detection._goemotions_scores("text") == {"joy": 0.8, "anger": 0.2}

    def test_wraps_a_failing_scoring_call(self, fake_local_pipeline: Any) -> None:
        def explode(text: str) -> Any:
            raise RuntimeError("model exploded")

        fake_local_pipeline.return_value = explode
        with pytest.raises(EmotionModelUnavailableError) as excinfo:
            emotion_detection._goemotions_scores("text")
        assert "model exploded" in str(excinfo.value)

    @pytest.mark.usefixtures("fake_local_pipeline")
    def test_local_scores_aggregates_onto_the_five_labels(self) -> None:
        result = emotion_detection._local_scores("text")
        assert set(result) == set(EMOTION_LABELS)
        assert result["joy"] == pytest.approx(0.8)
        assert result["anger"] == pytest.approx(0.2)

    def test_local_scores_raises_without_usable_output(self, fake_local_pipeline: Any) -> None:
        fake_local_pipeline.return_value = lambda text: [{"label": "neutral", "score": 0.9}]
        with pytest.raises(EmotionModelUnavailableError) as excinfo:
            emotion_detection._local_scores("text")
        assert "no scores" in str(excinfo.value)


class TestBackendStatus:
    """Tests for engine selection and the health summary."""

    def test_defaults_to_auto(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert configured_engine() == "auto"

    def test_reads_the_environment_variable(self) -> None:
        with patch.dict("os.environ", {"EMOTION_ENGINE": "LOCAL"}):
            assert configured_engine() == "local"

    def test_rejects_an_unknown_engine(self) -> None:
        with patch.dict("os.environ", {"EMOTION_ENGINE": "nonsense"}):
            assert configured_engine() == "auto"

    def test_status_reports_every_label(self) -> None:
        status = backend_status()
        assert status["emotion_labels"] == list(EMOTION_LABELS)
        assert status["watson_model_id"] == "emotion_aggregated-workflow_lang_en_stock"


class TestServerRoutes:
    """Tests for the Flask routes and their error handling."""

    def test_index_page_renders(self, app_client: Any) -> None:
        response = app_client.get("/")
        assert response.status_code == 200
        assert b"Emotion Detection" in response.data

    @pytest.mark.usefixtures("watson_scores")
    def test_valid_text_returns_the_formatted_breakdown(self, app_client: Any) -> None:
        response = app_client.get("/emotionDetector?textToAnalyse=I%20am%20glad")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        for label in EMOTION_LABELS:
            assert f"'{label}'" in body
        assert "dominant emotion is joy" in body

    @pytest.mark.usefixtures("watson_scores")
    def test_json_body_is_accepted(self, app_client: Any) -> None:
        response = app_client.post("/emotionDetector", json={"textToAnalyse": "I am glad"})
        assert response.status_code == 200
        assert "dominant emotion is joy" in response.get_data(as_text=True)

    @pytest.mark.usefixtures("watson_scores")
    def test_form_encoded_body_is_accepted(self, app_client: Any) -> None:
        response = app_client.post("/emotionDetector", data={"textToAnalyse": "I am glad"})
        assert response.status_code == 200
        assert "dominant emotion is joy" in response.get_data(as_text=True)

    def test_non_object_json_body_returns_400(self, app_client: Any) -> None:
        response = app_client.post("/emotionDetector", json=["I am glad"])
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "invalid_input"

    def test_blank_input_returns_400(self, app_client: Any) -> None:
        response = app_client.get("/emotionDetector?textToAnalyse=")
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["error"]["code"] == "invalid_input"
        assert payload["error"]["status"] == 400

    def test_whitespace_only_input_returns_400(self, app_client: Any) -> None:
        response = app_client.post("/emotionDetector", json={"textToAnalyse": "   \t "})
        assert response.status_code == 400

    def test_missing_input_returns_400(self, app_client: Any) -> None:
        response = app_client.get("/emotionDetector")
        assert response.status_code == 400

    def test_unreachable_model_returns_503(self, app_client: Any) -> None:
        with patch.object(
            server, "emotion_detector", side_effect=EmotionModelUnavailableError("down")
        ):
            response = app_client.get("/emotionDetector?textToAnalyse=hello")
        assert response.status_code == 503
        assert response.get_json()["error"]["code"] == "model_unavailable"

    def test_unknown_route_returns_a_json_404(self, app_client: Any) -> None:
        response = app_client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.get_json()["error"]["code"] == "not_found"

    def test_wrong_method_returns_a_json_405(self, app_client: Any) -> None:
        response = app_client.post("/health")
        assert response.status_code == 405
        assert response.get_json()["error"]["code"] == "method_not_allowed"

    def test_health_reports_the_backends(self, app_client: Any) -> None:
        response = app_client.get("/health")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ok"
        assert payload["backends"]["emotion_labels"] == list(EMOTION_LABELS)

    def test_uncaught_emotion_error_uses_the_app_error_handler(
        self, app_client: Any
    ) -> None:
        with patch.object(
            server, "render_template", side_effect=EmotionDetectionError("boom")
        ):
            response = app_client.get("/")
        assert response.status_code == 500
        assert response.get_json()["error"]["code"] == "model_unavailable"

    def test_unhandled_error_returns_a_json_500(self, app_client: Any) -> None:
        server.app.config["PROPAGATE_EXCEPTIONS"] = False
        try:
            with patch.object(server, "render_template", side_effect=RuntimeError("boom")):
                response = app_client.get("/")
        finally:
            server.app.config["PROPAGATE_EXCEPTIONS"] = None
        assert response.status_code == 500
        payload = response.get_json()
        assert payload["error"]["code"] == "internal_error"
        assert payload["error"]["status"] == 500

    def test_format_result_matches_the_documented_sentence(self) -> None:
        result: Dict[str, Any] = dict.fromkeys(EMOTION_LABELS, 0.0)
        result.update({"joy": 0.6, "dominant_emotion": "joy"})
        formatted = server._format_result(result)
        assert formatted == (
            "'anger' : 0.00, 'disgust' : 0.00, 'fear' : 0.00, "
            "'joy' : 0.60 and 'sadness' : 0.00. The dominant emotion is joy."
        )


class TestExceptionHierarchy:
    """Tests that the exceptions carry the HTTP status codes server.py relies on."""

    @pytest.mark.parametrize(
        ("exception", "status"),
        [(InvalidInputError, 400), (EmotionModelUnavailableError, 503)],
    )
    def test_status_codes(self, exception: type, status: int) -> None:
        assert exception.status_code == status
        assert issubclass(exception, EmotionDetectionError)
        assert issubclass(exception, Exception)

    def test_base_error_defaults_to_500(self) -> None:
        assert EmotionDetectionError.status_code == 500


def _backend_is_available() -> bool:
    """Report whether a real emotion model can be scored with.

    Probing downloads and loads the local model, which is far too slow to do on
    every collection run, so integration tests are opt-in. Set
    ``EMOTION_INTEGRATION_TESTS=1`` to enable them.

    :returns: ``True`` when a backend answers without raising.
    """
    if os.getenv("EMOTION_INTEGRATION_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
        return False
    try:
        with patch.object(emotion_detection, "REQUEST_TIMEOUT", 3):
            emotion_detector("I am glad this happened")
    except EmotionDetectionError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(
    not _backend_is_available(),
    reason="set EMOTION_INTEGRATION_TESTS=1 to run tests that need a real emotion model",
)
class TestCanonicalEmotions:
    """End to end checks against a real Watson NLP emotion model."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I am glad this happened", "joy"),
            ("I am really mad about this", "anger"),
            ("I feel disgusted just hearing about this", "disgust"),
            ("I am so sad about this", "sadness"),
            ("I am really afraid that this will happen", "fear"),
        ],
    )
    def test_dominant_emotion(self, text: str, expected: str) -> None:
        result: Optional[Dict[str, Any]] = emotion_detector(text)
        assert result is not None
        assert result["dominant_emotion"] == expected
