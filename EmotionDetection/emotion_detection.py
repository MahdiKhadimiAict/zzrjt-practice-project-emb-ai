"""Emotion detection for customer feedback text using the IBM Watson NLP library.

The Watson NLP ``emotion_aggregated-workflow_lang_en_stock`` model classifies a
whole feedback document into five emotions: anger, disgust, fear, joy and
sadness. This module calls that deployed model and returns its scores.

The IBM SkillsBuild endpoint hosting the model sits on an internal network, so
when it cannot be reached the module transparently falls back to an equivalent
local transformer classifier. Both backends emit the same five-label score
distribution, so callers never need to know which engine answered.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict, Final, List, Optional, Tuple

import requests

try:
    from transformers import pipeline as _transformer_pipeline
except ImportError:
    _transformer_pipeline = None

LOGGER = logging.getLogger(__name__)

EMOTION_LABELS: Final[Tuple[str, ...]] = ("anger", "disgust", "fear", "joy", "sadness")
TEXT_FIELD_LABEL: Final[str] = "textToAnalyse"

WATSON_EMOTION_URL: Final[str] = (
    "https://sn-watson-emotion.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_MODEL_ID: Final[str] = "emotion_aggregated-workflow_lang_en_stock"
LOCAL_MODEL_ID: Final[str] = "SamLowe/roberta-base-go_emotions"

MAX_INPUT_CHARS: Final[int] = 1000
REQUEST_TIMEOUT: Final[int] = 10
WATSON_RETRY_COOLDOWN: Final[int] = 300

ENGINE_AUTO: Final[str] = "auto"
ENGINE_WATSON: Final[str] = "watson"
ENGINE_LOCAL: Final[str] = "local"

# Mutable module state, so intentionally not UPPER_CASE.
# pylint: disable=invalid-name
_watson_retry_after: float = 0.0
# pylint: enable=invalid-name

GOEMOTIONS_TO_WATSON: Final[Dict[str, str]] = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "confusion": "fear",
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "caring": "joy",
    "desire": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "surprise": "joy",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "sadness": "sadness",
}


class EmotionDetectionError(Exception):
    """Base class for every failure raised by this module."""

    status_code = 500


class InvalidInputError(EmotionDetectionError, ValueError):
    """Raised when the supplied feedback text cannot be analyzed.

    Carries ``status_code = 400`` so the Flask layer can translate the
    exception straight into an HTTP 400 Bad Request response.
    """

    status_code = 400


class EmotionModelUnavailableError(EmotionDetectionError, RuntimeError):
    """Raised when no emotion model backend can be reached."""

    status_code = 503


def validate_text(text_to_analyse: Any) -> str:
    """Validate and normalise the raw feedback text.

    :param text_to_analyse: Raw customer feedback supplied by the caller.
    :returns: The trimmed feedback text.
    :raises InvalidInputError: If the text is not a non-empty string within the
        documented 1000 character limit.
    """
    if not isinstance(text_to_analyse, str):
        raise InvalidInputError(
            f"'{TEXT_FIELD_LABEL}' must be a string, got {type(text_to_analyse).__name__}."
        )

    cleaned = text_to_analyse.strip()
    if not cleaned:
        raise InvalidInputError(f"'{TEXT_FIELD_LABEL}' must not be blank.")
    if len(cleaned) > MAX_INPUT_CHARS:
        raise InvalidInputError(
            f"'{TEXT_FIELD_LABEL}' must not exceed {MAX_INPUT_CHARS} characters; "
            f"received {len(cleaned)}."
        )
    return cleaned


def _renormalise(scores: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Project a raw score mapping onto the five Watson emotion labels.

    :param scores: Mapping of label to score, from either backend.
    :returns: Scores for the five Watson labels summing to 1.0, or ``None`` when
        the mapping carries no usable signal.
    """
    projected = {label: 0.0 for label in EMOTION_LABELS}
    for label in EMOTION_LABELS:
        projected[label] = float(scores.get(label, 0.0))

    total = sum(projected.values())
    if total <= 0.0:
        return None
    return {label: round(value / total, 6) for label, value in projected.items()}


def _watson_cooldown_active() -> bool:
    """Report whether the Watson workflow is still inside its failure cooldown.

    :returns: ``True`` while the endpoint is being treated as down.
    """
    return time.monotonic() < _watson_retry_after


def _start_watson_cooldown() -> None:
    """Mark the Watson workflow as down for the cooldown period."""
    global _watson_retry_after  # pylint: disable=global-statement
    _watson_retry_after = time.monotonic() + WATSON_RETRY_COOLDOWN


def _watson_scores(text: str) -> Optional[Dict[str, float]]:
    """Score the text with the deployed Watson NLP emotion workflow.

    A short lived cooldown is applied after a failure so that a web request does
    not pay the connection timeout on every call while the endpoint is down.

    :param text: Validated feedback text.
    :returns: The five Watson emotion scores, or ``None`` when the endpoint is
        unreachable, in cooldown, returns a non-200 status, or sends an unusable
        body.
    """
    if _watson_cooldown_active():
        LOGGER.debug("Skipping the Watson workflow; still in cooldown.")
        return None

    payload = {"raw_document": {"text": text}}
    headers = {"grpc-metadata-mm-model-id": WATSON_MODEL_ID}

    try:
        response = requests.post(
            WATSON_EMOTION_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            LOGGER.warning(
                "Watson NLP emotion workflow returned HTTP %s.", response.status_code
            )
            _start_watson_cooldown()
            return None
        document = response.json()["emotionPredictions"][0]["emotion"]
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        LOGGER.warning("Watson NLP emotion workflow is unreachable: %s", exc)
        _start_watson_cooldown()
        return None

    return _renormalise(document)


@lru_cache(maxsize=1)
def _local_classifier() -> Any:
    """Build, and then memoise, the local transformer classification pipeline.

    :returns: A HuggingFace ``text-classification`` pipeline over the 28 label
        GoEmotions model.
    :raises EmotionModelUnavailableError: If ``transformers`` is not installed or
        the model cannot be downloaded and loaded.
    """
    if _transformer_pipeline is None:
        raise EmotionModelUnavailableError(
            "The local fallback needs 'transformers'. Install requirements.txt "
            "or set EMOTION_ENGINE=watson."
        )

    LOGGER.info("Loading local emotion model %s.", LOCAL_MODEL_ID)
    try:
        return _transformer_pipeline(
            task="text-classification",
            model=LOCAL_MODEL_ID,
            top_k=None,
            function_to_apply="sigmoid",
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise EmotionModelUnavailableError(
            f"Unable to load the local emotion model {LOCAL_MODEL_ID}: {exc}"
        ) from exc


def _flatten_predictions(predictions: Any) -> Dict[str, float]:
    """Flatten a transformers pipeline result into a label to score mapping.

    Asking for ``top_k=None`` returns every label, but the container varies
    between transformers releases: a mapping, a list of mappings, or a single
    batch wrapping a list of mappings.

    :param predictions: Raw value returned by the classification pipeline.
    :returns: A flat mapping of label to probability.
    """
    if isinstance(predictions, dict):
        entries: List[Any] = [predictions]
    elif isinstance(predictions, list):
        if predictions and isinstance(predictions[0], list):
            entries = list(predictions[0])
        else:
            entries = list(predictions)
    else:
        entries = []

    scores: Dict[str, float] = {}
    for entry in entries:
        if isinstance(entry, dict) and "label" in entry and "score" in entry:
            scores[str(entry["label"])] = float(entry["score"])
    return scores


def _goemotions_scores(text: str) -> Dict[str, float]:
    """Score the text with the local GoEmotions transformer.

    :param text: Validated feedback text.
    :returns: Raw 28 label GoEmotions scores.
    :raises EmotionModelUnavailableError: If the pipeline cannot be loaded.
    """
    classifier = _local_classifier()
    try:
        predictions = classifier(text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise EmotionModelUnavailableError(
            f"The local emotion model failed to score the input: {exc}"
        ) from exc

    return _flatten_predictions(predictions)


def _aggregate_goemotions(scores: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Fold the 28 GoEmotions labels into the five Watson emotion labels.

    Scores are summed per target emotion and renormalised, because GoEmotions is
    a multi-label classifier whose independent sigmoid outputs do not sum to one.

    :param scores: Raw GoEmotions label to probability mapping.
    :returns: The five Watson emotion scores, or ``None`` when no GoEmotions
        label mapped onto them.
    """
    grouped = {label: 0.0 for label in EMOTION_LABELS}
    for goemotions_label, score in scores.items():
        target = GOEMOTIONS_TO_WATSON.get(goemotions_label)
        if target is not None:
            grouped[target] += score

    return _renormalise(grouped)


def _local_scores(text: str) -> Dict[str, float]:
    """Score the text with the local fallback backend.

    :param text: Validated feedback text.
    :returns: The five Watson emotion scores.
    :raises EmotionModelUnavailableError: If the local model cannot produce a
        usable score distribution.
    """
    aggregated = _aggregate_goemotions(_goemotions_scores(text))
    if aggregated is None:
        raise EmotionModelUnavailableError(
            "The local emotion model produced no scores for the supplied text."
        )
    return aggregated


def configured_engine() -> str:
    """Return the configured engine preference.

    :returns: ``auto``, ``watson`` or ``local``, taken from the
        ``EMOTION_ENGINE`` environment variable and defaulting to ``auto``.
    """
    engine = os.getenv("EMOTION_ENGINE", ENGINE_AUTO).strip().lower()
    if engine not in {ENGINE_AUTO, ENGINE_WATSON, ENGINE_LOCAL}:
        LOGGER.warning("Unknown EMOTION_ENGINE %r; falling back to 'auto'.", engine)
        return ENGINE_AUTO
    return engine


def backend_status() -> Dict[str, Any]:
    """Describe the configured backends without contacting any of them.

    :returns: A JSON serialisable summary suitable for a health endpoint.
    """
    return {
        "configured_engine": configured_engine(),
        "watson_endpoint": WATSON_EMOTION_URL,
        "watson_model_id": WATSON_MODEL_ID,
        "watson_in_cooldown": _watson_cooldown_active(),
        "local_model_id": LOCAL_MODEL_ID,
        "local_backend_installed": _transformer_pipeline is not None,
        "emotion_labels": list(EMOTION_LABELS),
    }


def _resolve_scores(text: str) -> Tuple[str, Dict[str, float]]:
    """Score the text with the first available backend.

    :param text: Validated feedback text.
    :returns: The name of the backend that answered and its five emotion scores.
    :raises EmotionModelUnavailableError: If the preferred backend is
        unavailable and no fallback is permitted.
    """
    engine = configured_engine()

    if engine == ENGINE_LOCAL:
        return ENGINE_LOCAL, _local_scores(text)

    watson = _watson_scores(text)
    if watson is not None:
        return ENGINE_WATSON, watson
    if engine == ENGINE_WATSON:
        raise EmotionModelUnavailableError(
            f"The Watson NLP emotion workflow at {WATSON_EMOTION_URL} is unreachable."
        )

    return ENGINE_LOCAL, _local_scores(text)


def emotion_detector(text_to_analyse: Any) -> Dict[str, Any]:
    """Detect the emotion expressed in a piece of customer feedback.

    :param text_to_analyse: Raw customer feedback text.
    :returns: A mapping with one score per emotion in ``EMOTION_LABELS`` plus the
        ``dominant_emotion`` label holding the highest score.
    :raises InvalidInputError: If the text is blank, not a string, or longer than
        1000 characters. Carries ``status_code = 400``.
    :raises EmotionModelUnavailableError: If no emotion model backend is
        reachable. Carries ``status_code = 503``.
    """
    text = validate_text(text_to_analyse)
    engine, scores = _resolve_scores(text)

    result: Dict[str, Any] = dict(scores)
    result["dominant_emotion"] = max(scores, key=scores.get)
    LOGGER.info("Emotion detected by the %s backend: %s", engine, result)
    return result
