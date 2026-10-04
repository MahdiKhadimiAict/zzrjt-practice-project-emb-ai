"""Flask front end for the Watson NLP emotion detector.

Running this module deploys the emotion detection web application on
localhost:5000.

The app exposes three routes:

``GET /``
    Renders the single page interface that collects customer feedback.
``GET|POST /emotionDetector``
    Scores the submitted ``textToAnalyse`` field and returns the formatted
    emotion breakdown. Blank input is rejected with HTTP 400.
``GET /health``
    Reports which emotion model backends the process has been configured with.

Static code analysis is documented in the README and driven from the Makefile
via ``make lint``, which runs::

    pylint server.py EmotionDetection test_emotion_detection.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Final, Tuple

from flask import Flask, jsonify, render_template, request

from EmotionDetection.emotion_detection import (
    EMOTION_LABELS,
    TEXT_FIELD_LABEL,
    EmotionDetectionError,
    InvalidInputError,
    backend_status,
    emotion_detector,
)

LOGGER = logging.getLogger(__name__)

HOST: Final[str] = "0.0.0.0"
PORT: Final[int] = 5000

app = Flask("Emotion Detector")


def _error_response(code: str, message: str, status: int) -> Tuple[Any, int]:
    """Build a JSON error envelope.

    :param code: Short machine readable error code.
    :param message: Human readable explanation.
    :param status: HTTP status code to send with the response.
    :returns: A Flask response tuple of the JSON payload and the status code.
    """
    return jsonify({"error": {"code": code, "message": message, "status": status}}), status


def _extract_text() -> Any:
    """Read the feedback text from the request.

    The field is looked up in the JSON body first, then in a form encoded body,
    and finally in the query string, so both the browser client and a plain
    ``GET /emotionDetector?textToAnalyse=...`` call are supported.

    :returns: The submitted text, or ``None`` when the field is absent.
    """
    if request.is_json:
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            return payload.get(TEXT_FIELD_LABEL)
        return None
    if TEXT_FIELD_LABEL in request.form:
        return request.form[TEXT_FIELD_LABEL]
    return request.args.get(TEXT_FIELD_LABEL)


def _format_result(result: Dict[str, Any]) -> str:
    """Render the detector output in the format the web client displays.

    :param result: The mapping produced by ``emotion_detector``.
    :returns: A sentence listing every emotion score and the dominant emotion.
    """
    scored = [f"'{label}' : {result[label]:.2f}" for label in EMOTION_LABELS]
    body = ", ".join(scored[:-1]) + f" and {scored[-1]}"
    return f"{body}. The dominant emotion is {result['dominant_emotion']}."


@app.route("/emotionDetector", methods=["GET", "POST"])
def detect_emotion() -> Any:
    """Score the submitted feedback and return the emotion breakdown.

    Blank or otherwise unusable input is rejected with HTTP 400, and an
    unreachable emotion model is reported as HTTP 503.

    :returns: The formatted emotion breakdown, or a JSON error envelope.
    """
    text_to_analyse = _extract_text()
    try:
        result = emotion_detector(text_to_analyse)
    except InvalidInputError as exc:
        LOGGER.info("Rejected emotion detection request: %s", exc)
        return _error_response("invalid_input", str(exc), exc.status_code)
    except EmotionDetectionError as exc:
        LOGGER.error("Emotion model unavailable: %s", exc)
        return _error_response("model_unavailable", str(exc), exc.status_code)

    return _format_result(result)


@app.route("/")
def render_index_page() -> Any:
    """Render the main application page.

    :returns: The rendered ``index.html`` template.
    """
    return render_template("index.html")


@app.route("/health")
def health() -> Any:
    """Report the configured emotion model backends.

    :returns: A JSON summary of the backends available to this process.
    """
    return jsonify({"status": "ok", "backends": backend_status()})


@app.errorhandler(EmotionDetectionError)
def handle_emotion_detection_error(error: EmotionDetectionError) -> Any:
    """Translate any uncaught emotion detection failure into JSON.

    :param error: The raised exception.
    :returns: A JSON error envelope carrying the exception's status code.
    """
    code = "invalid_input" if isinstance(error, InvalidInputError) else "model_unavailable"
    return _error_response(code, str(error), error.status_code)


@app.errorhandler(404)
def handle_not_found(error: Any) -> Any:
    """Return a JSON envelope for unknown routes.

    :param error: The Werkzeug routing exception.
    :returns: A JSON error envelope with HTTP 404.
    """
    LOGGER.warning("Unknown route requested: %s", error)
    return _error_response("not_found", "The requested route does not exist.", 404)


@app.errorhandler(405)
def handle_method_not_allowed(error: Any) -> Any:
    """Return a JSON envelope for unsupported HTTP methods.

    :param error: The Werkzeug routing exception.
    :returns: A JSON error envelope with HTTP 405.
    """
    LOGGER.warning("Unsupported method requested: %s", error)
    return _error_response(
        "method_not_allowed", f"{request.method} is not allowed on {request.path}.", 405
    )


@app.errorhandler(500)
def handle_internal_error(error: Any) -> Any:
    """Return a JSON envelope for unhandled server side failures.

    :param error: The original exception.
    :returns: A JSON error envelope with HTTP 500.
    """
    LOGGER.exception("Unhandled error while serving a request: %s", error)
    return _error_response(
        "internal_error", "An unexpected error occurred while analyzing the text.", 500
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host=HOST, port=PORT)
