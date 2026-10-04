"""Emotion detection package for the IBM Watson NLP emotion workflow."""

# The package name is mandated by the IBM Watson NLP course conventions.
# pylint: disable=invalid-name

from .emotion_detection import (
    EMOTION_LABELS,
    EmotionDetectionError,
    EmotionModelUnavailableError,
    InvalidInputError,
    backend_status,
    configured_engine,
    emotion_detector,
    validate_text,
)

__all__ = [
    "EMOTION_LABELS",
    "EmotionDetectionError",
    "EmotionModelUnavailableError",
    "InvalidInputError",
    "backend_status",
    "configured_engine",
    "emotion_detector",
    "validate_text",
]
