from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.vision.preprocessing import preprocess_roi_for_model

from .model_loader import GestureClassifier


@dataclass(frozen=True)
class PredictionResult:
    gesture: str
    confidence: float
    class_index: int
    inference_time_ms: float
    enhanced_roi_gray: np.ndarray


class GesturePredictor:
    def __init__(self, classifier: GestureClassifier):
        self.classifier = classifier

    def predict_roi(self, roi_bgr: np.ndarray) -> PredictionResult:
        tensor, enhanced = preprocess_roi_for_model(
            roi_bgr,
            self.classifier.definition.input_spec,
            self.classifier.device.torch_device,
        )
        start = time.perf_counter()
        gesture, confidence, index = self.classifier.predict(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult(
            gesture=gesture,
            confidence=confidence,
            class_index=index,
            inference_time_ms=elapsed_ms,
            enhanced_roi_gray=enhanced,
        )

