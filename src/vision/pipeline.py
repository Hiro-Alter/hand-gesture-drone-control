from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .hand_landmarks import HandLandmarkDetector
from .roi_extractor import RoiResult, extract_hand_roi


@dataclass(frozen=True)
class VisionFrameResult:
    annotated_bgr: np.ndarray
    roi: RoiResult | None
    hand_detected: bool
    error: str = ""


class VisionPipeline:
    def __init__(self, config: dict):
        self.padding = float(config.get("hand_padding", 0.3))
        self.min_roi_size_px = int(config.get("min_roi_size_px", 5))
        self.detector = HandLandmarkDetector(
            max_num_hands=int(config.get("max_num_hands", 1)),
            min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )

    def process(self, frame_bgr: np.ndarray) -> VisionFrameResult:
        detection = self.detector.detect(frame_bgr)
        if detection is None:
            return VisionFrameResult(
                annotated_bgr=frame_bgr.copy(),
                roi=None,
                hand_detected=False,
            )

        annotated = self.detector.draw(frame_bgr, detection)
        roi = extract_hand_roi(
            frame_bgr,
            detection.landmarks.landmark,
            padding=self.padding,
            min_size_px=self.min_roi_size_px,
        )
        if roi is None:
            return VisionFrameResult(
                annotated_bgr=annotated,
                roi=None,
                hand_detected=True,
                error="ROI invalido",
            )

        x1, y1, x2, y2 = roi.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 160, 255), 2)
        return VisionFrameResult(
            annotated_bgr=annotated,
            roi=roi,
            hand_detected=True,
        )

    def close(self) -> None:
        self.detector.close()

