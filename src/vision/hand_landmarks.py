from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class HandDetection:
    landmarks: object


class HandLandmarkDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> HandDetection | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return None
        return HandDetection(landmarks=result.multi_hand_landmarks[0])

    def draw(self, frame_bgr: np.ndarray, detection: HandDetection) -> np.ndarray:
        annotated = frame_bgr.copy()
        self._drawer.draw_landmarks(
            annotated,
            detection.landmarks,
            self._mp_hands.HAND_CONNECTIONS,
        )
        return annotated

    def close(self) -> None:
        self._hands.close()

