from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RoiResult:
    image_bgr: np.ndarray
    box: tuple[int, int, int, int]


def extract_hand_roi(
    frame_bgr: np.ndarray,
    landmarks: Sequence[object],
    padding: float,
    min_size_px: int = 5,
) -> RoiResult | None:
    height, width = frame_bgr.shape[:2]
    xs = [float(landmark.x) * width for landmark in landmarks]
    ys = [float(landmark.y) * height for landmark in landmarks]
    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    side = int(max(max_x - min_x, max_y - min_y) * (1.0 + padding))
    side = max(side, min_size_px)

    x1 = max(0, int(center_x - side / 2))
    y1 = max(0, int(center_y - side / 2))
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    if x2 - x1 < min_size_px or y2 - y1 < min_size_px:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return RoiResult(image_bgr=roi, box=(x1, y1, x2, y2))

