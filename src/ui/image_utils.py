from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def with_prediction_overlay(image_bgr: np.ndarray, prediction: dict | None) -> np.ndarray:
    if not prediction:
        return image_bgr

    frame = image_bgr.copy()
    gesture = str(prediction.get("gesture", "-"))
    confidence = float(prediction.get("confidence", 0.0))
    command = str(prediction.get("command", "") or "-")
    text = f"Gesto: {gesture}  Conf: {confidence:.1%}  Comando: {command}"

    height, width = frame.shape[:2]
    font_scale = 0.78
    max_text_width = max(120, width - 36)
    while font_scale > 0.45:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
        if text_width <= max_text_width:
            break
        font_scale -= 0.05
    thickness = 2 if font_scale > 0.55 else 1

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 58), (width, height), (24, 24, 27), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(
        frame,
        text,
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 170, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def ndarray_to_pixmap(image: np.ndarray, width: int, height: int) -> QPixmap:
    if image.ndim == 2:
        qimage = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
    return QPixmap.fromImage(qimage).scaled(
        width,
        height,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
