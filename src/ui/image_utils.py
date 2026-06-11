from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - Pillow es dependencia runtime, pero mantenemos fallback.
    Image = ImageDraw = ImageFont = None


def _load_overlay_font(size: int):
    if ImageFont is None:
        return None
    for font_name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text_with_pillow(frame_bgr: np.ndarray, text: str, max_width: int) -> np.ndarray | None:
    if Image is None or ImageDraw is None:
        return None

    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)

    font_size = 24
    font = _load_overlay_font(font_size)
    while font_size > 15 and font is not None:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            break
        font_size -= 1
        font = _load_overlay_font(font_size)

    y = height - 42
    draw.text(
        (18, y),
        text,
        font=font,
        fill=(255, 184, 28),
        stroke_width=2,
        stroke_fill=(8, 12, 18),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def with_prediction_overlay(image_bgr: np.ndarray, prediction: dict | None) -> np.ndarray:
    if not prediction:
        return image_bgr

    frame = image_bgr.copy()
    gesture = str(prediction.get("gesture", "-"))
    confidence = float(prediction.get("confidence", 0.0))
    command = str(prediction.get("command", "") or "-")
    text = f"Predicción: {gesture}  Confianza: {confidence:.1%}  Comando: {command}"

    height, width = frame.shape[:2]
    max_text_width = max(120, width - 36)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 60), (width, height), (24, 24, 27), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    pillow_frame = _draw_text_with_pillow(frame, text, max_text_width)
    if pillow_frame is not None:
        return pillow_frame

    fallback_text = f"Gesto: {gesture}  Confianza: {confidence:.1%}  Comando: {command}"
    font_scale = 0.78
    while font_scale > 0.45:
        text_width = cv2.getTextSize(fallback_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
        if text_width <= max_text_width:
            break
        font_scale -= 0.05
    thickness = 2 if font_scale > 0.55 else 1
    cv2.putText(
        frame,
        fallback_text,
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        fallback_text,
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
