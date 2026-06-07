from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


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
