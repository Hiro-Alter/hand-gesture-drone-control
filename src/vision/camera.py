from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraInfo:
    index: int
    label: str


def list_available_cameras(probe_count: int = 5) -> list[CameraInfo]:
    cameras: list[CameraInfo] = []
    for index in range(max(0, probe_count)):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                cameras.append(CameraInfo(index=index, label=f"Camara {index}"))
        finally:
            cap.release()
    return cameras


class CameraDevice:
    def __init__(self, camera_index: int, mirror: bool = True):
        self.camera_index = camera_index
        self.mirror = mirror
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self.release()
            raise RuntimeError(f"No se pudo abrir la camara {self.camera_index}")

    def read(self):
        if self._capture is None:
            raise RuntimeError("La camara no esta abierta")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("No se pudo leer un frame de la camara")
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

