from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.airsim_client.client import AirSimClient
from src.inference.model_loader import ModelCatalog
from src.runtime.recognition_worker import RecognitionWorker
from src.ui.image_utils import ndarray_to_pixmap
from src.vision.camera import list_available_cameras


class MainWindow(QMainWindow):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.catalog = ModelCatalog(config["models"]["manifest"])
        self.airsim_client = AirSimClient(config.get("airsim", {}))
        self.worker: RecognitionWorker | None = None
        self.thread: QThread | None = None

        self.setWindowTitle(config.get("app", {}).get("window_title", "Control gestual"))
        self.resize(1180, 720)
        self._build_ui()
        self._apply_style()
        self.refresh_cameras()
        self._load_models()
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QGridLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.camera_view = QLabel("Camara")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(720, 460)
        self.camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_view.setObjectName("cameraView")
        layout.addWidget(self.camera_view, 0, 0, 2, 1)

        side = QFrame()
        side.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setSpacing(10)

        self.camera_combo = QComboBox()
        self.model_combo = QComboBox()
        self.refresh_button = QPushButton("Actualizar camaras")
        self.start_button = QPushButton("Iniciar")
        self.pause_button = QPushButton("Pausar")
        self.stop_button = QPushButton("Detener")
        self.airsim_button = QPushButton("Conectar AirSim")

        side_layout.addWidget(QLabel("Camara"))
        side_layout.addWidget(self.camera_combo)
        side_layout.addWidget(self.refresh_button)
        side_layout.addSpacing(8)
        side_layout.addWidget(QLabel("Modelo"))
        side_layout.addWidget(self.model_combo)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        side_layout.addLayout(buttons)
        side_layout.addWidget(self.airsim_button)

        self.roi_view = QLabel("ROI")
        self.roi_view.setAlignment(Qt.AlignCenter)
        self.roi_view.setFixedHeight(180)
        self.roi_view.setObjectName("roiView")
        side_layout.addWidget(self.roi_view)

        self.gesture_label = QLabel("Gesto: -")
        self.confidence_label = QLabel("Confianza: -")
        self.command_label = QLabel("Comando: -")
        self.device_label = QLabel("Dispositivo: -")
        self.model_label = QLabel("Modelo: -")
        self.airsim_label = QLabel("AirSim: no conectado")
        self.timing_label = QLabel("Tiempos: -")
        self.status_label = QLabel("Estado: listo")
        self.status_label.setWordWrap(True)

        for widget in [
            self.gesture_label,
            self.confidence_label,
            self.command_label,
            self.device_label,
            self.model_label,
            self.airsim_label,
            self.timing_label,
            self.status_label,
        ]:
            widget.setObjectName("metricLabel")
            side_layout.addWidget(widget)

        side_layout.addStretch(1)
        layout.addWidget(side, 0, 1, 2, 1)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 1)

        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.start_button.clicked.connect(self.start_recognition)
        self.pause_button.clicked.connect(self.pause_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.airsim_button.clicked.connect(self.connect_airsim)

    def _apply_style(self) -> None:
        accent = self.config.get("app", {}).get("theme_accent", "#d97706")
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: #f5f5f4;
                color: #1c1917;
                font-size: 14px;
            }}
            QLabel#cameraView, QLabel#roiView {{
                background: #1c1917;
                color: #fafaf9;
                border-radius: 6px;
            }}
            QFrame#sidePanel {{
                background: #ffffff;
                border: 1px solid #e7e5e4;
                border-radius: 8px;
            }}
            QPushButton {{
                background: {accent};
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 9px 12px;
                font-weight: 600;
            }}
            QPushButton:disabled {{
                background: #a8a29e;
            }}
            QComboBox {{
                border: 1px solid #d6d3d1;
                border-radius: 6px;
                padding: 7px;
                background: white;
            }}
            QLabel#metricLabel {{
                padding: 6px 0;
            }}
            """
        )

    def refresh_cameras(self) -> None:
        self.camera_combo.clear()
        probe_count = int(self.config.get("camera", {}).get("probe_count", 5))
        cameras = list_available_cameras(probe_count)
        if not cameras:
            default_index = int(self.config.get("camera", {}).get("default_index", 0))
            self.camera_combo.addItem(f"Camara {default_index} (sin validar)", default_index)
            self.status_label.setText("Estado: no se detectaron camaras; se deja la camara por defecto")
            return
        for camera in cameras:
            self.camera_combo.addItem(camera.label, camera.index)
        self.status_label.setText(f"Estado: {len(cameras)} camara(s) detectada(s)")

    def _load_models(self) -> None:
        self.model_combo.clear()
        default_model = self.config.get("models", {}).get("default", "resnet18")
        for name in self.catalog.model_names:
            self.model_combo.addItem(name, name)
            if name == default_model:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def start_recognition(self) -> None:
        if self.thread is not None:
            if self.worker is not None:
                self.worker.resume()
            return

        camera_index = int(self.camera_combo.currentData())
        model_name = str(self.model_combo.currentData())
        self.thread = QThread(self)
        self.worker = RecognitionWorker(self.config, camera_index, model_name, self.airsim_client)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_camera_view)
        self.worker.roi_ready.connect(self.update_roi_view)
        self.worker.status_ready.connect(self.update_status)
        self.worker.prediction_ready.connect(self.update_prediction)
        self.worker.finished.connect(self._worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self._set_running(True)

    def pause_recognition(self) -> None:
        if self.worker is not None:
            self.worker.pause()

    def stop_recognition(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("Estado: deteniendo...")

    def connect_airsim(self) -> None:
        status = self.airsim_client.connect()
        self.airsim_label.setText(f"AirSim: {status.message}")
        self.status_label.setText(f"Estado: {status.message}")

    def update_camera_view(self, frame) -> None:
        self.camera_view.setPixmap(
            ndarray_to_pixmap(frame, self.camera_view.width(), self.camera_view.height())
        )

    def update_roi_view(self, roi) -> None:
        if roi is None:
            self.roi_view.setText("ROI no disponible")
            return
        self.roi_view.setPixmap(ndarray_to_pixmap(roi, self.roi_view.width(), self.roi_view.height()))

    def update_status(self, message: str) -> None:
        self.status_label.setText(f"Estado: {message}")

    def update_prediction(self, data: dict) -> None:
        confidence = float(data.get("confidence", 0.0))
        self.gesture_label.setText(f"Gesto: {data.get('gesture', '-')}")
        self.confidence_label.setText(f"Confianza: {confidence:.1%}")
        self.command_label.setText(f"Comando: {data.get('command', '-') or '-'}")
        self.device_label.setText(f"Dispositivo: {data.get('device', '-') or '-'}")
        self.model_label.setText(f"Modelo: {data.get('model', '-')}")
        self.airsim_label.setText(f"AirSim: {data.get('airsim_status', '-')}")
        self.timing_label.setText(
            "Tiempos: "
            f"inferencia {float(data.get('inference_time_ms', 0.0)):.1f} ms / "
            f"pipeline {float(data.get('pipeline_time_ms', 0.0)):.1f} ms"
        )

    def _worker_finished(self) -> None:
        self.worker = None
        self.thread = None
        self._set_running(False)
        self.status_label.setText("Estado: detenido")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.camera_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            self.worker.stop()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(2000)
        self.airsim_client.disconnect()
        event.accept()

