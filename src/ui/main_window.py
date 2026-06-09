from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.airsim_client.client import AirSimClient, AirSimStatus
from src.inference.model_loader import ModelCatalog
from src.runtime.recognition_worker import RecognitionWorker
from src.ui.image_utils import ndarray_to_pixmap, with_prediction_overlay
from src.utils.paths import project_path
from src.vision.camera import list_available_cameras


class MainWindow(QMainWindow):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.catalog = ModelCatalog(config["models"]["manifest"])
        self.airsim_client = AirSimClient(config.get("airsim", {}))
        self.worker: RecognitionWorker | None = None
        self.thread: QThread | None = None
        self.latest_prediction: dict[str, Any] | None = None
        self.paused = False
        self.dark_mode = str(config.get("app", {}).get("theme_mode", "light")).lower() == "dark"

        self.setWindowTitle(config.get("app", {}).get("window_title", "Control gestual"))
        self.resize(1280, 780)
        self._build_ui()
        self._apply_style()
        self.refresh_cameras()
        self._load_models()
        self._load_config_controls()
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        header = QFrame()
        header.setObjectName("appHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.title_label = QLabel("Sistema de vision artificial para direccionamiento de dron virtual")
        self.title_label.setObjectName("appTitle")
        self.title_label.setWordWrap(True)
        self.subtitle_label = QLabel("Control mediante gestos de la mano")
        self.subtitle_label.setObjectName("appSubtitle")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)

        self.header_status_label = QLabel("Sistema listo")
        self.header_status_label.setObjectName("headerStatus")
        self.header_status_label.setMinimumWidth(170)
        self.header_status_label.setMaximumWidth(360)
        self.header_status_label.setAlignment(Qt.AlignCenter)
        self.header_status_label.setWordWrap(True)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("secondaryButton")
        self.theme_button.setMinimumWidth(120)

        header_layout.addLayout(title_box, 1)
        header_layout.addWidget(self.header_status_label)
        header_layout.addWidget(self.theme_button)
        root_layout.addWidget(header)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        root_layout.addWidget(self.tabs, 1)

        self.operation_tab = QWidget()
        self.config_tab = QWidget()
        self.airsim_tab = QWidget()
        self.logs_tab = QWidget()
        self.tests_tab = QWidget()

        self.tabs.addTab(self.operation_tab, "Operacion")
        self.tabs.addTab(self.config_tab, "Configuracion")
        self.tabs.addTab(self.airsim_tab, "AirSim")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.tests_tab, "Pruebas")

        self._build_operation_tab()
        self._build_config_tab()
        self._build_airsim_tab()
        self._build_logs_tab()
        self._build_tests_tab()
        self.theme_button.clicked.connect(self.toggle_theme)
        self._update_theme_button()

    def _build_operation_tab(self) -> None:
        layout = QGridLayout(self.operation_tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.camera_view = QLabel("Camara")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(760, 500)
        self.camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_view.setObjectName("cameraView")
        layout.addWidget(self.camera_view, 0, 0, 2, 1)

        side = QFrame()
        side.setObjectName("sidePanel")
        side.setMinimumWidth(320)
        side.setMaximumWidth(380)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)

        self.camera_combo = QComboBox()
        self.model_combo = QComboBox()
        self.refresh_button = QPushButton("Actualizar camaras")
        self.start_button = QPushButton("Iniciar")
        self.pause_button = QPushButton("Pausar")
        self.stop_button = QPushButton("Detener")
        self.airsim_button = QPushButton("Conectar AirSim")
        self.refresh_button.setObjectName("secondaryButton")
        self.pause_button.setObjectName("secondaryButton")
        self.stop_button.setObjectName("dangerButton")

        side_layout.addWidget(QLabel("Camara"))
        side_layout.addWidget(self.camera_combo)
        side_layout.addWidget(self.refresh_button)
        side_layout.addSpacing(8)
        side_layout.addWidget(QLabel("Modelo"))
        side_layout.addWidget(self.model_combo)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        side_layout.addLayout(buttons)
        side_layout.addWidget(self.airsim_button)

        self.gesture_value = self._metric_card(side_layout, "Gesto", "-")
        self.confidence_value = self._metric_card(side_layout, "Confianza", "-")
        self.command_value = self._metric_card(side_layout, "Comando", "-")

        self.roi_view = QLabel("ROI")
        self.roi_view.setAlignment(Qt.AlignCenter)
        self.roi_view.setFixedHeight(140)
        self.roi_view.setObjectName("roiView")
        side_layout.addWidget(self.roi_view)

        details_frame = QFrame()
        details_frame.setObjectName("statusDetails")
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(2)
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.details_scroll.setFrameShape(QFrame.NoFrame)
        self.details_scroll.setFixedHeight(150)
        self.details_scroll.setWidget(details_frame)

        self.device_label = QLabel("Dispositivo: -")
        self.model_label = QLabel("Modelo: -")
        self.op_airsim_label = QLabel("AirSim: no conectado")
        self.timing_label = QLabel("Tiempos: -")
        self.status_label = QLabel("Estado: listo")
        self.status_label.setWordWrap(True)

        for widget in [
            self.device_label,
            self.model_label,
            self.op_airsim_label,
            self.timing_label,
            self.status_label,
        ]:
            widget.setObjectName("metricLabel")
            widget.setWordWrap(True)
            widget.setMinimumHeight(22)
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            details_layout.addWidget(widget)

        details_layout.addStretch(1)
        side_layout.addWidget(self.details_scroll)
        side_layout.addStretch(1)
        layout.addWidget(side, 0, 1, 2, 1, Qt.AlignTop)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)

        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.start_button.clicked.connect(self.start_recognition)
        self.pause_button.clicked.connect(self.pause_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.airsim_button.clicked.connect(self.connect_airsim)

    def _build_config_tab(self) -> None:
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        group = QGroupBox("Parametros de operacion")
        group.setMaximumWidth(760)
        form = QFormLayout(group)

        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.0, 1.0)
        self.min_confidence_spin.setSingleStep(0.05)
        self.min_confidence_spin.setDecimals(2)

        self.stability_frames_spin = QSpinBox()
        self.stability_frames_spin.setRange(1, 30)

        self.send_rate_spin = QDoubleSpinBox()
        self.send_rate_spin.setRange(0.1, 30.0)
        self.send_rate_spin.setSingleStep(0.5)
        self.send_rate_spin.setDecimals(1)

        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(10, 500)
        self.frame_interval_spin.setSuffix(" ms")

        self.hand_padding_spin = QDoubleSpinBox()
        self.hand_padding_spin.setRange(0.0, 1.5)
        self.hand_padding_spin.setSingleStep(0.05)
        self.hand_padding_spin.setDecimals(2)

        self.mirror_check = QCheckBox("Vista espejo")

        form.addRow("Confianza minima", self.min_confidence_spin)
        form.addRow("Frames de estabilidad", self.stability_frames_spin)
        form.addRow("Frecuencia de envio", self.send_rate_spin)
        form.addRow("Intervalo de captura", self.frame_interval_spin)
        form.addRow("Padding de mano", self.hand_padding_spin)
        form.addRow("Camara", self.mirror_check)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignLeft)
        self.apply_config_button = QPushButton("Aplicar en sesion")
        self.save_config_button = QPushButton("Guardar configuracion")
        self.apply_config_button.setObjectName("secondaryButton")
        self.apply_config_button.setMaximumWidth(190)
        self.save_config_button.setMaximumWidth(210)
        buttons.addWidget(self.apply_config_button)
        buttons.addWidget(self.save_config_button)

        self.config_status_label = QLabel("Los cambios aplican al iniciar una nueva captura.")
        self.config_status_label.setObjectName("metricLabel")
        self.config_status_label.setWordWrap(True)
        self.config_status_label.setMaximumWidth(760)

        layout.addWidget(group)
        layout.addLayout(buttons)
        layout.addWidget(self.config_status_label)
        layout.addStretch(1)

        self.apply_config_button.clicked.connect(self.apply_config_controls)
        self.save_config_button.clicked.connect(self.save_config_controls)

    def _build_airsim_tab(self) -> None:
        layout = QVBoxLayout(self.airsim_tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        airsim_panel = QFrame()
        airsim_panel.setObjectName("contentPanel")
        airsim_panel.setMaximumWidth(780)
        panel_layout = QVBoxLayout(airsim_panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(12)

        self.airsim_status_label = QLabel("AirSim: no conectado")
        self.airsim_status_label.setObjectName("statusBanner")
        self.airsim_status_label.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.setAlignment(Qt.AlignLeft)
        self.airsim_connect_button = QPushButton("Conectar")
        self.airsim_disconnect_button = QPushButton("Desconectar")
        self.airsim_disconnect_button.setObjectName("secondaryButton")
        self.airsim_connect_button.setMaximumWidth(150)
        self.airsim_disconnect_button.setMaximumWidth(150)
        buttons.addWidget(self.airsim_connect_button)
        buttons.addWidget(self.airsim_disconnect_button)

        command_group = QGroupBox("Comandos manuales")
        command_group.setMaximumWidth(560)
        command_layout = QGridLayout(command_group)
        command_layout.setHorizontalSpacing(8)
        command_layout.setVerticalSpacing(8)
        commands = [
            ("Despegar", "takeoff"),
            ("Aterrizar", "land"),
            ("Detener", "hover"),
            ("Avanzar", "forward"),
            ("Retroceder", "backward"),
            ("Izquierda", "left"),
            ("Derecha", "right"),
            ("Subir", "ascend"),
            ("Bajar", "descend"),
            ("Girar", "rotate_yaw"),
        ]
        for index, (label, command) in enumerate(commands):
            button = QPushButton(label)
            button.setMinimumWidth(120)
            button.setMaximumWidth(165)
            button.clicked.connect(lambda _checked=False, value=command: self.send_manual_airsim_command(value))
            command_layout.addWidget(button, index // 3, index % 3)

        note = QLabel("Estos controles solo envian comandos si AirSim esta conectado. Las pruebas end-to-end quedan pendientes hasta configurar Unity/AirSim.")
        note.setWordWrap(True)
        note.setObjectName("metricLabel")

        panel_layout.addWidget(self.airsim_status_label)
        panel_layout.addLayout(buttons)
        panel_layout.addWidget(command_group)
        panel_layout.addWidget(note)
        layout.addWidget(airsim_panel, 0, Qt.AlignTop | Qt.AlignLeft)
        layout.addStretch(1)

        self.airsim_connect_button.clicked.connect(self.connect_airsim)
        self.airsim_disconnect_button.clicked.connect(self.disconnect_airsim)

    def _build_logs_tab(self) -> None:
        layout = QVBoxLayout(self.logs_tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        buttons = QHBoxLayout()
        self.refresh_logs_button = QPushButton("Actualizar logs")
        self.refresh_logs_button.setObjectName("secondaryButton")
        self.logs_path_label = QLabel("Log: -")
        self.logs_path_label.setObjectName("metricLabel")
        buttons.addWidget(self.refresh_logs_button)
        buttons.addWidget(self.logs_path_label, 1)

        self.logs_view = QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setObjectName("logsView")

        layout.addLayout(buttons)
        layout.addWidget(self.logs_view, 1)
        self.refresh_logs_button.clicked.connect(self.refresh_logs)

    def _build_tests_tab(self) -> None:
        layout = QVBoxLayout(self.tests_tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self.run_diagnostics_button = QPushButton("Ejecutar diagnostico")
        self.run_diagnostics_button.setObjectName("secondaryButton")
        self.diagnostics_view = QPlainTextEdit()
        self.diagnostics_view.setReadOnly(True)
        self.diagnostics_view.setObjectName("logsView")
        self.diagnostics_view.setPlainText(
            "Esta pestana ejecuta verificaciones locales sin conectar con Unity/AirSim.\n"
            "El modulo de pruebas sistematicas se implementara cuando existan imagenes etiquetadas de la estacion experimental."
        )

        layout.addWidget(self.run_diagnostics_button)
        layout.addWidget(self.diagnostics_view, 1)
        self.run_diagnostics_button.clicked.connect(self.run_diagnostics)

    def _metric_card(self, parent: QVBoxLayout, title: str, initial_value: str) -> QLabel:
        card = QFrame()
        card.setObjectName("metricCard")
        card.setFixedHeight(64)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(initial_value)
        value_label.setObjectName("metricValue")
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        parent.addWidget(card)
        return value_label

    def _theme_colors(self) -> dict[str, str]:
        if self.dark_mode:
            return {
                "accent": self.config.get("app", {}).get("theme_accent", "#f59e0b"),
                "accent_hover": "#d97706",
                "background": "#151515",
                "surface": "#202124",
                "surface_alt": "#183a3a",
                "panel": "#262626",
                "border": "#3f3f46",
                "text": "#f5f5f4",
                "muted": "#c7d2d0",
                "soft": "#2b2b2f",
                "camera": "#050505",
                "logs_bg": "#0b0b0f",
                "danger": "#dc2626",
                "danger_hover": "#b91c1c",
            }
        return {
            "accent": self.config.get("app", {}).get("theme_accent", "#d97706"),
            "accent_hover": "#b45309",
            "background": "#f7f8fa",
            "surface": "#ffffff",
            "surface_alt": "#eef6f5",
            "panel": "#1f2937",
            "border": "#d8dee7",
            "text": "#17202a",
            "muted": "#536171",
            "soft": "#e9edf3",
            "camera": "#0f172a",
            "logs_bg": "#111827",
            "danger": "#b91c1c",
            "danger_hover": "#991b1b",
        }

    def _apply_style(self) -> None:
        c = self._theme_colors()
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {c["background"]};
                color: {c["text"]};
                font-size: 14px;
            }}
            QLabel {{
                background: transparent;
            }}
            QFrame#appHeader {{
                background: {c["surface"]};
                border: 1px solid {c["border"]};
                border-radius: 8px;
            }}
            QLabel#appTitle {{
                color: {c["text"]};
                font-size: 22px;
                font-weight: 800;
            }}
            QLabel#appSubtitle {{
                color: {c["muted"]};
                font-size: 13px;
            }}
            QLabel#headerStatus {{
                background: {c["surface_alt"]};
                color: {c["text"]};
                border: 1px solid {c["border"]};
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 700;
            }}
            QTabWidget::pane {{
                border: 1px solid {c["border"]};
                background: {c["background"]};
                border-radius: 8px;
            }}
            QTabWidget::tab-bar {{
                left: 8px;
            }}
            QTabBar::tab {{
                background: {c["soft"]};
                color: {c["text"]};
                padding: 9px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 1px solid {c["border"]};
                margin-right: 2px;
            }}
            QTabBar::tab:hover {{
                background: {c["surface_alt"]};
            }}
            QTabBar::tab:selected {{
                background: {c["accent"]};
                color: white;
                font-weight: 700;
                border: 1px solid {c["accent"]};
            }}
            QLabel#cameraView, QLabel#roiView {{
                background: {c["camera"]};
                color: #fafaf9;
                border: 2px solid {c["border"]};
                border-radius: 8px;
                font-size: 18px;
                font-weight: 700;
            }}
            QFrame#sidePanel, QFrame#contentPanel, QFrame#statusDetails, QGroupBox {{
                background: {c["surface"]};
                border: 1px solid {c["border"]};
                border-radius: 8px;
            }}
            QScrollArea {{
                background: {c["surface"]};
                border: 1px solid {c["border"]};
                border-radius: 8px;
            }}
            QGroupBox {{
                margin-top: 12px;
                padding: 12px;
                font-weight: 700;
                color: {c["text"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QFrame#metricCard {{
                background: {c["surface_alt"]};
                border: 1px solid {c["border"]};
                border-left: 7px solid {c["accent"]};
                border-radius: 8px;
            }}
            QLabel#metricTitle {{
                color: {c["muted"]};
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }}
            QLabel#metricValue {{
                color: {c["text"]};
                font-size: 22px;
                font-weight: 800;
            }}
            QLabel#statusBanner {{
                background: {c["panel"]};
                color: #fafaf9;
                border: 1px solid {c["border"]};
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton {{
                background: {c["accent"]};
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 9px 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: {c["accent_hover"]};
            }}
            QPushButton#secondaryButton {{
                background: {c["surface_alt"]};
                color: {c["text"]};
                border: 1px solid {c["border"]};
            }}
            QPushButton#secondaryButton:hover {{
                background: {c["soft"]};
            }}
            QPushButton#dangerButton {{
                background: {c["danger"]};
                color: white;
            }}
            QPushButton#dangerButton:hover {{
                background: {c["danger_hover"]};
            }}
            QPushButton:disabled {{
                background: #a8a29e;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                border: 1px solid {c["border"]};
                border-radius: 6px;
                padding: 7px;
                background: {c["surface"]};
                color: {c["text"]};
            }}
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {c["accent"]};
            }}
            QCheckBox {{
                color: {c["text"]};
            }}
            QLabel#metricLabel {{
                color: {c["text"]};
                padding: 3px 4px;
            }}
            QPlainTextEdit#logsView {{
                background: {c["logs_bg"]};
                color: #f9fafb;
                border: 1px solid {c["border"]};
                border-radius: 8px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }}
            """
        )

    def toggle_theme(self) -> None:
        self.dark_mode = not self.dark_mode
        self.config.setdefault("app", {})["theme_mode"] = "dark" if self.dark_mode else "light"
        self._apply_style()
        self._update_theme_button()
        self.config_status_label.setText("Tema visual aplicado. Usa Guardar configuracion para conservarlo.")

    def _update_theme_button(self) -> None:
        self.theme_button.setText("Modo claro" if self.dark_mode else "Modo oscuro")

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

    def _load_config_controls(self) -> None:
        self.min_confidence_spin.setValue(float(self.config["inference"].get("min_confidence", 0.7)))
        self.stability_frames_spin.setValue(int(self.config["commands"].get("stability_frames", 5)))
        self.send_rate_spin.setValue(float(self.config["commands"].get("send_rate_hz", 5)))
        self.frame_interval_spin.setValue(int(self.config["camera"].get("frame_interval_ms", 30)))
        self.hand_padding_spin.setValue(float(self.config["vision"].get("hand_padding", 0.3)))
        self.mirror_check.setChecked(bool(self.config["camera"].get("mirror", True)))

    def apply_config_controls(self) -> None:
        self.config.setdefault("app", {})["theme_mode"] = "dark" if self.dark_mode else "light"
        self.config["inference"]["min_confidence"] = self.min_confidence_spin.value()
        self.config["commands"]["stability_frames"] = self.stability_frames_spin.value()
        self.config["commands"]["send_rate_hz"] = self.send_rate_spin.value()
        self.config["camera"]["frame_interval_ms"] = self.frame_interval_spin.value()
        self.config["vision"]["hand_padding"] = self.hand_padding_spin.value()
        self.config["camera"]["mirror"] = self.mirror_check.isChecked()
        self.config_status_label.setText("Configuracion aplicada para la proxima captura.")

    def save_config_controls(self) -> None:
        self.apply_config_controls()
        config_path = project_path("config/app_config.json")
        config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.config_status_label.setText(f"Configuracion guardada en {config_path}")

    def start_recognition(self) -> None:
        self.apply_config_controls()
        if self.thread is not None:
            if self.worker is not None:
                self.worker.resume()
                self.paused = False
                self.pause_button.setText("Pausar")
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
        if self.worker is None:
            return
        if self.paused:
            self.worker.resume()
            self.paused = False
            self.pause_button.setText("Pausar")
        else:
            self.worker.pause()
            self.paused = True
            self.pause_button.setText("Reanudar")

    def stop_recognition(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("Estado: deteniendo...")

    def connect_airsim(self) -> None:
        status = self.airsim_client.connect()
        self._set_airsim_status(status)

    def disconnect_airsim(self) -> None:
        status = self.airsim_client.disconnect()
        self._set_airsim_status(status)

    def send_manual_airsim_command(self, command: str) -> None:
        status = self.airsim_client.send_command(command)
        self._set_airsim_status(status)

    def _set_airsim_status(self, status: AirSimStatus) -> None:
        text = f"AirSim: {status.message}"
        self.op_airsim_label.setText(text)
        self.airsim_status_label.setText(text)
        self.status_label.setText(f"Estado: {status.message}")

    def update_camera_view(self, frame) -> None:
        visible_frame = with_prediction_overlay(frame, self.latest_prediction)
        self.camera_view.setPixmap(
            ndarray_to_pixmap(visible_frame, self.camera_view.width(), self.camera_view.height())
        )

    def update_roi_view(self, roi) -> None:
        if roi is None:
            self.roi_view.setText("ROI no disponible")
            return
        self.roi_view.setPixmap(ndarray_to_pixmap(roi, self.roi_view.width(), self.roi_view.height()))

    def update_status(self, message: str) -> None:
        self.status_label.setText(f"Estado: {message}")
        self.header_status_label.setText(message)

    def update_prediction(self, data: dict) -> None:
        self.latest_prediction = data
        confidence = float(data.get("confidence", 0.0))
        self.gesture_value.setText(str(data.get("gesture", "-")))
        self.confidence_value.setText(f"{confidence:.1%}")
        self.command_value.setText(str(data.get("command", "") or "-"))
        self.device_label.setText(f"Dispositivo: {data.get('device', '-') or '-'}")
        self.model_label.setText(f"Modelo: {data.get('model', '-')}")
        self.op_airsim_label.setText(f"AirSim: {data.get('airsim_status', '-')}")
        self.airsim_status_label.setText(f"AirSim: {data.get('airsim_status', '-')}")
        self.timing_label.setText(
            "Tiempos: "
            f"inferencia {float(data.get('inference_time_ms', 0.0)):.1f} ms / "
            f"pipeline {float(data.get('pipeline_time_ms', 0.0)):.1f} ms"
        )
        self.header_status_label.setText(f"{data.get('gesture', '-')} | {confidence:.1%}")

    def refresh_logs(self) -> None:
        latest = self._latest_log_file()
        if latest is None:
            self.logs_path_label.setText("Log: no hay archivos CSV")
            self.logs_view.setPlainText("No hay logs todavia. Inicia y deten una captura para generar una sesion.")
            return
        self.logs_path_label.setText(f"Log: {latest}")
        lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        self.logs_view.setPlainText("\n".join(lines[-250:]))

    def _latest_log_file(self) -> Path | None:
        logs_dir = project_path(self.config.get("logs", {}).get("directory", "logs"))
        if not logs_dir.exists():
            return None
        files = sorted(logs_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def run_diagnostics(self) -> None:
        from src.diagnostics import run_diagnostics

        results = run_diagnostics()
        lines = []
        for result in results:
            marker = "OK" if result.status == "OK" else "WARN" if result.status == "WARN" else "ERROR"
            required = "requerido" if result.required else "opcional"
            lines.append(f"[{marker}] {result.name} ({required}): {result.detail}")
        self.diagnostics_view.setPlainText("\n".join(lines))

    def _worker_finished(self) -> None:
        self.worker = None
        self.thread = None
        self._set_running(False)
        self.status_label.setText("Estado: detenido")
        self.header_status_label.setText("Sistema detenido")
        self.refresh_logs()

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.camera_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        if not running:
            self.paused = False
            self.pause_button.setText("Pausar")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            self.worker.stop()
            if self.thread is not None:
                self.thread.quit()
                self.thread.wait(2000)
        self.airsim_client.disconnect()
        event.accept()
