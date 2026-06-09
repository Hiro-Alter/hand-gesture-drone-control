from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from src.airsim_client.client import AirSimClient
from src.commands.gesture_command_mapper import CommandDecision, GestureCommandMapper
from src.commands.rate_limiter import CommandRateLimiter
from src.commands.stabilizer import GestureStabilizer
from src.inference.model_loader import GestureClassifier, ModelCatalog
from src.inference.predictor import GesturePredictor
from src.utils.csv_logger import CsvSessionLogger
from src.vision.camera import CameraDevice
from src.vision.pipeline import VisionPipeline


class RecognitionWorker(QObject):
    frame_ready = Signal(object)
    roi_ready = Signal(object)
    status_ready = Signal(str)
    prediction_ready = Signal(dict)
    finished = Signal()

    def __init__(
        self,
        config: dict[str, Any],
        camera_index: int,
        model_name: str,
        airsim_client: AirSimClient,
    ):
        super().__init__()
        self.config = config
        self.camera_index = camera_index
        self.model_name = model_name
        self.airsim_client = airsim_client
        self._stop_requested = False
        self._paused = False

    @Slot()
    def run(self) -> None:
        logger: CsvSessionLogger | None = None
        camera: CameraDevice | None = None
        vision: VisionPipeline | None = None
        try:
            logs_dir = self.config.get("logs", {}).get("directory", "logs")
            logger = CsvSessionLogger(logs_dir)
            logger.log("system_start", model_name=self.model_name, camera_id=self.camera_index)

            catalog = ModelCatalog(self.config["models"]["manifest"])
            classifier = GestureClassifier.load(
                catalog.get(self.model_name),
                preferred_device=self.config["inference"].get("preferred_device", "directml"),
                fallback_device=self.config["inference"].get("fallback_device", "cpu"),
            )
            predictor = GesturePredictor(classifier)
            logger.log(
                "model_loaded",
                model_name=self.model_name,
                camera_id=self.camera_index,
                airsim_status=classifier.device.message,
            )
            self.status_ready.emit(f"Modelo cargado: {self.model_name} ({classifier.device.name})")

            mapper = GestureCommandMapper(self.config["commands"]["mapping"])
            stabilizer = GestureStabilizer(
                self.config["commands"].get("stability_frames", 5),
                self.config["inference"].get("min_confidence", 0.7),
            )
            limiter = CommandRateLimiter(self.config["commands"].get("send_rate_hz", 5))

            camera = CameraDevice(
                self.camera_index,
                mirror=bool(self.config.get("camera", {}).get("mirror", True)),
            )
            camera.open()
            logger.log("camera_open", model_name=self.model_name, camera_id=self.camera_index)
            self.status_ready.emit(f"Camara activa: {self.camera_index}")

            vision = VisionPipeline(self.config.get("vision", {}))
            frame_sleep = max(0.0, float(self.config.get("camera", {}).get("frame_interval_ms", 30)) / 1000.0)

            while not self._stop_requested:
                if self._paused:
                    time.sleep(0.05)
                    continue

                pipeline_start = time.perf_counter()
                frame = camera.read()
                vision_result = vision.process(frame)
                self.frame_ready.emit(vision_result.annotated_bgr)

                if not vision_result.hand_detected or vision_result.roi is None:
                    stabilizer.reset()
                    self.roi_ready.emit(None)
                    reason = vision_result.error or "Sin mano detectada"
                    self._handle_safe_command(
                        reason,
                        mapper,
                        limiter,
                        logger,
                        classifier.definition.name,
                        classifier.device.name,
                        pipeline_start,
                    )
                    time.sleep(frame_sleep)
                    continue

                prediction = predictor.predict_roi(vision_result.roi.image_bgr)
                self.roi_ready.emit(prediction.enhanced_roi_gray)
                state = stabilizer.update(prediction.gesture, prediction.confidence)
                pipeline_time_ms = (time.perf_counter() - pipeline_start) * 1000.0

                command_label = ""
                command_name = ""
                event_type = "prediction"
                if prediction.confidence < float(self.config["inference"].get("min_confidence", 0.7)):
                    decision = self._safe_decision(mapper)
                    command_label = decision.label
                    command_name = decision.command
                    event_type = "low_confidence"
                    self._send_if_allowed(decision, limiter, logger, prediction.gesture, prediction.confidence, prediction.inference_time_ms, pipeline_time_ms)
                else:
                    decision = mapper.from_gesture(prediction.gesture)
                    if decision is None:
                        logger.log(
                            "model_error",
                            model_name=classifier.definition.name,
                            camera_id=self.camera_index,
                            gesture=prediction.gesture,
                            confidence=prediction.confidence,
                            error_message="Gesto sin comando configurado",
                        )
                    else:
                        command_label = decision.label
                        command_name = decision.command
                        if state.stable:
                            self._send_if_allowed(decision, limiter, logger, prediction.gesture, prediction.confidence, prediction.inference_time_ms, pipeline_time_ms)

                logger.log(
                    event_type,
                    model_name=classifier.definition.name,
                    camera_id=self.camera_index,
                    gesture=prediction.gesture,
                    confidence=f"{prediction.confidence:.4f}",
                    command=command_name,
                    inference_time_ms=f"{prediction.inference_time_ms:.2f}",
                    pipeline_time_ms=f"{pipeline_time_ms:.2f}",
                    airsim_status=self.airsim_client.status.message,
                )
                self.prediction_ready.emit(
                    {
                        "gesture": prediction.gesture,
                        "confidence": prediction.confidence,
                        "command": command_label,
                        "command_name": command_name,
                        "stable_count": state.count,
                        "stable_required": stabilizer.required_frames,
                        "device": classifier.device.name,
                        "model": classifier.definition.name,
                        "inference_time_ms": prediction.inference_time_ms,
                        "pipeline_time_ms": pipeline_time_ms,
                        "airsim_status": self.airsim_client.status.message,
                    }
                )
                time.sleep(frame_sleep)
        except Exception as exc:
            if logger is not None:
                logger.log(
                    "system_error",
                    model_name=self.model_name,
                    camera_id=self.camera_index,
                    error_message=str(exc),
                )
            self.status_ready.emit(f"Error: {exc}")
        finally:
            if camera is not None:
                camera.release()
            if vision is not None:
                vision.close()
            if logger is not None:
                logger.log("system_stop", model_name=self.model_name, camera_id=self.camera_index)
                logger.close()
            self.finished.emit()

    def _handle_safe_command(
        self,
        reason: str,
        mapper: GestureCommandMapper,
        limiter: CommandRateLimiter,
        logger: CsvSessionLogger,
        model_name: str,
        device_name: str,
        pipeline_start: float,
    ) -> None:
        decision = self._safe_decision(mapper)
        pipeline_time_ms = (time.perf_counter() - pipeline_start) * 1000.0
        self._send_if_allowed(decision, limiter, logger, "", 0.0, 0.0, pipeline_time_ms)
        logger.log(
            "no_hand_detected",
            model_name=model_name,
            camera_id=self.camera_index,
            command=decision.command,
            pipeline_time_ms=f"{pipeline_time_ms:.2f}",
            airsim_status=self.airsim_client.status.message,
            error_message=reason,
        )
        self.prediction_ready.emit(
            {
                "gesture": "Sin mano",
                "confidence": 0.0,
                "command": decision.label,
                "command_name": decision.command,
                "stable_count": 0,
                "stable_required": 0,
                "device": device_name,
                "model": model_name,
                "inference_time_ms": 0.0,
                "pipeline_time_ms": pipeline_time_ms,
                "airsim_status": self.airsim_client.status.message,
            }
        )

    def _safe_decision(self, mapper: GestureCommandMapper) -> CommandDecision:
        return mapper.safe()

    def _send_if_allowed(
        self,
        decision: CommandDecision,
        limiter: CommandRateLimiter,
        logger: CsvSessionLogger,
        gesture: str,
        confidence: float,
        inference_time_ms: float,
        pipeline_time_ms: float,
    ) -> None:
        if not limiter.should_send(decision.command):
            return
        status = self.airsim_client.send_command(decision.command) if self.airsim_client.connected else self.airsim_client.status
        logger.log(
            "command_sent" if status.connected else "command_ready",
            model_name=self.model_name,
            camera_id=self.camera_index,
            gesture=gesture,
            confidence=f"{confidence:.4f}" if confidence else "",
            command=decision.command,
            inference_time_ms=f"{inference_time_ms:.2f}" if inference_time_ms else "",
            pipeline_time_ms=f"{pipeline_time_ms:.2f}",
            airsim_status=status.message,
        )

    @Slot()
    def stop(self) -> None:
        self._stop_requested = True

    @Slot()
    def pause(self) -> None:
        self._paused = True
        self.status_ready.emit("Captura pausada")

    @Slot()
    def resume(self) -> None:
        self._paused = False
        self.status_ready.emit("Captura reanudada")
