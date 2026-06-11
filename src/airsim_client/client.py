from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

from src.commands.drone_control_state import DroneControlStateMachine


@dataclass
class AirSimStatus:
    connected: bool
    message: str
    accepted: bool = False


class AirSimClient:
    def __init__(self, config: dict):
        self.config = config
        self.ip = str(config.get("ip", "127.0.0.1"))
        self.port = int(config.get("port", 41451))
        self.timeout_s = float(config.get("timeout_s", 5.0))
        self.vehicle_name = str(config.get("vehicle_name", ""))
        horizontal_speed = float(config.get("horizontal_speed_mps", 2.0))
        self.forward_speed = float(config.get("forward_speed_mps", horizontal_speed))
        self.lateral_speed = float(config.get("lateral_speed_mps", horizontal_speed))
        self.vertical_speed = float(config.get("vertical_speed_mps", 1.5))
        self.yaw_rate = float(config.get("yaw_rate_deg_s", 45.0))
        self.duration = float(config.get("movement_duration_s", 0.5))
        self.use_body_frame = bool(config.get("use_body_frame", True))
        self.control_state = DroneControlStateMachine(config.get("motion_refresh_s", 2.0))
        self._airsim: Any | None = None
        self._client: Any | None = None
        self._lock = Lock()
        self._command_event = Event()
        self._command_stop = Event()
        self._command_thread: Thread | None = None
        self._pending_command = ""
        self.status = AirSimStatus(connected=False, message="AirSim no conectado")

    @property
    def connected(self) -> bool:
        return self.status.connected

    def update_config(self, config: dict) -> None:
        self.config = config
        self.ip = str(config.get("ip", self.ip))
        self.port = int(config.get("port", self.port))
        self.timeout_s = float(config.get("timeout_s", self.timeout_s))
        self.vehicle_name = str(config.get("vehicle_name", self.vehicle_name))
        horizontal_speed = float(config.get("horizontal_speed_mps", self.forward_speed))
        self.forward_speed = float(config.get("forward_speed_mps", horizontal_speed))
        self.lateral_speed = float(config.get("lateral_speed_mps", horizontal_speed))
        self.vertical_speed = float(config.get("vertical_speed_mps", self.vertical_speed))
        self.yaw_rate = float(config.get("yaw_rate_deg_s", self.yaw_rate))
        self.duration = float(config.get("movement_duration_s", self.duration))
        self.use_body_frame = bool(config.get("use_body_frame", self.use_body_frame))
        self.control_state.motion_refresh_s = max(0.05, float(config.get("motion_refresh_s", self.control_state.motion_refresh_s)))

    def connect(self) -> AirSimStatus:
        if self._client is not None and self.status.connected:
            return self.status
        try:
            import airsim

            self._airsim = airsim
            self._client = airsim.MultirotorClient(
                ip=self.ip,
                port=self.port,
                timeout_value=self.timeout_s,
            )
            self._client.confirmConnection()
            self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.control_state.on_connected(self._read_landed_state())
            self.status = AirSimStatus(True, f"AirSim conectado ({self.ip}:{self.port})")
            self._start_command_worker()
        except Exception as exc:
            self._client = None
            self.status = AirSimStatus(False, f"AirSim no disponible en {self.ip}:{self.port}: {exc}")
        return self.status

    def disconnect(self) -> AirSimStatus:
        self._stop_command_worker()
        if self._client is not None:
            try:
                self._client.enableApiControl(False, vehicle_name=self.vehicle_name)
            except Exception:
                pass
        self._client = None
        self.control_state.on_disconnected()
        self.status = AirSimStatus(False, "AirSim desconectado")
        return self.status

    def send_command(self, command: str) -> AirSimStatus:
        with self._lock:
            if self._client is None:
                return AirSimStatus(False, "AirSim no conectado")
            decision = self.control_state.request(command)
            if not decision.accepted:
                self.status = AirSimStatus(True, decision.reason)
                return self.status
            self._pending_command = command
            self.status = AirSimStatus(True, f"{decision.reason}: {command}", accepted=True)
            self._command_event.set()
            return self.status

    def _start_command_worker(self) -> None:
        if self._command_thread is not None and self._command_thread.is_alive():
            return
        self._command_stop.clear()
        self._command_event.clear()
        self._pending_command = ""
        self._command_thread = Thread(target=self._command_loop, name="airsim-command-worker", daemon=True)
        self._command_thread.start()

    def _stop_command_worker(self) -> None:
        self._command_stop.set()
        self._command_event.set()
        if self._command_thread is not None:
            self._command_thread.join(timeout=max(1.0, self.duration + 0.5))
            if not self._command_thread.is_alive():
                self._command_thread = None
        with self._lock:
            self._pending_command = ""

    def _command_loop(self) -> None:
        while not self._command_stop.is_set():
            self._command_event.wait()
            self._command_event.clear()
            if self._command_stop.is_set():
                break

            while not self._command_stop.is_set():
                with self._lock:
                    command = self._pending_command
                    self._pending_command = ""
                if not command:
                    break
                status = self._execute_command(command)
                landed = self._read_landed_state()
                with self._lock:
                    self.control_state.on_command_finished(command, status.connected, landed)
                    self.status = status

    def _execute_command(self, command: str) -> AirSimStatus:
        client = self._client
        if client is None:
            return AirSimStatus(False, "AirSim no conectado")
        try:
            if command == "takeoff":
                client.armDisarm(True, vehicle_name=self.vehicle_name)
                client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            elif command == "land":
                client.landAsync(vehicle_name=self.vehicle_name).join()
                client.armDisarm(False, vehicle_name=self.vehicle_name)
            elif command == "hover":
                client.hoverAsync(vehicle_name=self.vehicle_name)
            elif command == "forward":
                self._move(client, vx=self.forward_speed, vy=0.0, vz=0.0)
            elif command == "backward":
                self._move(client, vx=-self.forward_speed, vy=0.0, vz=0.0)
            elif command == "left":
                self._move(client, vx=0.0, vy=-self.lateral_speed, vz=0.0)
            elif command == "right":
                self._move(client, vx=0.0, vy=self.lateral_speed, vz=0.0)
            elif command == "ascend":
                self._move(client, vx=0.0, vy=0.0, vz=-self.vertical_speed)
            elif command == "descend":
                self._move(client, vx=0.0, vy=0.0, vz=self.vertical_speed)
            elif command == "rotate_yaw":
                client.rotateByYawRateAsync(
                    self.yaw_rate,
                    self.duration,
                    vehicle_name=self.vehicle_name,
                )
            else:
                return AirSimStatus(False, f"Comando no soportado: {command}")
            return AirSimStatus(True, f"Comando enviado: {command}")
        except Exception as exc:
            return AirSimStatus(False, f"Error AirSim: {exc}")

    def _move(self, client: Any, vx: float, vy: float, vz: float) -> None:
        move_by_velocity = (
            client.moveByVelocityBodyFrameAsync
            if self.use_body_frame and hasattr(client, "moveByVelocityBodyFrameAsync")
            else client.moveByVelocityAsync
        )
        move_by_velocity(
            vx,
            vy,
            vz,
            self.duration,
            vehicle_name=self.vehicle_name,
        )

    def _read_landed_state(self) -> bool | None:
        client = self._client
        if client is None or self._airsim is None:
            return None
        try:
            state = client.getMultirotorState(vehicle_name=self.vehicle_name)
            return int(state.landed_state) == int(self._airsim.LandedState.Landed)
        except Exception:
            return None
