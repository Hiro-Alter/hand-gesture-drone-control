from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AirSimStatus:
    connected: bool
    message: str


class AirSimClient:
    def __init__(self, config: dict):
        self.config = config
        self.vehicle_name = str(config.get("vehicle_name", ""))
        self.horizontal_speed = float(config.get("horizontal_speed_mps", 2.0))
        self.vertical_speed = float(config.get("vertical_speed_mps", 1.5))
        self.yaw_rate = float(config.get("yaw_rate_deg_s", 45.0))
        self.duration = float(config.get("movement_duration_s", 0.5))
        self._airsim: Any | None = None
        self._client: Any | None = None
        self.status = AirSimStatus(connected=False, message="AirSim no conectado")

    @property
    def connected(self) -> bool:
        return self.status.connected

    def connect(self) -> AirSimStatus:
        try:
            import airsim

            self._airsim = airsim
            self._client = airsim.MultirotorClient()
            self._client.confirmConnection()
            self._client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.status = AirSimStatus(True, "AirSim conectado")
        except Exception as exc:
            self._client = None
            self.status = AirSimStatus(False, f"AirSim no disponible: {exc}")
        return self.status

    def disconnect(self) -> AirSimStatus:
        if self._client is not None:
            try:
                self._client.enableApiControl(False, vehicle_name=self.vehicle_name)
            except Exception:
                pass
        self._client = None
        self.status = AirSimStatus(False, "AirSim desconectado")
        return self.status

    def send_command(self, command: str) -> AirSimStatus:
        if self._client is None:
            return AirSimStatus(False, "AirSim no conectado")
        try:
            if command == "takeoff":
                self._client.armDisarm(True, vehicle_name=self.vehicle_name)
                self._client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            elif command == "land":
                self._client.landAsync(vehicle_name=self.vehicle_name).join()
                self._client.armDisarm(False, vehicle_name=self.vehicle_name)
            elif command == "hover":
                self._client.hoverAsync(vehicle_name=self.vehicle_name).join()
            elif command == "forward":
                self._move(vx=self.horizontal_speed, vy=0.0, vz=0.0)
            elif command == "backward":
                self._move(vx=-self.horizontal_speed, vy=0.0, vz=0.0)
            elif command == "left":
                self._move(vx=0.0, vy=-self.horizontal_speed, vz=0.0)
            elif command == "right":
                self._move(vx=0.0, vy=self.horizontal_speed, vz=0.0)
            elif command == "ascend":
                self._move(vx=0.0, vy=0.0, vz=-self.vertical_speed)
            elif command == "descend":
                self._move(vx=0.0, vy=0.0, vz=self.vertical_speed)
            elif command == "rotate_yaw":
                self._client.rotateByYawRateAsync(
                    self.yaw_rate,
                    self.duration,
                    vehicle_name=self.vehicle_name,
                ).join()
            else:
                return AirSimStatus(False, f"Comando no soportado: {command}")
            self.status = AirSimStatus(True, f"Comando enviado: {command}")
        except Exception as exc:
            self.status = AirSimStatus(False, f"Error AirSim: {exc}")
        return self.status

    def _move(self, vx: float, vy: float, vz: float) -> None:
        self._client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.duration,
            vehicle_name=self.vehicle_name,
        ).join()

