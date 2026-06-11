from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


MOTION_COMMANDS = frozenset(
    {
        "forward",
        "backward",
        "left",
        "right",
        "ascend",
        "descend",
        "rotate_yaw",
    }
)
DISCRETE_COMMANDS = frozenset({"takeoff", "land"})
CONTROL_COMMANDS = frozenset({"hover"})
KNOWN_COMMANDS = MOTION_COMMANDS | DISCRETE_COMMANDS | CONTROL_COMMANDS


class DroneControlPhase(str, Enum):
    DISCONNECTED = "disconnected"
    READY = "ready"
    GROUNDED = "grounded"
    TAKING_OFF = "taking_off"
    AIRBORNE = "airborne"
    LANDING = "landing"
    ERROR = "error"


@dataclass(frozen=True)
class ControlDecision:
    accepted: bool
    command: str
    phase: DroneControlPhase
    reason: str


class DroneControlStateMachine:
    def __init__(self, motion_refresh_s: float = 2.0):
        self.motion_refresh_s = max(0.05, float(motion_refresh_s))
        self.phase = DroneControlPhase.DISCONNECTED
        self.active_command = ""
        self._last_accept_at = 0.0

    def on_connected(self, landed: bool | None = None) -> None:
        if landed is True:
            self.phase = DroneControlPhase.GROUNDED
            self.active_command = ""
        elif landed is False:
            self.phase = DroneControlPhase.AIRBORNE
            self.active_command = "hover"
        else:
            self.phase = DroneControlPhase.READY
            self.active_command = ""
        self._last_accept_at = 0.0

    def on_disconnected(self) -> None:
        self.phase = DroneControlPhase.DISCONNECTED
        self.active_command = ""
        self._last_accept_at = 0.0

    def request(self, command: str) -> ControlDecision:
        command = str(command or "").strip()
        now = time.monotonic()

        if command not in KNOWN_COMMANDS:
            return self._reject(command, "Comando no reconocido")
        if self.phase == DroneControlPhase.DISCONNECTED:
            return self._reject(command, "Dron desconectado")
        if self.phase == DroneControlPhase.LANDING:
            return self._reject(command, "Aterrizaje en curso; se ignoran comandos nuevos")
        if self.phase == DroneControlPhase.TAKING_OFF:
            return self._reject(command, "Despegue en curso")

        if command == "land":
            if self.phase == DroneControlPhase.GROUNDED:
                return self._reject(command, "El dron ya está en tierra")
            self.phase = DroneControlPhase.LANDING
            self.active_command = "land"
            self._last_accept_at = now
            return self._accept(command, "Aterrizaje bloqueante aceptado")

        if command == "takeoff":
            if self.phase == DroneControlPhase.AIRBORNE:
                return self._reject(command, "El dron ya está en vuelo")
            self.phase = DroneControlPhase.TAKING_OFF
            self.active_command = "takeoff"
            self._last_accept_at = now
            return self._accept(command, "Despegue aceptado")

        if self.phase == DroneControlPhase.GROUNDED:
            return self._reject(command, "Comando ignorado: primero debe despegar")

        if command == self.active_command:
            if command in MOTION_COMMANDS and now - self._last_accept_at >= self.motion_refresh_s:
                self._last_accept_at = now
                return self._accept(command, "Renovación temporizada de movimiento")
            return self._reject(command, "Comando duplicado; se mantiene el curso actual")

        if command in MOTION_COMMANDS or command == "hover":
            self.phase = DroneControlPhase.AIRBORNE
            self.active_command = command
            self._last_accept_at = now
            return self._accept(command, "Cambio de curso aceptado")

        return self._reject(command, "Transición no permitida")

    def on_command_finished(self, command: str, success: bool, landed: bool | None = None) -> None:
        command = str(command or "")
        if not success:
            self.phase = DroneControlPhase.ERROR
            return
        if command == "takeoff":
            self.phase = DroneControlPhase.AIRBORNE
            self.active_command = "hover"
            return
        if command == "land":
            self.phase = DroneControlPhase.GROUNDED
            self.active_command = ""
            return
        if landed is True:
            self.phase = DroneControlPhase.GROUNDED
            self.active_command = ""
            return
        if command in MOTION_COMMANDS or command == "hover":
            self.phase = DroneControlPhase.AIRBORNE
            self.active_command = command

    def _accept(self, command: str, reason: str) -> ControlDecision:
        return ControlDecision(True, command, self.phase, reason)

    def _reject(self, command: str, reason: str) -> ControlDecision:
        return ControlDecision(False, command, self.phase, reason)
