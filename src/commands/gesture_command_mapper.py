from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.config_loader import load_json


@dataclass(frozen=True)
class CommandDecision:
    command: str
    label: str
    gesture: str = ""


class GestureCommandMapper:
    def __init__(self, mapping_path: str | Path):
        self.mapping = load_json(mapping_path)
        self.gestures = self.mapping.get("gestures", {})
        self.commands = self.mapping.get("commands", {})
        safe = self.mapping.get("safe_command", {"command": "hover", "label": "Detener"})
        self.safe_command = CommandDecision(
            command=str(safe.get("command", "hover")),
            label=str(safe.get("label", "Detener")),
        )

    def from_gesture(self, gesture: str) -> CommandDecision | None:
        item = self.gestures.get(gesture)
        if item is None:
            return None
        command = str(item["command"])
        label = str(item.get("label") or self.commands.get(command, {}).get("label", command))
        return CommandDecision(command=command, label=label, gesture=gesture)

    def safe(self) -> CommandDecision:
        return self.safe_command

