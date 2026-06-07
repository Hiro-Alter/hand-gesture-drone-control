from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SelectedDevice:
    name: str
    torch_device: Any
    message: str


def select_device(preferred: str = "directml", fallback: str = "cpu") -> SelectedDevice:
    preferred = preferred.lower()
    if preferred == "directml":
        try:
            import torch_directml

            return SelectedDevice(
                name="directml",
                torch_device=torch_directml.device(),
                message="DirectML activo",
            )
        except Exception as exc:
            return SelectedDevice(
                name=fallback,
                torch_device=torch.device("cpu"),
                message=f"DirectML no disponible, usando CPU: {exc}",
            )

    return SelectedDevice(
        name="cpu",
        torch_device=torch.device("cpu"),
        message="CPU activo",
    )

