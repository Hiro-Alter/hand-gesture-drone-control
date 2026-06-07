from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilizerState:
    candidate: str
    count: int
    stable: bool


class GestureStabilizer:
    def __init__(self, required_frames: int, min_confidence: float):
        self.required_frames = max(1, int(required_frames))
        self.min_confidence = float(min_confidence)
        self._candidate = ""
        self._count = 0

    def update(self, gesture: str, confidence: float) -> StabilizerState:
        if confidence < self.min_confidence:
            self.reset()
            return StabilizerState(candidate="", count=0, stable=False)

        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture
            self._count = 1

        return StabilizerState(
            candidate=self._candidate,
            count=self._count,
            stable=self._count >= self.required_frames,
        )

    def reset(self) -> None:
        self._candidate = ""
        self._count = 0

