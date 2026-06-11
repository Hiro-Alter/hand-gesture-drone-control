from __future__ import annotations

import time


class CommandRateLimiter:
    def __init__(self, send_rate_hz: float, repeat_same_command_s: float | None = None):
        rate = max(0.1, float(send_rate_hz))
        self.min_interval_s = 1.0 / rate
        self.repeat_same_command_s = (
            max(self.min_interval_s, float(repeat_same_command_s))
            if repeat_same_command_s is not None
            else self.min_interval_s
        )
        self._last_command = ""
        self._last_sent_at = 0.0

    def should_send(self, command: str) -> bool:
        now = time.monotonic()
        if command != self._last_command:
            self._last_command = command
            self._last_sent_at = now
            return True
        if now - self._last_sent_at >= self.repeat_same_command_s:
            self._last_sent_at = now
            return True
        return False

    def reset(self) -> None:
        self._last_command = ""
        self._last_sent_at = 0.0
