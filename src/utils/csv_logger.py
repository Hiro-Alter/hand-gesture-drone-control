from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import project_path


CSV_FIELDS = [
    "timestamp",
    "event_type",
    "model_name",
    "camera_id",
    "gesture",
    "confidence",
    "command",
    "inference_time_ms",
    "pipeline_time_ms",
    "airsim_status",
    "error_message",
]


class CsvSessionLogger:
    def __init__(self, logs_directory: str | Path):
        directory = project_path(logs_directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"gesture_session_{stamp}.csv"
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, event_type: str, **values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
        row["event_type"] = event_type
        for key, value in values.items():
            if key in row and value is not None:
                row[key] = value
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CsvSessionLogger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

