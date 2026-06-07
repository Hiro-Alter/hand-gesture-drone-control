from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import project_path


DEFAULT_CONFIG_PATH = project_path("config/app_config.json")


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"La configuracion no es un objeto JSON: {resolved}")
    return data


def load_app_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return load_json(path)

