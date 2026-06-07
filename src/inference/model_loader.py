from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.utils.paths import project_path
from src.vision.preprocessing import ModelInputSpec

from .device import SelectedDevice, select_device


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    architecture: str
    torchscript_path: Path
    metadata_path: Path
    labels_path: Path
    input_spec: ModelInputSpec
    labels: list[str]


class ModelCatalog:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = project_path(manifest_path)
        self._models = self._load_manifest()

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    def get(self, model_name: str) -> ModelDefinition:
        if model_name not in self._models:
            raise KeyError(f"Modelo no definido: {model_name}")
        return self._models[model_name]

    def _load_manifest(self) -> dict[str, ModelDefinition]:
        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        models: dict[str, ModelDefinition] = {}
        for item in manifest.get("models", []):
            name = str(item["name"])
            artifacts = [project_path(path) for path in item.get("artifacts", [])]
            torchscript = next(
                (path for path in artifacts if path.name == "model_torchscript.pt"),
                None,
            )
            if torchscript is None:
                raise ValueError(f"El modelo {name} no define model_torchscript.pt")

            metadata_path = project_path(item["metadata"])
            labels_path = project_path(item["labels"])
            metadata = _load_json(metadata_path)
            labels = _load_labels(labels_path, metadata)
            input_data = metadata.get("input", manifest.get("input", {}))
            normalization = input_data.get("normalization", {})
            input_spec = ModelInputSpec(
                image_size=int(input_data.get("image_size", 224)),
                mean=tuple(float(v) for v in normalization.get("mean", [0.485, 0.456, 0.406])),
                std=tuple(float(v) for v in normalization.get("std", [0.229, 0.224, 0.225])),
                color_order=str(input_data.get("color_order", "RGB")),
            )
            models[name] = ModelDefinition(
                name=name,
                architecture=str(item.get("architecture", name)),
                torchscript_path=torchscript,
                metadata_path=metadata_path,
                labels_path=labels_path,
                input_spec=input_spec,
                labels=labels,
            )
        return models


class GestureClassifier:
    def __init__(self, definition: ModelDefinition, module: torch.jit.ScriptModule, device: SelectedDevice):
        self.definition = definition
        self.module = module
        self.device = device

    @classmethod
    def load(
        cls,
        definition: ModelDefinition,
        preferred_device: str = "directml",
        fallback_device: str = "cpu",
    ) -> "GestureClassifier":
        selected = select_device(preferred_device, fallback_device)
        try:
            module = _load_torchscript(definition.torchscript_path, selected.torch_device)
            return cls(definition=definition, module=module, device=selected)
        except Exception:
            if selected.name == "cpu":
                raise
            fallback = select_device("cpu", "cpu")
            module = _load_torchscript(definition.torchscript_path, fallback.torch_device)
            return cls(definition=definition, module=module, device=fallback)

    def predict(self, tensor: torch.Tensor) -> tuple[str, float, int]:
        with torch.no_grad():
            logits = self.module(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_id = torch.max(probs, dim=1)
            index = int(pred_id.item())
            label = self.definition.labels[index] if index < len(self.definition.labels) else f"ID_{index}"
            return label, float(confidence.item()), index


def _load_torchscript(path: Path, device: Any) -> torch.jit.ScriptModule:
    module = torch.jit.load(str(path), map_location="cpu")
    module.eval()
    module.to(device)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalido: {path}")
    return data


def _load_labels(labels_path: Path, metadata: dict[str, Any]) -> list[str]:
    if labels_path.exists():
        return [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    class_names = metadata.get("class_names")
    if isinstance(class_names, list):
        return [str(item) for item in class_names]
    raise FileNotFoundError(f"No se encontraron labels para {metadata.get('name')}: {labels_path}")

