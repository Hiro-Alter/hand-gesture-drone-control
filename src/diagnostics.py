from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from src.commands.gesture_command_mapper import GestureCommandMapper
from src.inference.model_loader import GestureClassifier, ModelCatalog
from src.inference.predictor import GesturePredictor
from src.utils.config_loader import load_app_config
from src.utils.paths import PROJECT_ROOT
from src.vision.camera import list_available_cameras
from src.vision.pipeline import VisionPipeline


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def run_diagnostics() -> list[CheckResult]:
    config = load_app_config()
    results: list[CheckResult] = []

    results.append(
        CheckResult(
            "Python",
            "OK" if sys.version_info >= (3, 10) else "ERROR",
            sys.version.split()[0],
        )
    )
    results.extend(_check_imports())
    results.append(_check_config(config))
    results.extend(_check_models(config))
    results.append(_check_gesture_mapping(config))
    results.append(_check_camera_probe(config))
    results.append(_check_pipeline_sample(config))
    return results


def print_results(results: list[CheckResult]) -> None:
    print("Diagnostico de entorno y aplicacion")
    print("=" * 40)
    for result in results:
        marker = "[OK]" if result.status == "OK" else "[WARN]" if result.status == "WARN" else "[ERROR]"
        required = "requerido" if result.required else "opcional"
        print(f"{marker} {result.name} ({required}): {result.detail}")

    errors = [item for item in results if item.required and item.status == "ERROR"]
    warnings = [item for item in results if item.status == "WARN"]
    print("=" * 40)
    print(f"Errores requeridos: {len(errors)}")
    print(f"Advertencias: {len(warnings)}")


def exit_code(results: list[CheckResult]) -> int:
    return 1 if any(item.required and item.status == "ERROR" for item in results) else 0


def _check_imports() -> list[CheckResult]:
    modules = [
        ("OpenCV", "cv2", True),
        ("MediaPipe", "mediapipe", True),
        ("PyTorch", "torch", True),
        ("Torch DirectML", "torch_directml", False),
        ("PySide6", "PySide6", True),
        ("AirSim", "airsim", False),
    ]
    results: list[CheckResult] = []
    for label, module_name, required in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "instalado")
            results.append(CheckResult(label, "OK", str(version), required=required))
        except Exception as exc:
            status = "ERROR" if required else "WARN"
            results.append(CheckResult(label, status, str(exc), required=required))
    return results


def _check_config(config: dict) -> CheckResult:
    required_keys = ["camera", "vision", "models", "inference", "commands", "logs"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        return CheckResult("Configuracion", "ERROR", f"Faltan claves: {', '.join(missing)}")
    return CheckResult("Configuracion", "OK", "config/app_config.json cargado")


def _check_models(config: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        catalog = ModelCatalog(config["models"]["manifest"])
    except Exception as exc:
        return [CheckResult("Catalogo de modelos", "ERROR", str(exc))]

    expected = {"resnet18", "mobilenetv3_small"}
    missing = expected.difference(catalog.model_names)
    if missing:
        results.append(CheckResult("Catalogo de modelos", "ERROR", f"Faltan: {', '.join(sorted(missing))}"))
        return results
    results.append(CheckResult("Catalogo de modelos", "OK", ", ".join(catalog.model_names)))

    for model_name in catalog.model_names:
        try:
            start = perf_counter()
            classifier = GestureClassifier.load(
                catalog.get(model_name),
                preferred_device="cpu",
                fallback_device="cpu",
            )
            elapsed_ms = (perf_counter() - start) * 1000.0
            results.append(
                CheckResult(
                    f"Modelo {model_name}",
                    "OK",
                    f"carga CPU en {elapsed_ms:.1f} ms, clases={len(classifier.definition.labels)}",
                )
            )
        except Exception as exc:
            results.append(CheckResult(f"Modelo {model_name}", "ERROR", str(exc)))
    return results


def _check_gesture_mapping(config: dict) -> CheckResult:
    try:
        mapper = GestureCommandMapper(config["commands"]["mapping"])
        like = mapper.from_gesture("like")
        stop = mapper.from_gesture("stop")
        if like is None or like.command != "forward":
            return CheckResult("Mapeo gesto-comando", "ERROR", "like no apunta a forward")
        if stop is None or stop.command != "hover":
            return CheckResult("Mapeo gesto-comando", "ERROR", "stop no apunta a hover")
        return CheckResult("Mapeo gesto-comando", "OK", "gestos principales configurados")
    except Exception as exc:
        return CheckResult("Mapeo gesto-comando", "ERROR", str(exc))


def _check_camera_probe(config: dict) -> CheckResult:
    probe_count = int(config.get("camera", {}).get("probe_count", 5))
    try:
        cameras = list_available_cameras(probe_count)
    except Exception as exc:
        return CheckResult("Camaras", "WARN", str(exc), required=False)
    if not cameras:
        return CheckResult("Camaras", "WARN", "no se detectaron camaras", required=False)
    labels = ", ".join(camera.label for camera in cameras)
    return CheckResult("Camaras", "OK", labels, required=False)


def _check_pipeline_sample(config: dict) -> CheckResult:
    sample_dir = PROJECT_ROOT / "tests" / "assets" / "hand_samples"
    samples = sorted(sample_dir.glob("*.jpg"))
    if not samples:
        return CheckResult("Pipeline con muestra", "WARN", "no hay imagenes locales", required=False)

    pipeline = VisionPipeline(config.get("vision", {}))
    try:
        for sample in samples:
            image = cv2.imread(str(sample))
            if image is None:
                continue
            result = pipeline.process(image)
            if result.roi is None:
                continue
            catalog = ModelCatalog(config["models"]["manifest"])
            classifier = GestureClassifier.load(
                catalog.get(config["models"]["default"]),
                preferred_device="cpu",
                fallback_device="cpu",
            )
            prediction = GesturePredictor(classifier).predict_roi(result.roi.image_bgr)
            return CheckResult(
                "Pipeline con muestra",
                "OK",
                f"{sample.name}: {prediction.gesture} ({prediction.confidence:.2%})",
            )
        return CheckResult(
            "Pipeline con muestra",
            "WARN",
            "imagenes leidas, pero sin ROI detectable",
            required=False,
        )
    except Exception as exc:
        return CheckResult("Pipeline con muestra", "ERROR", str(exc))
    finally:
        pipeline.close()


def main() -> int:
    results = run_diagnostics()
    print_results(results)
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

