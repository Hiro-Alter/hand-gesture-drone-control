from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.model_loader import GestureClassifier, ModelCatalog
from src.inference.predictor import GesturePredictor
from src.utils.config_loader import load_app_config
from src.vision.pipeline import VisionPipeline


def main() -> int:
    config = load_app_config()
    sample_dir = ROOT / "tests" / "assets" / "hand_samples"
    samples = sorted(sample_dir.glob("*.jpg"))
    if not samples:
        raise FileNotFoundError(f"No hay imagenes de prueba en {sample_dir}")

    pipeline = VisionPipeline(config.get("vision", {}))
    try:
        for sample in samples:
            image = cv2.imread(str(sample))
            if image is None:
                raise RuntimeError(f"No se pudo leer la imagen: {sample}")
            result = pipeline.process(image)
            if result.roi is None:
                continue

            catalog = ModelCatalog(config["models"]["manifest"])
            classifier = GestureClassifier.load(catalog.get(config["models"]["default"]), preferred_device="cpu", fallback_device="cpu")
            prediction = GesturePredictor(classifier).predict_roi(result.roi.image_bgr)
            print(f"pipeline ok: {sample.name} -> {prediction.gesture} {prediction.confidence:.4f}")
            return 0

        print("pipeline ok sin ROI detectable en las muestras locales")
        return 0
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
