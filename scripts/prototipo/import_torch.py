from pathlib import Path
import json
import torch
from torchvision import models

REPO_ROOT = Path(r"C:\Users\Arley\Documents\GitHub\hand-gesture-drone-control")
CHECKPOINT_PATH = REPO_ROOT / "models" / "outputs_mobilenetv3_small_directml" / "mobilenetv3_small_directml.pt"
CLASSMAP_PATH = REPO_ROOT / "models" / "outputs_mobilenetv3_small_directml" / "class_mapping.json"
OUTPUT_TS_PATH = REPO_ROOT / "models" / "outputs_mobilenetv3_small_directml" / "mobilenetv3_small_android.torchscript.pt"


def load_class_mapping(path: Path) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "idx_to_class" in data:
        return {int(k): str(v) for k, v in data["idx_to_class"].items()}

    if isinstance(data, list):
        return {i: str(lbl) for i, lbl in enumerate(data)}

    if isinstance(data, dict):
        return {int(k): str(v) for k, v in data.items()}

    raise ValueError(f"Formato no soportado en {path}")


def build_mobilenetv3_small(num_classes: int) -> torch.nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    return model


def load_checkpoint_model(checkpoint_path: Path, classmap_path: Path) -> torch.nn.Module:
    class_mapping = load_class_mapping(classmap_path)
    num_classes = len(class_mapping)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Formato no reconocido en {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[key] = value

    model = build_mobilenetv3_small(num_classes)
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    return model


def export_to_torchscript():
    model = load_checkpoint_model(CHECKPOINT_PATH, CLASSMAP_PATH)

    example_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
        traced_model = torch.jit.freeze(traced_model)

    traced_model.save(str(OUTPUT_TS_PATH))
    print(f"Exportado correctamente a: {OUTPUT_TS_PATH}")

    loaded = torch.jit.load(str(OUTPUT_TS_PATH), map_location="cpu")
    loaded.eval()

    with torch.no_grad():
        output = loaded(example_input)

    print("Verificación OK")
    print("Salida:", tuple(output.shape))


if __name__ == "__main__":
    export_to_torchscript()