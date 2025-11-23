import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_directml as dml
from torchvision import models

import mediapipe as mp


# =========================
#  RUTAS Y CONFIGURACIÓN
# =========================

# Este archivo está en scripts/prototipo, subimos 2 niveles al root del repo
REPO_ROOT = Path(__file__).resolve().parents[2]

# Rutas del modelo y mapeo de clases
MODEL_PATH = REPO_ROOT / "models" / "outputs_resnet18_directml" / "resnet18_directml_finetuned.pt"
CLASSMAP_PATH = REPO_ROOT / "models" / "outputs_resnet18_directml" / "class_mapping.json"

# Hacemos disponible el módulo de preprocesamiento para importar funciones
PREPROC_DIR = REPO_ROOT / "scripts" / "preprocesamiento"
if str(PREPROC_DIR) not in sys.path:
    sys.path.append(str(PREPROC_DIR))

try:
    import mejorar_imagenes as mi
except Exception as e:
    raise ImportError(f"No se pudo importar 'mejorar_imagenes.py' desde {PREPROC_DIR}: {e}")


# =========================
#  utilidades de clases/modelo
# =========================

def load_class_mapping(path: Path):
    """
    Carga el mapeo de clases de forma robusta.
    Acepta formatos:
    - lista de labels ["palm", "fist", ...]
    - dict id->label {"0": "palm", ...} o {0: "palm", ...}
    - dict label->id {"palm": 0, ...}
    - objeto con claves "idx_to_class" y/o "class_to_idx"
    Devuelve: dict id(int) -> label(str)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # formatos con claves estándar
    if isinstance(data, dict) and ("idx_to_class" in data or "class_to_idx" in data):
        if "idx_to_class" in data and isinstance(data["idx_to_class"], dict):
            return {int(k): str(v) for k, v in data["idx_to_class"].items()}
        if "class_to_idx" in data and isinstance(data["class_to_idx"], dict):
            c2i = {str(k): int(v) for k, v in data["class_to_idx"].items()}
            return {v: k for k, v in c2i.items()}

    # lista simple
    if isinstance(data, list):
        return {i: str(lbl) for i, lbl in enumerate(data)}

    if isinstance(data, dict):
        # detectar si es id->label o label->id
        # casos: claves dígitos => id->label
        if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in data.keys()):
            id_to_label = {}
            for k, v in data.items():
                idx = int(k)
                id_to_label[idx] = str(v)
            return id_to_label
        else:
            # asumimos label->id, invertimos
            label_to_id = {str(k): int(v) for k, v in data.items()}
            return {v: k for k, v in label_to_id.items()}

    raise ValueError("Formato de class_mapping.json no soportado")


def strip_module_prefix(state_dict: dict) -> dict:
    """Elimina prefijo 'module.' si viene de DataParallel."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def build_resnet18(num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    # un state_dict típico es dict[str->Tensor]
    return all(isinstance(k, str) for k in d.keys()) and any(isinstance(v, torch.Tensor) for v in d.values())


def extract_state_dict_from_checkpoint(state: dict):
    """Intenta extraer un state_dict desde múltiples formatos comunes."""
    if state is None:
        return None

    # Caso directo: ya es un state_dict
    if isinstance(state, dict) and _looks_like_state_dict(state):
        return state

    # Formatos comunes
    candidates = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "module",
        "checkpoint",
    ]
    for key in candidates:
        if isinstance(state, dict) and key in state:
            inner = state[key]
            if isinstance(inner, dict) and _looks_like_state_dict(inner):
                return inner
            # algunos guardan anidado aún más
            if isinstance(inner, dict):
                for k2, v2 in inner.items():
                    if isinstance(v2, dict) and _looks_like_state_dict(v2):
                        return v2

    # Búsqueda recursiva superficial
    if isinstance(state, dict):
        for v in state.values():
            if isinstance(v, dict) and _looks_like_state_dict(v):
                return v

    return None


def load_model(model_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Carga el modelo de forma robusta:
    1) intenta cargar como TorchScript (jit) si aplica
    2) intenta cargar state_dict en un resnet18 nuevo
    3) intenta cargar un modelo completo con torch.load
    """
    # 1) TorchScript
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    # 2) state_dict
    try:
        state = torch.load(str(model_path), map_location="cpu")
        sd = extract_state_dict_from_checkpoint(state)
        if sd is not None:
            sd = strip_module_prefix(sd)
            model = build_resnet18(num_classes)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"Aviso: faltan claves en state_dict: {missing}")
            if unexpected:
                print(f"Aviso: claves inesperadas en state_dict: {unexpected}")
            model.to(device).eval()
            return model
    except Exception:
        pass

    # 3) modelo completo
    model = torch.load(str(model_path), map_location=device)
    if isinstance(model, torch.nn.Module):
        model.to(device).eval()
        return model

    raise RuntimeError(f"No se pudo cargar el modelo desde {model_path}")


# =========================
#  Preprocesamiento para arrays (usa funciones de mejorar_imagenes.py)
# =========================

def preprocess_roi_bgr_to_gray224(roi_bgr: np.ndarray) -> np.ndarray:
    """
    - Redimensiona a 224x224
    - Convierte a RGB
    - Aplica pipeline: gris -> contrast_stretch -> gamma -> CLAHE -> unsharp
    Devuelve imagen gris uint8 (224x224)
    """
    roi_resized = cv2.resize(roi_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    gray = mi.to_grayscale(roi_rgb)
    stretched = mi.contrast_stretch(gray)
    gamma_img = mi.gamma_correction(stretched)
    clahe_img = mi.apply_clahe(gamma_img)
    sharp = mi.gaussian_unsharp_paper(clahe_img)
    return sharp


def to_model_tensor_from_gray(gray224: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convierte una imagen gris 224x224 (uint8) a tensor 1x3x224x224 float32 en [0,1]
    Repite el canal a 3 canales para modelos RGB (ResNet).
    """
    if gray224.ndim != 2:
        raise ValueError("Se esperaba imagen en escala de grises 2D")
    img3 = np.stack([gray224, gray224, gray224], axis=2)  # HxWx3
    img3 = img3.astype(np.float32) / 255.0
    chw = np.transpose(img3, (2, 0, 1))  # 3x224x224
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)  # 1x3x224x224
    return tensor


def normalize_imagenet_inplace(x: torch.Tensor):
    """Normaliza in-place un tensor 1x3xHxW a mean/std de ImageNet."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x.sub_(mean).div_(std)


# =========================
#  Detección de mano y recorte ROI con MediaPipe
# =========================

def get_hand_square_roi(frame_bgr: np.ndarray, hands, padding: float = 0.3):
    """
    Usa MediaPipe Hands para detectar la mano principal y devuelve:
    - roi_bgr: recorte cuadrado con margen
    - (x1, y1, x2, y2): coordenadas del recorte para dibujar
    Si no hay mano, devuelve (None, None)
    """
    h, w = frame_bgr.shape[:2]

    # MediaPipe espera RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if not result.multi_hand_landmarks:
        return None, None

    # Tomamos la primera mano
    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Centro y tamaño cuadrado con padding
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    box_w = max_x - min_x
    box_h = max_y - min_y
    side = int(max(box_w, box_h) * (1.0 + padding))
    side = max(side, 1)

    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = x1 + side
    y2 = y1 + side

    # Recortar a los límites de la imagen
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, None

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    return roi_bgr, (x1, y1, x2, y2)


# =========================
#  MAIN LOOP
# =========================

def main():
    # Cargar mapeo clases
    id_to_label = load_class_mapping(CLASSMAP_PATH)
    num_classes = max(id_to_label.keys()) + 1 if id_to_label else 0

    # Configurar dispositivo DirectML
    device = dml.device()
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    model = load_model(MODEL_PATH, num_classes=num_classes, device=device)
    print(f"Modelo cargado desde: {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (índice 0)")

    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Vista espejo para interacción natural
                frame = cv2.flip(frame, 1)

                roi, box = get_hand_square_roi(frame, hands, padding=0.3)
                label_text = "Sin mano"

                if roi is not None:
                    # Preprocesamiento
                    gray224 = preprocess_roi_bgr_to_gray224(roi)
                    tensor = to_model_tensor_from_gray(gray224, device)
                    # Normalización ImageNet (el modelo fue entrenado con esta)
                    normalize_imagenet_inplace(tensor)

                    # Inferencia
                    with torch.no_grad():
                        logits = model(tensor)
                        pred_id = int(torch.argmax(logits, dim=1).item())
                        label_text = id_to_label.get(pred_id, str(pred_id))

                    # Dibujar recuadro
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Mostrar la clase predicha
                cv2.putText(
                    frame,
                    label_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Predicción de Gestos", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
