import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
#  PARÁMETROS A AJUSTAR
# =====================================

CLASS_NAME = "two_up_inverted"   # carpeta de la clase
SPLIT = "test"                  # "train", "test" o "val"

# JSON por split y clase
JSON_PATH = Path(f"dataset/annotations_ROI_224/{SPLIT}/{CLASS_NAME}.json")

# Carpetas de imágenes
ROI_IMG_ROOT = Path("dataset/hagridv2_ROI_224_processed")  # ROI 224 procesadas
IMG512_ROOT = Path("dataset/hagridv2_512")                 # imágenes 512p originales

# Si quieres un ID específico, ponlo aquí. Si lo dejas en None elige uno aleatorio.
SAMPLE_ID = "a5bd4f77-d7be-439e-a2a8-d60ab4ce6d8a"

RANDOM_SEED = 0


# =====================================
#  FUNCIONES AUXILIARES
# =====================================

def load_annotations(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def draw_landmarks_on_image(img, landmarks_norm,
                            color=(0, 0, 255),
                            radius=3,
                            thickness=-1):
    """
    img: imagen BGR
    landmarks_norm: array (N, 2) con coords normalizadas [0,1] (x,y)
    """
    h, w = img.shape[:2]
    for x_norm, y_norm in landmarks_norm:
        x = int(round(x_norm * w))
        y = int(round(y_norm * h))
        cv2.circle(img, (x, y), radius, color, thickness)


def draw_bbox_norm_on_image(img, bbox_norm,
                            color=(0, 255, 0),
                            thickness=2):
    """
    bbox_norm: [x, y, w, h] normalizado en [0,1] respecto a la imagen completa.
    """
    h, w = img.shape[:2]
    x_norm, y_norm, w_norm, h_norm = bbox_norm
    x1 = int(round(x_norm * w))
    y1 = int(round(y_norm * h))
    x2 = int(round((x_norm + w_norm) * w))
    y2 = int(round((y_norm + h_norm) * h))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


# (OPCIONAL) proyectar landmarks de la ROI 224 a la imagen completa usando roi_bbox_norm
def project_roi_landmarks_to_full(landmarks_roi_norm, bbox_norm):
    """
    landmarks_roi_norm: array (N,2) [0,1] dentro de la ROI 224
    bbox_norm: [bx, by, bw, bh] normalizado en la imagen completa.
    Devuelve landmarks normalizados en coords de imagen completa.
    """
    bx, by, bw, bh = bbox_norm
    # x_full = bx + x_roi * bw;  y_full = by + y_roi * bh
    xs_full = bx + landmarks_roi_norm[:, 0] * bw
    ys_full = by + landmarks_roi_norm[:, 1] * bh
    return np.stack([xs_full, ys_full], axis=1)


# =====================================
#  SCRIPT PRINCIPAL
# =====================================

def main():
    random.seed(RANDOM_SEED)

    # Cargar anotaciones de la clase
    data = load_annotations(JSON_PATH)
    all_ids = list(data.keys())
    if not all_ids:
        raise ValueError(f"El JSON {JSON_PATH} no tiene anotaciones.")

    # Escoger sample
    if SAMPLE_ID is None:
        sample_id = random.choice(all_ids)
    else:
        if SAMPLE_ID not in data:
            raise KeyError(f"El ID {SAMPLE_ID} no está en {JSON_PATH}")
        sample_id = SAMPLE_ID

    sample = data[sample_id]

    print(f"Usando sample_id: {sample_id}")
    print("Labels:", sample.get("labels"))

    # Rutas a las imágenes
    roi_img_path = ROI_IMG_ROOT / CLASS_NAME / f"{sample_id}.jpg"
    img512_path = IMG512_ROOT / CLASS_NAME / f"{sample_id}.jpg"

    if not roi_img_path.is_file():
        raise FileNotFoundError(f"No se encontró la ROI procesada: {roi_img_path}")
    if not img512_path.is_file():
        raise FileNotFoundError(f"No se encontró la imagen 512p: {img512_path}")

    # Leer imágenes
    roi_img = cv2.imread(str(roi_img_path))
    full_img = cv2.imread(str(img512_path))

    if roi_img is None:
        raise RuntimeError(f"cv2 no pudo leer la ROI: {roi_img_path}")
    if full_img is None:
        raise RuntimeError(f"cv2 no pudo leer la imagen 512: {img512_path}")

    # Landmarks dentro de la ROI 224
    if "hand_landmarks_roi224" not in sample:
        raise KeyError(f"El sample {sample_id} no tiene 'hand_landmarks_roi224'")
    hand_lms_roi224 = np.array(sample["hand_landmarks_roi224"][0], dtype=np.float32)  # (21,2)

    # Bounding box normalizado respecto a la imagen 512
    if "roi_bbox_norm" not in sample:
        raise KeyError(f"El sample {sample_id} no tiene 'roi_bbox_norm'")
    roi_bbox_norm = sample["roi_bbox_norm"]

    # Dibujar landmarks en la ROI procesada
    roi_img_draw = roi_img.copy()
    draw_landmarks_on_image(roi_img_draw, hand_lms_roi224, color=(0, 0, 255))

    # Dibujar bbox en la imagen 512
    full_img_draw = full_img.copy()
    draw_bbox_norm_on_image(full_img_draw, roi_bbox_norm, color=(0, 255, 0))

    # (Opcional) también dibujar los landmarks proyectados en la imagen completa:
    lms_full_norm = project_roi_landmarks_to_full(hand_lms_roi224, roi_bbox_norm)
    draw_landmarks_on_image(full_img_draw, lms_full_norm, color=(255, 0, 0))

    # Mostrar resultados
    roi_rgb = cv2.cvtColor(roi_img_draw, cv2.COLOR_BGR2RGB)
    full_rgb = cv2.cvtColor(full_img_draw, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(roi_rgb)
    plt.title(f"ROI 224 - {CLASS_NAME}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(full_rgb)
    plt.title("Imagen 512 con roi_bbox_norm")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
