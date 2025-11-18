import os
import json
import random
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ================================
#  Parámetros
# ================================
CLASS_NAME = "two_up_inverted"      # Cambia por la clase que quieras visualizar
SPLIT = "train"            # "train", "val" o "test"
NUM_IMGS = 10               # Número de imágenes a mostrar (máx.)

ANN_ROOT = "dataset/annotations_ROI_224"   # JSONs con los landmarks transformados
IMG_ROOT = "dataset/hagridv2_ROI_224_processed"      # Imágenes ROI 224x224

# Si quieres seleccionar siempre las mismas imágenes, fija la semilla:
# random.seed(0)


def load_annotations(class_name: str, split: str) -> Dict[str, Any]:
    json_path = os.path.join(ANN_ROOT, split, f"{class_name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No existe el archivo de anotaciones: {json_path}")

    with open(json_path, "r") as f:
        annotations = json.load(f)

    if not annotations:
        raise ValueError(f"El JSON {json_path} está vacío.")

    print(f"Anotaciones cargadas desde: {json_path}")
    print(f"Total de imágenes anotadas: {len(annotations)}")
    return annotations


def get_image_path(class_name: str, image_name: str) -> str:
    """
    Devuelve la ruta a la imagen ROI 224x224.
    Si no existe con ese nombre, prueba cambiar .jpg <-> .png.
    """
    img_dir = os.path.join(IMG_ROOT, class_name)
    img_path = os.path.join(img_dir, image_name)

    if os.path.exists(img_path):
        return img_path

    # Intentar alternativa .jpg <-> .png
    if image_name.lower().endswith(".jpg"):
        alt = image_name[:-4] + ".png"
    elif image_name.lower().endswith(".png"):
        alt = image_name[:-4] + ".jpg"
    else:
        alt = None

    if alt is not None:
        alt_path = os.path.join(img_dir, alt)
        if os.path.exists(alt_path):
            return alt_path

    raise FileNotFoundError(
        f"No se encontró la imagen '{image_name}' ni su variante en {img_dir}"
    )


def plot_grid(
    annotations: Dict[str, Any],
    class_name: str,
    split: str,
    num_imgs: int = 9,
):
    # Convertir dict en lista para poder muestrear
    items = list(annotations.items())
    total_imgs = len(items)

    num_to_show = min(num_imgs, total_imgs)
    sampled_items = random.sample(items, num_to_show)

    # Calcular filas/columnas de la cuadrícula
    cols = int(np.ceil(np.sqrt(num_to_show)))
    rows = int(np.ceil(num_to_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # aplanar por comodidad

    for ax in axes[num_to_show:]:
        ax.axis("off")  # ocultar ejes sobrantes

    for ax, (img_id, ann) in zip(axes, sampled_items):
        image_name = ann.get("image_name", f"{img_id}.jpg")
        img_path = get_image_path(class_name, image_name)

        img = Image.open(img_path)
        W, H = img.size
        img_np = np.array(img)

        ax.imshow(img_np)
        ax.set_title(f"{split} | {img_id}", fontsize=9)
        ax.axis("off")

        # Landmarks transformados (puede haber varias manos)
        manos_lms: List[List[List[float]]] = ann.get("hand_landmarks_roi224", [])

        for mano in manos_lms:
            xs = [p[0] * W for p in mano]
            ys = [p[1] * H for p in mano]
            ax.scatter(xs, ys, s=10)

    plt.tight_layout()
    plt.show()


def main():
    annotations = load_annotations(CLASS_NAME, SPLIT)
    plot_grid(annotations, CLASS_NAME, SPLIT, NUM_IMGS)


if __name__ == "__main__":
    main()
