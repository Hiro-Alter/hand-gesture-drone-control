import os
import json
from typing import List
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from roi_utils import make_square_bbox
from generar_roi224_dataset import CLASS_NAME, ANNOT_ROOT

IMG_512_ROOT = "dataset/hagridv2_512"
N_SAMPLES = 0
TEST_IMG_ID = "0b63b729-012f-4ef6-9bb0-d23589c63839"


def plot_crop_with_landmarks(crop: Image.Image, lms_px: List[List[float]], title: str):
    arr = np.array(crop)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.title(title)
    plt.axis("off")

    xs = [p[0] for p in lms_px]
    ys = [p[1] for p in lms_px]
    plt.scatter(xs, ys, s=20, c="red")
    plt.tight_layout()
    plt.show()


def plot_full_512_with_landmarks_multi(
    img: Image.Image,
    bboxes: List[List[float]],
    hand_lms_list: List[List[List[float]]],
    labels: List[str],
    class_name: str,
    title: str,
):
    """Muestra la imagen 512p completa con TODOS los bboxes y landmarks.

    - Para cada índice i:
      - bbox[i] se dibuja con un rectángulo.
      - hand_landmarks[i] se dibujan como puntos.
    - Si labels[i] == class_name → rojo; si no → azul.
    """
    W, H = img.size
    arr = np.array(img)

    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.title(title)
    plt.axis("off")

    for i, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            continue
        if i >= len(hand_lms_list):
            continue

        color = "red" if i < len(labels) and labels[i] == class_name else "blue"

        x_norm, y_norm, w_norm, h_norm = bbox
        x = x_norm * W
        y = y_norm * H
        w = w_norm * W
        h = h_norm * H

        # rectángulo del bbox
        rect_x = [x, x + w, x + w, x, x]
        rect_y = [y, y, y + h, y + h, y]
        plt.plot(rect_x, rect_y, color=color, linewidth=2)

        # puntos de landmarks
        lms_norm = hand_lms_list[i]
        xs = [p[0] * W for p in lms_norm]
        ys = [p[1] * H for p in lms_norm]
        plt.scatter(xs, ys, s=20, c=color)

    plt.tight_layout()
    plt.show()


def visualizar_test_id_en_splits():
    """Busca TEST_IMG_ID en train/val/test y lo visualiza donde se encuentre."""
    for split in ["train", "val", "test"]:
        json_path = os.path.join(ANNOT_ROOT, split, f"{CLASS_NAME}.json")
        if not os.path.exists(json_path):
            print(f"[WARN] No existe {json_path}")
            continue

        with open(json_path, "r") as f:
            annotations = json.load(f)

        if TEST_IMG_ID not in annotations:
            print(f"[INFO] {TEST_IMG_ID} no está en {json_path}")
            continue

        print(f"[OK] Encontrado {TEST_IMG_ID} en split '{split}'")
        ann = annotations[TEST_IMG_ID]
        labels = ann.get("labels", [])
        bboxes = ann.get("bboxes", [])          # <- sin coma
        hand_lms_list = ann.get("hand_landmarks", [])

        if not bboxes or not hand_lms_list:
            print(f"[WARN] Falta bbox/landmarks en {TEST_IMG_ID}")
            continue

        # Índices de todas las manos con ese label
        indices_clase = [i for i, lab in enumerate(labels) if lab == CLASS_NAME]
        print(f"  labels: {labels}")
        print(f"  num bboxes: {len(bboxes)}, num hand_landmarks: {len(hand_lms_list)}")
        print(f"  indices_clase (labels == {CLASS_NAME}): {indices_clase}")
        if not indices_clase:
            print(f"[WARN] No hay ninguna mano con label {CLASS_NAME} en {TEST_IMG_ID}")
            continue

        # Cargar imagen 512p una sola vez
        img_dir_512 = os.path.join(IMG_512_ROOT, CLASS_NAME)
        img_name_jpg = f"{TEST_IMG_ID}.jpg"
        img_path = os.path.join(img_dir_512, img_name_jpg)
        if not os.path.exists(img_path):
            img_name_png = f"{TEST_IMG_ID}.png"
            img_path = os.path.join(img_dir_512, img_name_png)
            if not os.path.exists(img_path):
                print(f"[WARN] No imagen 512 para {TEST_IMG_ID}")
                continue

        img = Image.open(img_path)
        W, H = img.size

        # 1) Imagen 512 completa con todos los bboxes/landmarks
        plot_full_512_with_landmarks_multi(
            img,
            bboxes,
            hand_lms_list,
            labels,
            CLASS_NAME,
            title=f"{split} - {TEST_IMG_ID} (512p + todos los bboxes/landmarks)",
        )

        # 2) Dibujar ROI solo para las manos de la clase
        for hl_index in indices_clase:
            if hl_index >= len(bboxes) or hl_index >= len(hand_lms_list):
                print(f"[WARN] Índices inconsistentes en {TEST_IMG_ID}, idx={hl_index}")
                continue

            bbox = bboxes[hl_index]
            if len(bbox) != 4:
                print(f"[WARN] bbox con longitud {len(bbox)} en {TEST_IMG_ID}, idx={hl_index}")
                continue

            print(f"  Usando hl_index={hl_index} -> label='{labels[hl_index]}'")
            hand_lms_norm = hand_lms_list[hl_index]

            x_norm, y_norm, w_norm, h_norm = bbox
            x = x_norm * W
            y = y_norm * H
            w = w_norm * W
            h = h_norm * H

            sq_x, sq_y, sq_side = make_square_bbox(x, y, w, h, W, H)
            if sq_side <= 0:
                print(f"[WARN] ROI vacía en {TEST_IMG_ID} idx={hl_index}")
                continue

            crop = img.crop((sq_x, sq_y, sq_x + sq_side, sq_y + sq_side))

            lms_px_in_crop = []
            for (lx_norm, ly_norm) in hand_lms_norm:
                lx = lx_norm * W
                ly = ly_norm * H
                lx_crop = lx - sq_x
                ly_crop = ly - sq_y
                lms_px_in_crop.append([lx_crop, ly_crop])

            plot_crop_with_landmarks(
                crop,
                lms_px_in_crop,
                title=f"{split} - {TEST_IMG_ID} (ROI idx={hl_index})"
            )

        # Solo mostramos en el primer split donde se encuentre
        return

    print(f"[INFO] {TEST_IMG_ID} no se encontró en train/val/test para la clase {CLASS_NAME}")


def main():
    # Visualización normal aleatoria (opcional, la dejo igual)
    splits = ["train", "val", "test"]

    for split in splits:
        json_path = os.path.join(ANNOT_ROOT, split, f"{CLASS_NAME}.json")
        if not os.path.exists(json_path):
            print(f"[WARN] No existe {json_path}")
            continue

        with open(json_path, "r") as f:
            annotations = json.load(f)

        img_dir_512 = os.path.join(IMG_512_ROOT, CLASS_NAME)
        print(f"\n=== Split: {split} ===")
        img_ids = list(annotations.keys())
        random.shuffle(img_ids)

        shown = 0
        for img_id in img_ids:
            if shown >= N_SAMPLES:
                break

            ann = annotations[img_id]
            labels = ann.get("labels", [])
            bboxes = ann.get("bboxes", [])
            hand_lms_list = ann.get("hand_landmarks", [])

            if not bboxes or not hand_lms_list or CLASS_NAME not in labels:
                continue

            hl_index = labels.index(CLASS_NAME)
            if hl_index >= len(bboxes) or hl_index >= len(hand_lms_list):
                continue

            # Cargar imagen 512p
            img_name_jpg = f"{img_id}.jpg"
            img_path = os.path.join(img_dir_512, img_name_jpg)
            if not os.path.exists(img_path):
                img_name_png = f"{img_id}.png"
                img_path = os.path.join(img_dir_512, img_name_png)
                if not os.path.exists(img_path):
                    print(f"[WARN] No imagen 512 para {img_id}")
                    continue

            img = Image.open(img_path)
            W, H = img.size

            # bbox normalizado -> píxeles (512p)
            x_norm, y_norm, w_norm, h_norm = bboxes[hl_index]
            x = x_norm * W
            y = y_norm * H
            w = w_norm * W
            h = h_norm * H

            # Cuadrar bbox
            sq_x, sq_y, sq_side = make_square_bbox(x, y, w, h, W, H)
            if sq_side <= 0:
                continue

            # Recortar ROI desde 512p (antes del resize a 224)
            crop = img.crop((sq_x, sq_y, sq_x + sq_side, sq_y + sq_side))

            # Landmarks normalizados -> píxeles y luego relativos al crop
            hand_lms_norm = hand_lms_list[hl_index]  # [[x,y], ...] en [0,1]
            lms_px_in_crop = []
            for (lx_norm, ly_norm) in hand_lms_norm:
                lx = lx_norm * W
                ly = ly_norm * H
                lx_crop = lx - sq_x
                ly_crop = ly - sq_y
                lms_px_in_crop.append([lx_crop, ly_crop])

            plot_crop_with_landmarks(
                crop,
                lms_px_in_crop,
                title=f"{split} - {img_id} (ROI desde 512p)"
            )

            shown += 1

    # Visualizar explícitamente TEST_IMG_ID buscando en todos los splits
    print("\n=== Buscando y visualizando TEST_IMG_ID en train/val/test ===")
    visualizar_test_id_en_splits()


if __name__ == "__main__":
    main()