import os
import json
from typing import List
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from segmentar_ROI import make_square_bbox   # reutilizamos la misma lógica
from recalcular_anotaciones import CLASS_NAME, ANNOT_ROOT

IMG_512_ROOT = "dataset/hagridv2_512"   # ajusta esto al path real
N_SAMPLES = 1
TEST_IMG_ID = "5fa260f8-a3be-4f8b-9851-376de8b74c43"


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


def plot_full_512_with_landmarks(img: Image.Image, lms_norm: List[List[float]], title: str):
    """Muestra la imagen 512p completa con los landmarks del JSON (normalizados)."""
    W, H = img.size
    arr = np.array(img)

    xs = [p[0] * W for p in lms_norm]
    ys = [p[1] * H for p in lms_norm]

    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.title(title)
    plt.axis("off")
    plt.scatter(xs, ys, s=20, c="cyan")
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
        bboxes = ann.get("bboxes", []),
        hand_lms_list = ann.get("hand_landmarks", [])

        if not bboxes or not hand_lms_list or CLASS_NAME not in labels:
            print(f"[WARN] Falta bbox/landmarks o label {CLASS_NAME} en {TEST_IMG_ID}")
            continue

        hl_index = labels.index(CLASS_NAME)
        if hl_index >= len(bboxes) or hl_index >= len(hand_lms_list):
            print(f"[WARN] Índices inconsistentes en {TEST_IMG_ID}")
            continue

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

        hand_lms_norm = hand_lms_list[hl_index]

        plot_full_512_with_landmarks(
            img,
            hand_lms_norm,
            title=f"{split} - {TEST_IMG_ID} (512p completa + landmarks)"
        )

        x_norm, y_norm, w_norm, h_norm = bboxes[hl_index]
        x = x_norm * W
        y = y_norm * H
        w = w_norm * W
        h = h_norm * H

        sq_x, sq_y, sq_side = make_square_bbox(x, y, w, h, W, H)
        if sq_side <= 0:
            print(f"[WARN] ROI vacía en {TEST_IMG_ID}")
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
            title=f"{split} - {TEST_IMG_ID} (ROI desde 512p)"
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

            # Cuadrar bbox EXACTAMENTE como en segmentar_ROI
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