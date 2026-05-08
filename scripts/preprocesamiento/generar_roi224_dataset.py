from typing import Dict, Any, List, Tuple
import os
import json

from PIL import Image

from roi_utils import make_square_bbox  # lógica reutilizable de recorte

# ================================
#  Parámetros
# ================================
CLASS_NAME = "peace"   # clase a procesar

ANNOT_ROOT = "dataset/annotations"              # JSON original (HaGRID)
IMG_ORIG_ROOT = "dataset/hagridv2"             # imágenes originales por clase (NO divididas por split)
OUT_IMG_ROOT = "dataset/hagridv2_ROI_224"      # recortes 224x224 (por clase)
OUT_ANN_ROOT = "dataset/annotations_ROI_224"   # salida de nuevos json

TARGET_SIZE = 224

os.makedirs(OUT_ANN_ROOT, exist_ok=True)
os.makedirs(OUT_IMG_ROOT, exist_ok=True)


# ================================
#  Transformación usando original → crop → 224
# ================================
def transform_hand_landmarks_via_orig(
    hand_landmarks_norm: List[List[float]],
    bbox_norm: List[float],
    img_orig: Image.Image,
) -> Tuple[List[List[float]], List[float], Image.Image]:
    """
    Usa la imagen ORIGINAL para:
      1) recrear el crop cuadrado en píxeles (como en segmentar_ROI)
      2) recortar y hacer resize a 224x224
      3) proyectar landmarks al espacio 224 y normalizar en [0,1]

    Devuelve:
      - new_hand_lms_norm_roi224: landmarks normalizados en 224x224
      - roi_bbox_norm: [sq_x_norm, sq_y_norm, sq_side_norm, sq_side_norm] respecto a la imagen original
      - crop_224: imagen recortada y reescalada a 224x224
    """
    W, H = img_orig.size

    # bbox normalizado -> píxeles
    x_norm, y_norm, w_norm, h_norm = bbox_norm
    x = x_norm * W
    y = y_norm * H
    w = w_norm * W
    h = h_norm * H

    # Cuadrar bbox
    sq_x, sq_y, sq_side = make_square_bbox(x, y, w, h, W, H)
    if sq_side <= 0:
        raise ValueError("sq_side <= 0 al hacer make_square_bbox")

    # Recorte desde original (ROI antes de resize)
    crop = img_orig.crop((sq_x, sq_y, sq_x + sq_side, sq_y + sq_side))

    if crop.size[0] == 0 or crop.size[1] == 0:
        raise ValueError("ROI vacío")

    # Resize a 224x224
    crop_224 = crop.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    # Proyectar landmarks normalizados (imagen original) al espacio 224
    new_hand_lms_norm_roi224: List[List[float]] = []
    for (lx_norm, ly_norm) in hand_landmarks_norm:
        lx_orig = lx_norm * W
        ly_orig = ly_norm * H

        lx_crop = lx_orig - sq_x
        ly_crop = ly_orig - sq_y

        lx_224 = (lx_crop / sq_side) * TARGET_SIZE
        ly_224 = (ly_crop / sq_side) * TARGET_SIZE

        lx_224_norm = lx_224 / TARGET_SIZE
        ly_224_norm = ly_224 / TARGET_SIZE

        new_hand_lms_norm_roi224.append([lx_224_norm, ly_224_norm])

    # ROI en normalizado respecto a la imagen original
    sq_x_norm = sq_x / W
    sq_y_norm = sq_y / H
    sq_side_norm = sq_side / max(W, H)

    roi_bbox_norm = [sq_x_norm, sq_y_norm, sq_side_norm, sq_side_norm]

    return new_hand_lms_norm_roi224, roi_bbox_norm, crop_224


# ================================
#  Main
# ================================
def main():
    splits = ["train", "val", "test"]

    for split in splits:
        json_path = os.path.join(ANNOT_ROOT, split, f"{CLASS_NAME}.json")
        if not os.path.exists(json_path):
            print(f"[WARN] No existe {json_path}")
            continue

        with open(json_path, "r") as f:
            annotations = json.load(f)

        print(f"\n=== Procesando split: {split} ===")
        print(f"Total anotaciones originales: {len(annotations)}")

        out_split_dir = os.path.join(OUT_ANN_ROOT, split)
        os.makedirs(out_split_dir, exist_ok=True)
        out_json_path = os.path.join(out_split_dir, f"{CLASS_NAME}.json")

        # Carpeta de imágenes originales y recortes para esta clase
        img_dir_orig = os.path.join(IMG_ORIG_ROOT, CLASS_NAME)
        out_img_dir = os.path.join(OUT_IMG_ROOT, CLASS_NAME)
        os.makedirs(out_img_dir, exist_ok=True)

        new_annotations: Dict[str, Any] = {}

        for img_id, ann in annotations.items():
            labels = ann.get("labels", [])
            bboxes = ann.get("bboxes", [])
            hand_lms_list = ann.get("hand_landmarks", [])
            user_id = ann.get("user_id")

            if not bboxes or not hand_lms_list or CLASS_NAME not in labels:
                continue

            hl_index = labels.index(CLASS_NAME)
            if hl_index >= len(bboxes) or hl_index >= len(hand_lms_list):
                continue

            bbox_norm = bboxes[hl_index]
            hand_lms_norm = hand_lms_list[hl_index]

            # Cargar imagen ORIGINAL (mismo id y clase; sin split)
            img_name_jpg = f"{img_id}.jpg"
            img_path = os.path.join(img_dir_orig, img_name_jpg)
            if not os.path.exists(img_path):
                img_name_png = f"{img_id}.png"
                img_path = os.path.join(img_dir_orig, img_name_png)
                if not os.path.exists(img_path):
                    print(f"[WARN] No hay imagen original para {img_id}. Busqué:")
                    print(f"       {os.path.join(img_dir_orig, img_id + '.jpg')}")
                    print(f"       {os.path.join(img_dir_orig, img_id + '.png')}")
                    continue

            img_orig = Image.open(img_path)

            try:
                new_hand_lms_norm_roi224, roi_bbox_norm, crop_224 = transform_hand_landmarks_via_orig(
                    hand_lms_norm, bbox_norm, img_orig
                )
            except ValueError:
                continue

            # Guardar recorte 224x224 (se sobreescribe si ya existía para este id)
            out_img_path = os.path.join(out_img_dir, f"{img_id}.jpg")
            crop_224.save(out_img_path)

            # Construir anotación filtrada
            filtered_ann = {
                "user_id": user_id,
                "labels": [CLASS_NAME],
                "hand_landmarks": [hand_lms_norm],
                "hand_landmarks_roi224": [new_hand_lms_norm_roi224],
                "roi_bbox_norm": roi_bbox_norm,
            }

            new_annotations[img_id] = filtered_ann

        with open(out_json_path, "w") as f_out:
            json.dump(new_annotations, f_out, indent=2)

        print(f"Anotaciones nuevas guardadas en: {out_json_path}")
        print(f"Total anotaciones nuevas: {len(new_annotations)}")


if __name__ == "__main__":
    main()
