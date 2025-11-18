import json
import os
from PIL import Image

CLASS_NAME = "stop"   # Cambia esto para procesar otra clase

ANNOT_ROOT = "dataset/annotations"
IMG_DIR_ROOT = "dataset/hagridv2"
OUT_ROOT = f"dataset/hagridv2_ROI_224/{CLASS_NAME}"

TARGET_SIZE = 224  # Resize final

os.makedirs(OUT_ROOT, exist_ok=True)


# ================================
#  Hacer cuadrado un bbox
# ================================
def make_square_bbox(x, y, w, h, img_w, img_h):
    side = max(w, h)

    new_x = x - (side - w) / 2
    new_y = y - (side - h) / 2

    new_x = max(0, new_x)
    new_y = max(0, new_y)

    if new_x + side > img_w:
        side = img_w - new_x
    if new_y + side > img_h:
        side = img_h - new_y

    return int(new_x), int(new_y), int(side)


# ================================
#  Procesar un único JSON
# ================================
def procesar_json(json_path):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    print(f"\nProcesando JSON: {json_path}")
    print(f"Total anotaciones: {len(annotations)}")

    img_dir = f"{IMG_DIR_ROOT}/{CLASS_NAME}"

    for idx, (img_id, ann) in enumerate(annotations.items()):

        # Nombre de imagen
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(img_dir, img_name)

        # Chequear si existe .jpg o .png
        if not os.path.exists(img_path):
            alt = img_path.replace(".jpg", ".png")
            if os.path.exists(alt):
                img_path = alt
            else:
                print(f"[WARN] No se encontró la imagen {img_name}")
                continue

        img = Image.open(img_path)
        W, H = img.size

        # Buscar el bbox correspondiente a la clase
        target_bbox = None
        for bbox, label in zip(ann["bboxes"], ann["labels"]):
            if label == CLASS_NAME:
                target_bbox = bbox
                break

        if target_bbox is None:
            print(f"[WARN] No bbox para clase {CLASS_NAME} en {img_id}")
            continue

        # Normalizado → pixeles
        x_norm, y_norm, w_norm, h_norm = target_bbox
        x = x_norm * W
        y = y_norm * H
        w = w_norm * W
        h = h_norm * H

        # Cuadrar bbox
        sq_x, sq_y, sq_side = make_square_bbox(x, y, w, h, W, H)
        if sq_side <= 0:
            print(f"[ERROR] side=0 en {img_id}")
            continue

        # Recortar
        crop = img.crop((sq_x, sq_y, sq_x + sq_side, sq_y + sq_side))

        if crop.size[0] == 0 or crop.size[1] == 0:
            print(f"[ERROR] ROI vacío en {img_id}")
            continue

        # Resize 224x224
        crop = crop.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # Guardar (todas juntas)
        out_path = os.path.join(OUT_ROOT, img_name)
        crop.save(out_path)

        if idx % 500 == 0:
            print(f"{idx} procesadas...")

    print(f"✔ JSON completado: {json_path}")


# ================================
#  Ejecutar para train + val + test
# ================================
def main():
    splits = ["train", "val", "test"]

    for split in splits:
        json_path = f"{ANNOT_ROOT}/{split}/{CLASS_NAME}.json"
        if os.path.exists(json_path):
            procesar_json(json_path)
        else:
            print(f"[WARN] No existe {json_path}")

    print("\nPROCESO COMPLETO. Imágenes guardadas en:")
    print(f"   {OUT_ROOT}\n")


if __name__ == "__main__":
    main()