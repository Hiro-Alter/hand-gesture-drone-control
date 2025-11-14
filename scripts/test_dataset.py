import os
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Opcional: prueba DataLoader + DirectML
try:
    import torch
    import torch_directml
    from torch.utils.data import Dataset, DataLoader
    DML_AVAILABLE = True
except Exception:
    DML_AVAILABLE = False


SUBSET_CLASSES = [
    "dislike", "fist", "palm", "peace", "peace_inverted",
    "point", "rock", "stop", "two_up", "two_up_inverted",
]

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".avif"]


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dataset_dirs() -> Tuple[str, str]:
    root = repo_root()
    ann_dir = os.path.join(root, "dataset", "annotations")
    img_dir = os.path.join(root, "dataset", "hagridv2_512")
    return ann_dir, img_dir


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image(img_dir: str, label: str, image_id: str) -> Optional[str]:
    label_dir = os.path.join(img_dir, label)
    if not os.path.isdir(label_dir):
        return None
    for ext in IMG_EXTS:
        p = os.path.join(label_dir, image_id + ext)
        if os.path.exists(p):
            return p
    # Algunos dumps traen el UUID con extensión incluida; probar tal cual
    p = os.path.join(label_dir, image_id)
    if os.path.exists(p):
        return p
    return None


def norm_box_to_xyxy(box, W: int, H: int, assume_xywh=True) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    if assume_xywh:
        x1 = int(max(0, min(W - 1, x * W)))
        y1 = int(max(0, min(H - 1, y * H)))
        x2 = int(max(0, min(W - 1, (x + w) * W)))
        y2 = int(max(0, min(H - 1, (y + h) * H)))
    else:
        # Interpretar como cx, cy, w, h normalizados
        cx = x * W
        cy = y * H
        ww = w * W
        hh = h * H
        x1 = int(max(0, min(W - 1, cx - ww / 2)))
        y1 = int(max(0, min(H - 1, cy - hh / 2)))
        x2 = int(max(0, min(W - 1, cx + ww / 2)))
        y2 = int(max(0, min(H - 1, cy + hh / 2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_boxes(img_bgr: np.ndarray, boxes: List[List[float]], assume_xywh=True) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    for b in boxes:
        if not (len(b) == 4):
            continue
        x1, y1, x2, y2 = norm_box_to_xyxy(b, W, H, assume_xywh=assume_xywh)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def validate_split(ann_path: str, img_dir: str, allowed_labels: List[str], assume_xywh=True):
    data = load_json(ann_path)
    ok, missing_img, bad_bbox, bad_label = 0, 0, 0, 0
    class_hist = Counter()
    samples_for_viz = []

    for image_id, entry in data.items():
        labels = entry.get("labels", [])
        boxes = entry.get("bboxes", [])
        if len(labels) != len(boxes):
            bad_bbox += 1
            continue

        # Filtrar a subconjunto de clases
        pairs = [(lab, box) for lab, box in zip(labels, boxes) if lab in allowed_labels]
        if not pairs:
            # Saltar imágenes sin clases de interés
            continue

        # Verificar imagen
        # Si hay varias etiquetas, tomar la primera para resolver la carpeta
        first_label = pairs[0][0]
        img_path = find_image(img_dir, first_label, image_id)
        if img_path is None:
            missing_img += 1
            continue

        # Verificar bboxes
        bad_box_flag = False
        for lab, box in pairs:
            if lab not in allowed_labels:
                bad_label += 1
                bad_box_flag = True
                break
            if (len(box) != 4) or any([not (0.0 <= v <= 1.0) for v in box]) or (box[2] <= 0) or (box[3] <= 0):
                bad_bbox += 1
                bad_box_flag = True
                break

        if bad_box_flag:
            continue

        ok += 1
        for lab, _ in pairs:
            class_hist[lab] += 1

        # Guardar algunas muestras para visualización
        if len(samples_for_viz) < 24:
            samples_for_viz.append((img_path, [b for _, b in pairs]))

    summary = {
        "ok_items": ok,
        "missing_images": missing_img,
        "bad_bboxes": bad_bbox,
        "bad_labels": bad_label,
        "class_hist": dict(class_hist),
        "samples_for_viz": samples_for_viz,
    }
    return summary


class MiniHagridDataset(Dataset):
    def __init__(self, items: List[Tuple[str, List[List[float]]]], assume_xywh=True, resize=512):
        self.items = items
        self.assume_xywh = assume_xywh
        self.size = resize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, boxes = self.items[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
        return {
            "image": torch.from_numpy(img) if DML_AVAILABLE else img,
            "boxes": boxes,
            "path": img_path,
        }


def grid_visualization(samples: List[Tuple[str, List[List[float]]]], out_path: str, assume_xywh=True, cols=6):
    tiles = []
    for img_path, boxes in samples:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: 
            continue
        img = draw_boxes(img, boxes, assume_xywh=assume_xywh)
        img = cv2.resize(img, (384, 384))
        tiles.append(img)
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    H, W = tiles[0].shape[:2]
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = tile
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assume_xywh", action="store_true", default=True,
                        help="Interpretar bbox como xywh normalizado (top-left).")
    parser.add_argument("--assume_cxcywh", action="store_true", default=False,
                        help="Si se pasa, interpretar bbox como cx,cy,w,h.")
    parser.add_argument("--viz", action="store_true", help="Guardar rejilla de muestras con bboxes.")
    parser.add_argument("--max_items", type=int, default=None, help="Limitar items por split para debug.")
    args = parser.parse_args()

    assume_xywh = not args.assume_cxcywh

    ann_dir, img_dir = dataset_dirs()
    splits = ["train", "val", "test"]

    global_hist = Counter()
    for split in splits:
        split_dir = os.path.join(ann_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split no encontrado: {split_dir}")
            continue

        summaries = []
        json_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
        json_files.sort()
        if args.max_items:
            json_files = json_files[:args.max_items]

        all_samples = []
        for jf in json_files:
            ann_path = os.path.join(split_dir, jf)
            summary = validate_split(ann_path, img_dir, SUBSET_CLASSES, assume_xywh=assume_xywh)
            summaries.append((jf, summary))
            for k, v in summary["class_hist"].items():
                global_hist[k] += v
            all_samples.extend(summary["samples_for_viz"])

        print(f"\n=== Split: {split} ===")
        total_ok = sum(s["ok_items"] for _, s in summaries)
        total_missing = sum(s["missing_images"] for _, s in summaries)
        total_bad_bbox = sum(s["bad_bboxes"] for _, s in summaries)
        total_bad_label = sum(s["bad_labels"] for _, s in summaries)
        print(f"OK: {total_ok} | missing_img: {total_missing} | bad_bbox: {total_bad_bbox} | bad_label: {total_bad_label}")

        # Top clases del split
        split_hist = Counter()
        for _, s in summaries:
            split_hist.update(s["class_hist"])
        if split_hist:
            print("Distribución (top 10):", split_hist.most_common(10))

        if args.viz and all_samples:
            out_img = os.path.join(repo_root(), f"_debug_{split}.jpg")
            grid_visualization(all_samples[:24], out_img, assume_xywh=assume_xywh)
            print(f"Guardada visualización: {out_img}")

        # DirectML sanity (mini)
        if DML_AVAILABLE and total_ok > 0:
            device = torch_directml.device()
            ds = MiniHagridDataset(all_samples[:32], assume_xywh=assume_xywh, resize=512)
            dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            try:
                batch = next(iter(dl))
                imgs = batch["image"].to(device)
                print(f"DirectML batch OK -> {imgs.shape} en {device}")
            except Exception as e:
                print(f"[DirectML] Error moviendo batch al dispositivo: {e}")

    print("\n=== Totales (todas las clases del subconjunto) ===")
    if global_hist:
        print(global_hist.most_common())


if __name__ == "__main__":
    main()