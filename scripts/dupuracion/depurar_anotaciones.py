import json
from pathlib import Path

import numpy as np

# ==============================
#  PARÁMETROS A AJUSTAR
# ==============================

CLASS_NAME = ["dislike", "like", "stop", "two_up_inverted",
              "two_up", "peace", "peace_inverted", "rock", "palm", "fist"]  # clases a procesar

# Procesar estos splits (puedes dejar solo "train" si quieres)
SPLITS = ["train", "val", "test"]

# Directorios de entrada y salida
IN_ROOT = Path("dataset/annotations_ROI_224")
OUT_ROOT = Path("dataset/annotations_ROI_224_Refined")

# Umbral mínimo de área (normalizada) del bounding box de los landmarks en la ROI
# 0.01 ≈ 1 % del área de la ROI 224x224 -> caja ~22x22 píxeles
MIN_AREA_RATIO = 0.08


# ==============================
#  FUNCIONES AUXILIARES
# ==============================

def refine_annotations_for_split(split: str, class_name: str):
    """
    Lee dataset/annotations_ROI_224/{split}/{class_name}.json,
    limpia las anotaciones y escribe:
      - dataset/annotations_ROI_224_Refined/{split}/{class_name}.json
      - dataset/annotations_ROI_224_Refined/{split}/{class_name}_removed_ids.txt
    """

    in_path = IN_ROOT / split / f"{class_name}.json"
    if not in_path.is_file():
        print(f"[{split}] No se encontró el archivo: {in_path} (se omite)")
        return

    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{class_name}.json"
    log_path = out_dir / f"{class_name}_removed_ids.txt"

    print(f"[{split}] Cargando: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    refined_data = {}
    removed = []  # lista de (id, reason)

    for sample_id, sample in data.items():
        # Eliminar campos que ya no queremos conservar
        sample.pop("hand_landmarks", None)   # eliminar hand_landmarks
        sample.pop("roi_bbox_norm", None)    # eliminar roi_bbox_norm

        # Validar hand_landmarks_roi224
        lms_roi_list = sample.get("hand_landmarks_roi224", None)
        if not lms_roi_list:
            removed.append((sample_id, "missing_hand_landmarks_roi224"))
            continue

        # Por formato HaGRID: una sola mano => lista de longitud 1 con 21 puntos (x,y)
        try:
            lms_arr = np.array(lms_roi_list[0], dtype=float)  # (N,2)
        except Exception as e:
            removed.append((sample_id, f"invalid_hand_landmarks_roi224_parse_error: {e}"))
            continue

        if lms_arr.ndim != 2 or lms_arr.shape[1] != 2:
            removed.append((sample_id, f"invalid_shape_{lms_arr.shape}"))
            continue

        # Chequeo de rango [0,1]
        min_val = float(lms_arr.min())
        max_val = float(lms_arr.max())
        if min_val < 0.0 or max_val > 1.0:
            removed.append(
                (sample_id, f"coords_out_of_range_[min={min_val:.4f},max={max_val:.4f}]")
            )
            continue

        # Bounding box de landmarks en la ROI normalizada
        xs = lms_arr[:, 0]
        ys = lms_arr[:, 1]
        width = float(xs.max() - xs.min())
        height = float(ys.max() - ys.min())
        area = width * height  # área normalizada [0,1]

        if area < MIN_AREA_RATIO:
            removed.append(
                (sample_id, f"too_small_area_{area:.6f}_<_{MIN_AREA_RATIO}")
            )
            continue

        # Si pasa todos los filtros, se conserva el sample "limpio"
        refined_data[sample_id] = sample

    # Guardar JSON refinado
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)

    # Guardar log de ids eliminados
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"TOTAL_IDS_REMOVED: {len(removed)}\n")
        f.write("sample_id\treason\n")
        for sid, reason in removed:
            f.write(f"{sid}\t{reason}\n")

    print(f"[{split}] Samples totales: {total_samples}")
    print(f"[{split}] Samples conservados: {len(refined_data)}")
    print(f"[{split}] Samples eliminados: {len(removed)}")
    print(f"[{split}] JSON refinado guardado en: {out_path}")
    print(f"[{split}] Log de eliminados guardado en: {log_path}")
    print("-" * 60)


# ==============================
#  MAIN
# ==============================

if __name__ == "__main__":
    for split in SPLITS:
        for class_name in CLASS_NAME:
            refine_annotations_for_split(split, class_name)
