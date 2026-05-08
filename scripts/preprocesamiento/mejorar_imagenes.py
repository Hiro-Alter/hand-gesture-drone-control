import cv2
import numpy as np
from pathlib import Path

# =========================
#  RUTAS Y PARÁMETROS GLOBALES
# =========================

# Raíces de entrada y salida del dataset
DATASET_IN_ROOT = Path("dataset/hagridv2_ROI_224")
DATASET_OUT_ROOT = Path("dataset/hagridv2_ROI_224_processed")

# Clase a procesar
SELECTED_CLASS = ""   # Ejemplo: "fist", "palm", etc.

# Parámetros del pipeline
GAMMA = 0.75                     # gamma correction (ec. 3)
CLAHE_CLIP_LIMIT = 2.0           # clip limit de CLAHE
CLAHE_TILE_GRID_SIZE = (8, 8)    # tamaño de los tiles para CLAHE

GAUSSIAN_SIGMA = 1.0             # σ del blur gaussiano
GAUSSIAN_KERNEL_SIZE = 5         # kernel 5x5
UNSHARP_LAMBDA = 1.5             # λ del unsharp masking (ec. 4)


# =========================
#  FUNCIONES DE PREPROCESAMIENTO
# =========================

def load_image(path: Path):
    """
    Carga una imagen en BGR (OpenCV) y la pasa a RGB,
    sin cambiar su tamaño original.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No pude leer la imagen: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """Convierte RGB a escala de grises."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return gray


def contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """
    Contrast stretching según ecuación (2):

        NewPixel = (Pixel - min) * 255 / (max - min)
    """
    gray = gray.astype(np.float32)
    min_val = np.min(gray)
    max_val = np.max(gray)

    # Evitar división por cero
    if max_val - min_val < 1e-5:
        return np.zeros_like(gray, dtype=np.uint8)

    stretched = (gray - min_val) * (255.0 / (max_val - min_val))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def gamma_correction(gray: np.ndarray) -> np.ndarray:
    """
    Gamma correction según ecuación (3):

        I_out = 255 * (I_in / 255) ** GAMMA
    """
    gray_f = gray.astype(np.float32) / 255.0
    corrected = np.power(gray_f, GAMMA)
    corrected = np.clip(corrected * 255.0, 0, 255)
    return corrected.astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE sobre la imagen en escala de grises,
    usando los parámetros globales.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)


def gaussian_unsharp_paper(gray: np.ndarray) -> np.ndarray:
    """
    Implementación EXACTA de la ecuación (4) del paper:

        I_sharp = I_blurred + λ (I_blurred - Gσ * I_blurred)
    """
    k = (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE)

    # 1) Primer suavizado: I_blurred
    I_blurred = cv2.GaussianBlur(
        gray, k,
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA
    )

    # 2) Segundo suavizado: Gσ * I_blurred
    G_I_blurred = cv2.GaussianBlur(
        I_blurred, k,
        sigmaX=GAUSSIAN_SIGMA,
        sigmaY=GAUSSIAN_SIGMA
    )

    # 3) Aplicar fórmula
    I_b_f = I_blurred.astype(np.float32)
    G_I_b_f = G_I_blurred.astype(np.float32)

    I_sharp_f = I_b_f + UNSHARP_LAMBDA * (I_b_f - G_I_b_f)
    I_sharp = np.clip(I_sharp_f, 0, 255).astype(np.uint8)

    return I_sharp


def preprocess_image(path_in: Path) -> np.ndarray:
    """
    Aplica todo el pipeline del paper a una imagen
    y devuelve la imagen resultante (sharp, en gris).
    """
    rgb = load_image(path_in)
    gray = to_grayscale(rgb)
    stretched = contrast_stretch(gray)
    gamma_img = gamma_correction(stretched)
    clahe_img = apply_clahe(gamma_img)
    sharp = gaussian_unsharp_paper(clahe_img)
    return sharp


# =========================
#  PROCESAMIENTO POR CARPETA
# =========================

def process_class_folder(class_name: str):
    """
    Procesa todas las imágenes de una clase específica:

    Entrada:  dataset/hagridv2_ROI_224/class_name/*.jpg|.png|...
    Salida:   dataset/hagridv2_ROI_224_processed/class_name/ (mismos nombres)
    """
    in_dir = DATASET_IN_ROOT / class_name
    out_dir = DATASET_OUT_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths = [p for p in in_dir.iterdir()
                   if p.suffix.lower() in valid_exts and p.is_file()]

    print(f"Clase: {class_name}")
    print(f"Imágenes encontradas: {len(image_paths)}")
    print(f"Guardando en: {out_dir}")

    for i, path_in in enumerate(sorted(image_paths), start=1):
        path_out = out_dir / path_in.name

        # Aplicar preprocesamiento
        img_proc = preprocess_image(path_in)

        # Guardar como imagen en escala de grises
        cv2.imwrite(str(path_out), img_proc)

        # Mensaje ligero de progreso
        if i % 100 == 0 or i == len(image_paths):
            print(f"  Procesadas {i}/{len(image_paths)}")


# =========================
#  MAIN
# =========================

if __name__ == "__main__":
    process_class_folder(SELECTED_CLASS)
