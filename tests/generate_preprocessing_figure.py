from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.image_enhancement import (
    GAMMA,
    apply_clahe,
    contrast_stretch,
    gamma_correction,
    gaussian_unsharp,
    to_grayscale,
)


INPUT_IMAGE = ROOT / "tests" / "assets" / "hand_samples" / "0d42fb07-e340-453b-b01d-251ece3906b9 ROI.jpg"
OUTPUT_IMAGE = ROOT / "docs" / "figures" / "preprocesamiento_roi_0d42fb07.png"


def _as_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_preprocessing_states(roi_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    gray = to_grayscale(roi_rgb)
    stretched = contrast_stretch(gray)
    gamma_img = gamma_correction(stretched)
    clahe_img = apply_clahe(gamma_img)
    final_img = gaussian_unsharp(clahe_img)

    return [
        ("ROI original", roi_bgr),
        ("Escala de grises", _as_bgr(gray)),
        ("Expansión de contraste", _as_bgr(stretched)),
        (f"Corrección gamma ({GAMMA:g})", _as_bgr(gamma_img)),
        ("CLAHE", _as_bgr(clahe_img)),
        ("Final: unsharp masking", _as_bgr(final_img)),
    ]


def _fit_to_square(image: np.ndarray, side_px: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(side_px / width, side_px / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((side_px, side_px, 3), 255, dtype=np.uint8)
    y0 = (side_px - new_height) // 2
    x0 = (side_px - new_width) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


from PIL import Image, ImageDraw, ImageFont

# ... (el resto de tus imports y código se mantiene igual)

def _draw_centered_label(canvas: np.ndarray, text: str, center_x: int, baseline_y: int) -> None:
    # 1. Convertir el canvas de BGR (OpenCV) a RGB (Pillow)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # 2. Intentar cargar una fuente del sistema que soporte acentos (Arial es estándar)
    # Si estás en Linux y no encuentra "arial.ttf", puedes cambiarla por "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except IOError:
            font = ImageFont.load_default()

    # 3. Calcular el tamaño del texto para centrarlo correctamente
    # textbbox es el método moderno en Pillow para obtener las dimensiones
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = center_x - text_width // 2
    # OpenCV usa la línea base (abajo), PIL usa la esquina superior izquierda. 
    # Restamos la altura para aproximar la posición original.
    y = baseline_y - text_height 
    
    # 4. Dibujar el texto en RGB
    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    
    # 5. Convertir de vuelta a BGR y mutar el canvas original
    canvas[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def compose_figure(states: list[tuple[str, np.ndarray]]) -> np.ndarray:
    columns = 3
    rows = 2
    panel_side = 480
    label_height = 64
    gap = 36
    margin = 48

    width = margin * 2 + columns * panel_side + (columns - 1) * gap
    height = margin * 2 + rows * (panel_side + label_height) + (rows - 1) * gap
    figure = np.full((height, width, 3), 255, dtype=np.uint8)

    for idx, (label, image) in enumerate(states):
        row = idx // columns
        col = idx % columns
        x = margin + col * (panel_side + gap)
        y = margin + row * (panel_side + label_height + gap)

        _draw_centered_label(figure, label, x + panel_side // 2, y + 38)
        panel = _fit_to_square(image, panel_side)
        panel_y = y + label_height
        figure[panel_y : panel_y + panel_side, x : x + panel_side] = panel
        cv2.rectangle(
            figure,
            (x, panel_y),
            (x + panel_side - 1, panel_y + panel_side - 1),
            (210, 210, 210),
            1,
        )

    return figure


def main() -> None:
    roi_bgr = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if roi_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen ROI: {INPUT_IMAGE}")

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    figure = compose_figure(build_preprocessing_states(roi_bgr))
    if not cv2.imwrite(str(OUTPUT_IMAGE), figure):
        raise OSError(f"No se pudo guardar la figura: {OUTPUT_IMAGE}")
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
