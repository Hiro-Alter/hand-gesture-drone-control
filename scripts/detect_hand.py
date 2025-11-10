import argparse
import os
import cv2
import mediapipe as mp


# --- Argumentos de línea de comandos ---
# Usar por defecto la misma nombre de archivo que había en el script,
# pero ubicado en la misma carpeta donde está este .py
script_dir = os.path.dirname(os.path.abspath(__file__))
_default_filename = os.path.basename(r"WIN_20251110_14_39_03_Pro.jpg")
default_image_path = os.path.join(script_dir, _default_filename)

parser = argparse.ArgumentParser(description="Detectar mano en una imagen usando MediaPipe")
parser.add_argument("--image", "-i", default=default_image_path,
                    help=f"Ruta a la imagen de entrada (por defecto: {default_image_path})")
args = parser.parse_args()
image_path = args.image


if not os.path.exists(image_path):
    raise FileNotFoundError(f"La imagen no existe: {image_path}")

# Leer la imagen (intentar con OpenCV, y si falla usar Pillow como fallback — útil para formatos como AVIF)
image = cv2.imread(image_path)
image_rgb = None
if image is None:
    try:
        from PIL import Image
        import numpy as np

        pil_im = Image.open(image_path).convert("RGB")
        image_rgb = np.array(pil_im)  # RGB
        # Convertir RGB -> BGR para las operaciones de dibujo con OpenCV
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        print(f"Imagen cargada con Pillow (fallback). Formato: {pil_im.format}")
    except Exception as e:
        raise ValueError(f"No se pudo leer la imagen con cv2 ni con Pillow: {e}")
else:
    # si cv2 pudo leer la imagen, preparar la versión RGB para MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Inicializar mediapipe Hands (uso con contexto para cerrar recursos)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:

    # Convertir a RGB (MediaPipe trabaja en RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks de la mano
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener coordenadas normalizadas de la mano
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # Calcular bounding box
            h, w, _ = image.shape
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

            # Dibujar bounding box en la imagen
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


# Mostrar la imagen resultante
cv2.imshow("Resultado", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
