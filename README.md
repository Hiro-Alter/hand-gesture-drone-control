# Hand Gesture Drone Control (Prototipo)

Proyecto de reconocimiento de gestos de mano orientado a controlar un dron mediante eventos enviados por WebSocket. El estado actual implementa un pipeline de inferencia en tiempo real usando cámara + MediaPipe (detección/landmarks) + un clasificador (ResNet18 en PyTorch) acelerado con DirectML en Windows.

## Estado actual
- Prototipo end-to-end: captura de cámara → detección de mano → recorte ROI → preprocesamiento → clasificación → estabilización temporal → envío por WebSocket.
- Servidor WebSocket simple con broadcast para que Unity (u otro cliente) reciba mensajes.
- Scripts auxiliares para preparar/validar anotaciones y visualizar bboxes/landmarks (enfocados a flujos con HaGRIDv2), sin entrar en el contenido de dataset/.

## Arquitectura
1) Runtime (inferencia en tiempo real)
- Captura de video con OpenCV.
- Detección de mano con MediaPipe Hands.
- Extracción de ROI cuadrada (con padding configurable).
- Preprocesamiento (contrast stretch + gamma + CLAHE + unsharp).
- Inferencia con un modelo de clasificación (ResNet18) sobre ROI 224×224.
- "Debounce"/estabilización del gesto para evitar flicker.
- Envío de JSON vía WebSocket a un servidor local (broadcast).

2) Infra de datos (utilidades)
- Scripts de recorte a ROI 224 y transformación/limpieza de anotaciones.
- Scripts de debugging/visualización de landmarks y bboxes.

## Estructura del repositorio
- models/
  - outputs_resnet18_directml/: pesos .pt + class_mapping.json (modelo principal del runtime actual)
  - outputs_mobilenetv3_small_directml/: pesos .pt + class_mapping.json (artefacto alternativo)
  - outputs_mlp_keras/: best_mlp_landmarks.keras (artefacto alternativo)
  - README_HaGRIDv2_Resumen.md: resumen del dataset (referencial)
- scripts/
  - prototipo/
    - camara_prediccion.py: pipeline en tiempo real + envío por WS
    - gesture_ws_server.py: servidor WS (broadcast)
  - preprocesamiento/
    - segmentar_ROI.py: recorta ROI 224 a partir de anotaciones
    - recalcular_anotaciones.py: recalcula landmarks al espacio ROI 224
    - mejorar_imagenes.py: pipeline de mejora de imagen (usado también en runtime)
    - debug_landmarks_512.py: visualización/depuración de landmarks vs imagen 512
  - dupuracion/
    - depurar_anotaciones.py: filtra/limpia anotaciones ROI (por heurística de área)
  - test/
    - detect_hand.py: prueba rápida de detección de mano en imagen con MediaPipe
    - test_dataset.py: validación de JSON + existencia de imágenes + mini sanity DirectML
    - test_anotaciones.py: visualiza landmarks ROI y proyección a imagen 512
    - test.py: visualiza grid de landmarks en ROI
- start_gesture_system.bat: arranque (activa venv, levanta WS server, ejecuta cámara)
- requirements.txt: dependencias (incluye PyTorch + DirectML + MediaPipe + OpenCV, etc.)
- LICENSE: MIT

## Modelos y clases
Clases actuales (según class_mapping.json en models/outputs_resnet18_directml/):
- dislike, fist, like, palm, peace, peace_inverted, rock, stop, two_up, two_up_inverted

Archivos relevantes:
- models/outputs_resnet18_directml/resnet18_directml_finetuned.pt
- models/outputs_resnet18_directml/class_mapping.json

Notas:
- El runtime usa ResNet18 sin pesos preentrenados (weights=None) y reemplaza la capa final (fc) por num_classes.
- El loader soporta distintos formatos de checkpoint: state_dict directo o dict con model_state/state_dict.

## Protocolo WebSocket (runtime)
Servidor: ws://127.0.0.1:8765

Mensaje (JSON) enviado desde el cliente de cámara:
- gesture: etiqueta de la clase (string)
- ts: timestamp Unix (float)

Ejemplo:
```json
{
  "gesture": "palm",
  "ts": 1730000000.123
}
```

El servidor hace broadcast: reenvía mensajes recibidos a todos los clientes conectados (excepto el emisor).

Archivos:
- scripts/prototipo/gesture_ws_server.py
- scripts/prototipo/camara_prediccion.py

## Pipeline de inferencia en tiempo real
Implementado en scripts/prototipo/camara_prediccion.py:

1) Captura y UI
- cv2.VideoCapture(0)
- Flip horizontal (mirror) para UX.

2) Detección de mano y ROI
- MediaPipe Hands detecta landmarks.
- Se calcula un bounding box cuadrado centrado y se aplica padding (HAND_PADDING).
- Se recorta roi_bgr del frame original.

3) Preprocesamiento (ROI)
- Resize a 224×224.
- BGR→RGB.
- Conversión a gris y mejora:
  - contrast_stretch
  - gamma_correction
  - CLAHE
  - gaussian_unsharp_paper
- Se replica el canal gris a 3 canales (para input 3×224×224).
- Normalización ImageNet (mean/std).

4) Inferencia
- Softmax → predicción argmax.
- Umbral de confianza CONFIDENCE_THRESHOLD.

5) Estabilización temporal (anti-flicker)
- Se exige consistencia por frames: GESTURE_STABILITY_COUNT
- y además permanencia mínima: GESTURE_STABILITY_TIME
- Se limita la tasa de envío: MIN_SEND_INTERVAL
- Solo se envía si el gesto es estable y supera el threshold.

## Requisitos del entorno
- Windows (recomendado si se usa torch-directml para acelerar en GPU vía DirectML).
- Python 3.10.
- Cámara accesible por OpenCV.
- (Opcional) GPU con drivers compatibles con DirectML.

Dependencias: ver requirements.txt.

## Instalación (venv)
El .bat asume un entorno virtual en .venv

1) Crear venv:
```bash
py -3.11 -m venv .venv
```

2) Activar:
```bash
.venv\Scripts\activate
```

3) Instalar:
```bash
pip install -r requirements.txt
```

## Ejecución
Opción A: con el script de arranque
- start_gesture_system.bat

Qué hace:
- activa .venv
- inicia el servidor WS en una ventana nueva
- espera 2s
- ejecuta el cliente de cámara
- al cerrar la cámara intenta cerrar la ventana del servidor

Opción B: manual
1) Servidor:
```bash
python scripts/prototipo/gesture_ws_server.py
```

2) Cliente cámara:
```bash
python scripts/prototipo/camara_prediccion.py
```

## Configuración rápida (parámetros importantes)
En scripts/prototipo/camara_prediccion.py:
- WEBSOCKET_URI: endpoint WS
- CONFIDENCE_THRESHOLD: umbral de confianza de predicción
- MIN_SEND_INTERVAL: límite de frecuencia de envío
- GESTURE_STABILITY_TIME / GESTURE_STABILITY_COUNT: estabilidad requerida
- HAND_PADDING / HAND_DETECTION_CONFIDENCE / HAND_TRACKING_CONFIDENCE: MediaPipe + ROI

## Scripts
Preprocesamiento / utilidades (no documenta dataset/, solo propósito):
- scripts/preprocesamiento/segmentar_ROI.py
  - Genera recortes ROI 224×224 a partir de bboxes anotados.
- scripts/preprocesamiento/recalcular_anotaciones.py
  - Proyecta landmarks del espacio "imagen original" al espacio "ROI 224" y guarda un JSON nuevo.
- scripts/preprocesamiento/mejorar_imagenes.py
  - Implementa el pipeline de mejora de contraste usado tanto para preparar imágenes como para el runtime.
- scripts/preprocesamiento/debug_landmarks_512.py
  - Visualiza sobre imagen 512: bboxes y landmarks; y también el recorte ROI con landmarks relativos (debug).

Depuración:
- scripts/dupuracion/depurar_anotaciones.py
  - Filtra anotaciones en ROI 224 (elimina samples con landmarks fuera de rango o con área mínima insuficiente).

Tests / validaciones (scripts ejecutables, no framework de tests):
- scripts/test/detect_hand.py
  - Prueba MediaPipe Hands sobre una imagen local.
- scripts/test/test_dataset.py
  - Valida consistencia de JSONs (labels/bboxes), existencia de imágenes y genera visualizaciones opcionales.
  - Incluye un sanity check opcional moviendo batches a DirectML si está disponible.
- scripts/test/test_anotaciones.py y scripts/test/test.py
  - Visualización de landmarks en ROI y su proyección.

## Licencia
MIT. Ver LICENSE.
