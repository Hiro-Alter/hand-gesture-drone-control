"""
Sistema de reconocimiento de gestos de mano en tiempo real.
Captura video desde la cámara, detecta manos con MediaPipe, clasifica gestos
con ResNet18 y envía los resultados vía WebSocket a Unity.
"""

import sys
import json
import time
import asyncio
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_directml as dml
from torchvision import models
import mediapipe as mp
import websockets


# =========================
#  CONFIGURACIÓN GLOBAL
# =========================

# Rutas del proyecto
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "outputs_resnet18_directml" / "resnet18_directml_finetuned.pt"
CLASSMAP_PATH = REPO_ROOT / "models" / "outputs_resnet18_directml" / "class_mapping.json"

# WebSocket
WEBSOCKET_URI = "ws://127.0.0.1:8765"  # Dirección del servidor WebSocket

# Parámetros de inferencia
CONFIDENCE_THRESHOLD = 0.90  # Confianza mínima para aceptar predicción (90%)
MIN_SEND_INTERVAL = 0.15     # Segundos mínimos entre envíos de gestos
GESTURE_STABILITY_TIME = 0.3  # Tiempo que debe mantenerse un gesto para enviarlo (segundos)
GESTURE_STABILITY_COUNT = 3   # Frames consecutivos requeridos con el mismo gesto

# Parámetros de detección de mano (MediaPipe)
HAND_PADDING = 0.3                # Margen alrededor de la mano detectada (30%)
HAND_DETECTION_CONFIDENCE = 0.5   # Confianza mínima para detección
HAND_TRACKING_CONFIDENCE = 0.5    # Confianza mínima para tracking

# Normalización ImageNet (usada durante el entrenamiento)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Agregar módulo de preprocesamiento al path
PREPROC_DIR = REPO_ROOT / "scripts" / "preprocesamiento"
if str(PREPROC_DIR) not in sys.path:
    sys.path.append(str(PREPROC_DIR))

try:
    import mejorar_imagenes as mi  # type: ignore
except Exception as e:
    raise ImportError(f"No se pudo importar 'mejorar_imagenes.py': {e}")


# =========================
#  CARGA DE CONFIGURACIÓN
# =========================

def load_class_mapping(path: Path) -> dict:
    """
    Carga el mapeo de índices a etiquetas de clase.
    
    Soporta formatos:
    - Lista: ["palm", "fist", ...] → {0: "palm", 1: "fist", ...}
    - Dict con idx_to_class: {"idx_to_class": {"0": "palm", ...}}
    - Dict directo: {"0": "palm", "1": "fist", ...}
    
    Returns:
        dict: Mapeo {índice: etiqueta}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Formato con clave estándar
    if isinstance(data, dict) and "idx_to_class" in data:
        return {int(k): str(v) for k, v in data["idx_to_class"].items()}

    # Lista simple
    if isinstance(data, list):
        return {i: str(lbl) for i, lbl in enumerate(data)}

    # Dict directo id->label
    if isinstance(data, dict):
        return {int(k): str(v) for k, v in data.items()}

    raise ValueError(f"Formato no soportado en {path}")


# =========================
#  CONSTRUCCIÓN Y CARGA DEL MODELO
# =========================

def build_resnet18(num_classes: int) -> torch.nn.Module:
    """
    Construye ResNet18 con capa final adaptada al número de clases.
    
    Args:
        num_classes: Número de clases de salida
        
    Returns:
        Modelo ResNet18 sin pesos preentrenados
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Carga el modelo desde archivo .pt con state_dict.
    
    Args:
        model_path: Ruta al archivo .pt
        num_classes: Número de clases
        device: Dispositivo DirectML
        
    Returns:
        Modelo cargado en modo evaluación
    """
    # Cargar checkpoint (puede contener state_dict u otros metadatos)
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    
    # Extraer state_dict según el formato
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        # Formato: {"model_state": {...}, "classes": [...]}
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        # Ya es un state_dict directo
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Formato no reconocido en {model_path}")
    
    # Limpiar prefijos 'module.' de DataParallel si existen
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[key] = v
    
    # Construir modelo y cargar pesos
    model = build_resnet18(num_classes)
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    return model


# =========================
#  PREPROCESAMIENTO DE IMAGEN
# =========================

def preprocess_roi(roi_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Preprocesa ROI de mano para inferencia.
    
    Pipeline:
    1. Resize a 224x224
    2. BGR → RGB → Escala de grises
    3. Contrast stretch → Gamma → CLAHE → Unsharp
    4. Conversión a tensor 1x3x224x224
    5. Normalización ImageNet
    
    Args:
        roi_bgr: Imagen BGR de entrada
        device: Dispositivo DirectML
        
    Returns:
        Tensor normalizado listo para el modelo
    """
    # Redimensionar
    roi_resized = cv2.resize(roi_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # Pipeline de mejora (del módulo mejorar_imagenes)
    gray = mi.to_grayscale(roi_rgb)
    stretched = mi.contrast_stretch(gray)
    gamma_img = mi.gamma_correction(stretched)
    clahe_img = mi.apply_clahe(gamma_img)
    sharp = mi.gaussian_unsharp_paper(clahe_img)
    
    # Convertir a tensor RGB (replicar canal gris 3 veces)
    img3 = np.stack([sharp, sharp, sharp], axis=2).astype(np.float32) / 255.0
    chw = np.transpose(img3, (2, 0, 1))  # HWC → CHW
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)  # 1x3x224x224
    
    # Normalización ImageNet
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tensor.sub_(mean).div_(std)
    
    return tensor


# =========================
#  DETECCIÓN DE MANO
# =========================

def get_hand_roi(frame_bgr: np.ndarray, hands, padding: float = HAND_PADDING):
    """
    Detecta mano con MediaPipe y extrae ROI cuadrado.
    
    Args:
        frame_bgr: Frame BGR de la cámara
        hands: Instancia de MediaPipe Hands
        padding: Margen adicional alrededor de la mano (por defecto: HAND_PADDING)
        
    Returns:
        tuple: (roi_bgr, box) o (None, None) si no hay mano
            - roi_bgr: Recorte cuadrado de la mano
            - box: (x1, y1, x2, y2) coordenadas del recuadro
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if not result.multi_hand_landmarks:
        return None, None

    # Usar primera mano detectada
    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    # Calcular bounding box cuadrado centrado
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    side = int(max(max(xs) - min(xs), max(ys) - min(ys)) * (1.0 + padding))
    side = max(side, 1)

    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, None

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    return roi_bgr, (x1, y1, x2, y2)


# =========================
#  CLIENTE WEBSOCKET
# =========================

class WebSocketSender:
    """
    Cliente WebSocket asíncrono que envía gestos a Unity.
    Mantiene conexión persistente y reconecta automáticamente.
    """
    
    def __init__(self, uri: str):
        """
        Args:
            uri: URL del servidor WebSocket (ej: ws://127.0.0.1:8765)
        """
        self.uri = uri
        self._loop = asyncio.new_event_loop()
        self._ws = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._ensure_connection(), self._loop)

    def _run_loop(self):
        """Ejecuta el event loop de asyncio en hilo separado."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _ensure_connection(self):
        """Intenta conectar al servidor con reintentos."""
        while self._ws is None:
            try:
                self._ws = await websockets.connect(self.uri)
                print(f"[WS] Conectado a {self.uri}")
            except Exception as e:
                print(f"[WS] Error de conexión: {e}. Reintentando...")
                await asyncio.sleep(1)

    async def _send(self, payload: dict):
        """
        Envía JSON al servidor con reconexión automática.
        
        Args:
            payload: Diccionario a enviar como JSON
        """
        if self._ws is None:
            await self._ensure_connection()
        
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            print(f"[WS] Error enviando: {e}. Reconectando...")
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None
            await self._ensure_connection()
            await self._ws.send(json.dumps(payload))

    def send_gesture(self, label: str):
        """
        Envía gesto desde el hilo principal (interfaz síncrona).
        
        Args:
            label: Etiqueta del gesto detectado
        """
        payload = {"gesture": label, "ts": time.time()}
        asyncio.run_coroutine_threadsafe(self._send(payload), self._loop)

    def close(self):
        """Cierra la conexión y detiene el event loop."""
        try:
            if self._ws is not None:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result(timeout=2)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


# =========================
#  BUCLE PRINCIPAL
# =========================

def main():
    """
    Loop principal de captura, detección y clasificación.
    
    Flujo:
    1. Carga modelo y mapeo de clases
    2. Inicia conexión WebSocket
    3. Abre cámara
    4. Por cada frame:
       - Detecta mano con MediaPipe
       - Clasifica gesto con ResNet18
       - Envía resultado a Unity
    """
    # Cargar configuración
    id_to_label = load_class_mapping(CLASSMAP_PATH)
    num_classes = max(id_to_label.keys()) + 1

    # Configurar dispositivo DirectML (GPU)
    device = dml.device()
    print(f"[INIT] Usando dispositivo: {device}")

    # Cargar modelo
    model = load_model(MODEL_PATH, num_classes=num_classes, device=device)
    print(f"[INIT] Modelo cargado desde: {MODEL_PATH.name}")

    # Iniciar cliente WebSocket
    ws_sender = WebSocketSender(WEBSOCKET_URI)

    # Abrir cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        ) as hands:
            
            # Control de envío de gestos
            last_sent_label = None
            last_sent_time = 0.0
            
            # Variables para estabilización de gestos
            stable_gesture = None       # Gesto que se está estabilizando
            stable_count = 0            # Contador de frames consecutivos
            stable_start_time = 0.0     # Momento en que empezó la estabilización

            print("[READY] Presiona 'q' para salir\n")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Vista espejo (más natural para usuario)
                frame = cv2.flip(frame, 1)

                # Detectar mano
                roi, box = get_hand_roi(frame, hands)
                label_text = "Sin mano"

                if roi is not None:
                    # Preprocesar e inferir
                    tensor = preprocess_roi(roi, device)
                    
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1)
                        confidence, pred_id = torch.max(probs, dim=1)
                        confidence = float(confidence.item())
                        pred_id = int(pred_id.item())
                        label_text = id_to_label.get(pred_id, f"ID_{pred_id}")

                    # Dibujar bounding box
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Sistema de estabilización
                    now = time.time()
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Verificar si es el mismo gesto que se está estabilizando
                        if label_text == stable_gesture:
                            stable_count += 1
                        else:
                            # Nuevo gesto detectado, reiniciar contadores
                            stable_gesture = label_text
                            stable_count = 1
                            stable_start_time = now
                        
                        # Enviar solo si el gesto ha sido estable
                        time_stable = now - stable_start_time
                        if (stable_count >= GESTURE_STABILITY_COUNT and 
                            time_stable >= GESTURE_STABILITY_TIME):
                            
                            # Verificar si debe enviarse (cambió o pasó suficiente tiempo)
                            if label_text != last_sent_label or (now - last_sent_time) >= MIN_SEND_INTERVAL:
                                ws_sender.send_gesture(label_text)
                                print(f"[SENT] {label_text} (conf: {confidence:.2%}, stable: {time_stable:.2f}s)")
                                last_sent_label = label_text
                                last_sent_time = now
                    
                    # Mostrar estado de estabilización en pantalla
                    if stable_count < GESTURE_STABILITY_COUNT or time_stable < GESTURE_STABILITY_TIME:
                        label_text = f"{label_text} [{stable_count}/{GESTURE_STABILITY_COUNT}]"
                else:
                    # Sin mano, resetear estabilización
                    stable_gesture = None
                    stable_count = 0

                # Mostrar resultado en pantalla
                cv2.putText(
                    frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

                cv2.imshow("Reconocimiento de Gestos", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ws_sender.close()
        print("[EXIT] Sistema cerrado")


if __name__ == "__main__":
    main()
