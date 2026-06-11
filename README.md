# Hand Gesture Drone Control

Aplicación de escritorio para Windows orientada al control de un dron virtual en AirSim mediante reconocimiento de gestos estáticos de la mano.

El flujo principal es:

```text
Cámara RGB -> MediaPipe -> ROI -> mejoramiento -> PyTorch -> gesto -> comando -> AirSim
```

## Estado Actual

- Interfaz gráfica en PySide6.
- Interfaz por pestañas: Operación, Configuración, AirSim, Logs y Pruebas.
- Predicciones visibles en tarjetas de alto contraste y sobre la imagen de cámara.
- Captura de cámara con OpenCV.
- Detección de mano y landmarks con MediaPipe.
- Extracción de ROI de la mano.
- Mejoramiento de imagen migrado desde los scripts previos.
- Inferencia con modelos TorchScript de PyTorch.
- ResNet18 como modelo por defecto.
- MobileNetV3-Small como modelo alternativo.
- DirectML como dispositivo preferido y CPU como respaldo.
- Estabilidad configurable de predicción.
- Mapeo gesto-comando desde `config/gesture_commands.json`.
- Cliente AirSim encapsulado con envío de comandos en segundo plano.
- Logs CSV por sesión.

WebSocket no forma parte del flujo final.

## Estructura Principal

```text
src/
  main.py
  ui/
  runtime/
  vision/
  inference/
  commands/
  airsim_client/
  utils/
config/
models/
docs/
tests/
tests_future/
```

## Modelos

La fuente canónica de modelos es `models/manifest.json`.

Modelos disponibles:

- `resnet18`
- `mobilenetv3_small`

Cada modelo conserva:

- `model_torchscript.pt`
- `model_state_dict.pth`
- `model.onnx`
- `metadata.json`
- `labels.txt`

## Instalación

Usar Python 3.10 en Windows.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

AirSim se instala aparte cuando el simulador ya esté listo. El paquete requiere desactivar build isolation en algunas instalaciones:

```powershell
pip install -r requirements-airsim-prereqs.txt
pip install --no-build-isolation -r requirements-airsim.txt
```

`requirements-airsim-prereqs.txt` fija `msgpack==0.6.2` porque `msgpack-rpc-python`, usado por AirSim, no es compatible con `msgpack` 1.x.

La configuración de AirSim usa `127.0.0.1:41451` por defecto. Para evitar lag en Unity, los comandos se encolan en segundo plano y `commands.repeat_same_command_s` controla cada cuánto se repite una orden sostenida. Los desplazamientos usan marco del cuerpo del dron (`use_body_frame=true`), por lo que `avanzar` respeta el frente actual después de girar.

Velocidades configurables principales:

- `airsim.forward_speed_mps`: velocidad para avanzar y retroceder.
- `airsim.lateral_speed_mps`: velocidad para moverse a izquierda y derecha.
- `airsim.vertical_speed_mps`: velocidad para subir y bajar.
- `airsim.yaw_rate_deg_s`: velocidad de giro, 45 °/s por defecto.

## Ejecución

```powershell
python -m src.main
```

También se puede usar:

```powershell
start_gesture_system.bat
```

## Pruebas Locales

Estas pruebas no requieren AirSim ni Unity:

```powershell
python tests\smoke_test_core.py
python tests\smoke_test_pipeline.py
```

Diagnóstico general del entorno:

```powershell
python -m src.diagnostics
```

Las pruebas funcionales con AirSim requieren tener Unity ejecutando el proyecto `DroneDemo`.

## Gestos Y Comandos

El mapeo vigente está en `config/gesture_commands.json`:

- `fist`: despegar
- `palm`: aterrizar
- `stop`: detener
- `like`: avanzar
- `dislike`: retroceder
- `peace`: mover izquierda
- `peace_inverted`: mover derecha
- `two_up`: subir
- `two_up_inverted`: bajar
- `rock`: girar

## Documentación

- `ESQUEMA_CONTROL_DRON.md`: máquina de estados y reglas de control del dron.
- `DOCUMENTO_TECNICO_SOFTWARE.md`: documento maestro.
- `AGENTS.md`: reglas persistentes para Codex.
- `docs/diagnostico_repositorio.md`: decisiones de auditoría y limpieza.
- `docs/arquitectura.md`: separación de responsabilidades.
- `docs/pipeline_vision.md`: pipeline visual.
- `docs/integracion_airsim.md`: integración prevista con AirSim.
- `docs/modulo_pruebas_futuro.md`: pruebas sistemáticas futuras.
