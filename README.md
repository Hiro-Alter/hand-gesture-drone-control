# Hand Gesture Drone Control

Aplicacion de escritorio para Windows orientada al control de un dron virtual en AirSim mediante reconocimiento de gestos estaticos de la mano.

El flujo principal es:

```text
Camara RGB -> MediaPipe -> ROI -> mejoramiento -> PyTorch -> gesto -> comando -> AirSim
```

## Estado Actual

- Interfaz grafica en PySide6.
- Captura de camara con OpenCV.
- Deteccion de mano y landmarks con MediaPipe.
- Extraccion de ROI de la mano.
- Mejoramiento de imagen migrado desde los scripts previos.
- Inferencia con modelos TorchScript de PyTorch.
- ResNet18 como modelo por defecto.
- MobileNetV3-Small como modelo alternativo.
- DirectML como dispositivo preferido y CPU como respaldo.
- Estabilidad configurable de prediccion.
- Mapeo gesto-comando desde `config/gesture_commands.json`.
- Cliente AirSim encapsulado.
- Logs CSV por sesion.

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

La fuente canonica de modelos es `models/manifest.json`.

Modelos disponibles:

- `resnet18`
- `mobilenetv3_small`

Cada modelo conserva:

- `model_torchscript.pt`
- `model_state_dict.pth`
- `model.onnx`
- `metadata.json`
- `labels.txt`

## Instalacion

Usar Python 3.10 en Windows.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

AirSim se instala aparte cuando el simulador ya este listo. El paquete requiere desactivar build isolation en algunas instalaciones:

```powershell
pip install -r requirements-airsim-prereqs.txt
pip install --no-build-isolation -r requirements-airsim.txt
```

## Ejecucion

```powershell
python -m src.main
```

Tambien se puede usar:

```powershell
start_gesture_system.bat
```

## Pruebas Locales

Estas pruebas no requieren AirSim ni Unity:

```powershell
python tests\smoke_test_core.py
python tests\smoke_test_pipeline.py
```

Las pruebas funcionales con AirSim se omiten hasta que el proyecto en Unity este configurado.

## Gestos Y Comandos

El mapeo vigente esta en `config/gesture_commands.json`:

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

## Documentacion

- `DOCUMENTO_TECNICO_SOFTWARE.md`: documento maestro.
- `AGENTS.md`: reglas persistentes para Codex.
- `docs/diagnostico_repositorio.md`: decisiones de auditoria y limpieza.
- `docs/arquitectura.md`: separacion de responsabilidades.
- `docs/pipeline_vision.md`: pipeline visual.
- `docs/integracion_airsim.md`: integracion prevista con AirSim.
- `docs/modulo_pruebas_futuro.md`: pruebas sistematicas futuras.
