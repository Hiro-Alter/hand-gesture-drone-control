# Esquema de control del dron virtual

Este documento define el esquema de control entre los gestos reconocidos, los comandos internos de la aplicación y las llamadas a AirSim. El objetivo es evitar que cada predicción estable se traduzca en una orden repetida sin contexto, y garantizar que las maniobras críticas, especialmente el aterrizaje, no puedan ser interrumpidas por gestos posteriores.

## Objetivos

- Separar reconocimiento de gestos y control del dron.
- Evitar comandos duplicados cuando el dron ya está ejecutando el mismo curso.
- Mantener movimiento continuo sin saturar Unity con llamadas por frame.
- Bloquear comandos nuevos durante el aterrizaje.
- Desarmar motores después de aterrizar.
- Mantener comportamiento seguro ante baja confianza o ausencia de mano.
- Registrar si un comando fue programado, ignorado o queda listo pero sin conexión.

## Componentes

### Entrada

La entrada de control es un comando semántico ya estabilizado:

```text
takeoff, land, hover, forward, backward, left, right, ascend, descend, rotate_yaw
```

La clasificación del gesto, la confianza y el filtro de estabilidad se resuelven antes de llegar a la máquina de estados.

### Máquina de estados

Módulo:

```text
src/commands/drone_control_state.py
```

Clase principal:

```text
DroneControlStateMachine
```

Esta clase decide si un comando se acepta o se ignora. No llama directamente a AirSim; solo valida transiciones y conserva el estado lógico del dron.

### Cliente AirSim

Módulo:

```text
src/airsim_client/client.py
```

`AirSimClient` recibe comandos desde la interfaz o desde el pipeline, consulta la máquina de estados y solo programa el comando en AirSim si la transición fue aceptada.

Los comandos se ejecutan en un worker interno en segundo plano para que el hilo de cámara, ROI e inferencia no se bloquee por llamadas RPC a Unity.

## Orientación y marco de referencia

Los comandos de desplazamiento usan `moveByVelocityBodyFrameAsync` cuando está disponible. Esto significa que:

- `forward` avanza hacia el frente actual del dron;
- `backward` retrocede respecto al frente actual;
- `left` y `right` se aplican respecto a los laterales actuales del dron;
- `ascend` y `descend` conservan el eje vertical de AirSim.

Esta decisión corrige el problema de orientación donde, después de ejecutar `rotate_yaw`, un comando como `forward` seguía moviéndose sobre el eje global original del simulador. Si por compatibilidad la API no expone `moveByVelocityBodyFrameAsync`, la aplicación cae automáticamente a `moveByVelocityAsync`.

## Estados

| Estado | Significado |
| --- | --- |
| `DISCONNECTED` | No hay cliente AirSim activo. |
| `READY` | Hay conexión, pero no se conoce con certeza si el dron está en tierra o en vuelo. |
| `GROUNDED` | El dron está en tierra. Solo acepta `takeoff`. |
| `TAKING_OFF` | Despegue en curso. Se ignoran comandos de movimiento hasta finalizar. |
| `AIRBORNE` | El dron está en vuelo y puede aceptar movimiento, giro, `hover` o `land`. |
| `LANDING` | Aterrizaje en curso. Se ignoran todos los comandos nuevos. |
| `ERROR` | Una operación crítica falló. Requiere reconexión o intervención. |

## Reglas de transición

| Estado actual | Comando | Resultado |
| --- | --- | --- |
| `DISCONNECTED` | cualquiera | Ignorar. |
| `GROUNDED` | `takeoff` | Aceptar y pasar a `TAKING_OFF`. |
| `GROUNDED` | movimiento, `hover`, `land` | Ignorar. |
| `TAKING_OFF` | cualquiera | Ignorar hasta terminar despegue. |
| `AIRBORNE` | movimiento nuevo | Aceptar y cambiar curso. |
| `AIRBORNE` | mismo movimiento | Ignorar si aún no vence el refresco temporizado. |
| `AIRBORNE` | mismo movimiento después del refresco | Aceptar renovación de curso. |
| `AIRBORNE` | `hover` | Aceptar si el comando activo no es `hover`. |
| `AIRBORNE` | `land` | Aceptar y pasar a `LANDING`. |
| `LANDING` | cualquiera | Ignorar hasta terminar aterrizaje y desarme. |

## Política para comandos duplicados

Si el comando anterior es `forward` y el nuevo también es `forward`, la máquina de estados no envía otra llamada inmediatamente. El dron conserva el curso actual.

Para no perder movimiento cuando AirSim usa comandos de velocidad con duración finita, existe una renovación temporizada:

```json
"movement_duration_s": 2.5,
"motion_refresh_s": 2.0
```

Esto significa:

- la primera orden `forward` se envía;
- las repeticiones inmediatas de `forward` se ignoran;
- si el gesto sigue estable y pasa `motion_refresh_s`, se renueva la orden antes de que venza `movement_duration_s`;
- si el usuario cambia a `hover`, `left`, `right`, etc., el cambio se acepta inmediatamente.

## Aterrizaje seguro

Cuando se acepta `land`:

1. La máquina pasa a `LANDING`.
2. Se limpia el comando pendiente y no se aceptan comandos nuevos.
3. `AirSimClient` ejecuta `landAsync(...).join()` en el worker de AirSim.
4. Al terminar, ejecuta `armDisarm(False)`.
5. Consulta `getMultirotorState().landed_state` cuando está disponible.
6. Si `landAsync` finaliza y el desarme se ejecuta, la aplicación pasa a `GROUNDED`.
7. Si `landed_state` no se puede leer, la finalización de `landAsync` más `armDisarm(False)` se usa como respaldo de seguridad.

La API de AirSim también expone `simGetCollisionInfo()`, pero para aterrizaje se prefiere `getMultirotorState().landed_state` porque representa directamente el estado de vuelo. Las colisiones pueden incluir contactos laterales u objetos no relacionados con suelo firme.

Durante `LANDING`, gestos como `avanzar`, `subir`, `girar` o incluso baja confianza no interrumpen el aterrizaje.

## Manejo de ausencia de mano y baja confianza

El pipeline mantiene el comportamiento seguro actual:

```text
sin mano -> hover
baja confianza -> hover
```

La diferencia es que `hover` también pasa por la máquina de estados:

- si ya está en `hover`, se ignora como duplicado;
- si el dron está aterrizando, se ignora para no interrumpir `land`;
- si el dron está en tierra, se ignora.

## Configuración relevante

Archivo:

```text
config/app_config.json
```

Parámetros:

```json
"commands": {
  "send_rate_hz": 2.0,
  "repeat_same_command_s": 1.0
},
"airsim": {
  "forward_speed_mps": 3.5,
  "lateral_speed_mps": 3.0,
  "vertical_speed_mps": 2.2,
  "yaw_rate_deg_s": 45.0,
  "movement_duration_s": 2.5,
  "motion_refresh_s": 2.0,
  "use_body_frame": true
}
```

`forward_speed_mps` controla avanzar y retroceder. `lateral_speed_mps` controla izquierda y derecha. `yaw_rate_deg_s` queda en 45 °/s por defecto para que el giro sea más controlable.

`send_rate_hz` y `repeat_same_command_s` limitan lo que sale del pipeline de reconocimiento.

`motion_refresh_s` es una protección adicional dentro de la máquina de estados. Aunque el pipeline permita una repetición, la máquina no renueva el movimiento hasta que venza este intervalo.

## Logs

Los eventos CSV diferencian:

| Evento | Significado |
| --- | --- |
| `command_scheduled` | El comando fue aceptado por la máquina de estados y programado en AirSim. |
| `command_ignored` | La máquina de estados rechazó el comando por duplicado, aterrizaje, tierra o transición inválida. |
| `command_ready` | El comando fue calculado, pero AirSim no está conectado. |

## Consideraciones para Unity

El Editor de Unity se vuelve lento si la Console acumula miles de mensajes repetidos. Para evitarlo:

- `C:\Users\Arley\Documents\AirSim\settings.json` debe tener `"LogMessagesVisible": false`;
- en el wrapper Unity usado por este proyecto, `Vehicle.cs` fue ajustado para iniciar `print_log_messages_ = false`;
- la tecla `T` en Unity permite alternar esos logs si se necesitan durante depuración.

## Criterios de aceptación

- Mantener un gesto `avanzar` no debe generar comandos por frame.
- Cambiar de `avanzar` a `detener` debe detener el curso.
- Cambiar de `avanzar` a `izquierda` debe cambiar de curso.
- Ejecutar `aterrizar` debe bloquear cualquier comando posterior hasta terminar.
- Tras aterrizar, el dron debe quedar desarmado.
- El pipeline de cámara debe continuar aunque AirSim tarde en responder.
