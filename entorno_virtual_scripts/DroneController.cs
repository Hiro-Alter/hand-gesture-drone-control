using UnityEngine;
using System.Collections;

public enum DroneState { MotorsOff, TakingOff, Hover, Flying, Landing }

public enum MovementCommand { None, Forward, Backward, Left, Right, Ascend, Descend, RotateYaw }

[RequireComponent(typeof(Rigidbody))]
public class DroneController : MonoBehaviour
{
    public DroneState state = DroneState.MotorsOff;

    [Header("Velocities")]
    public float moveSpeed = 2.0f;        // m/s horizontal
    public float verticalSpeed = 1.5f;    // m/s vertical
    public float yawSpeed = 45f;          // deg/s
    public float takeoffHeight = 1.5f;    // m

    [Header("Timed Movement Settings")]
    public float movementDuration = 0.5f;  // 500ms
    public float stopCooldown = 5.0f;       // 5 segundos

    private Rigidbody rb;
    private Vector3 targetVelocity = Vector3.zero; // local space
    private float targetVertical = 0f;   // m/s vertical
    private float targetYaw = 0f;        // normalized -1..1
    private float currentYaw = 0f;

    private MovementCommand currentCommand = MovementCommand.None;
    private MovementCommand pendingCommand = MovementCommand.None;
    private Coroutine currentMovementCoroutine;
    private bool isInStopCooldown = false;
    private bool hasReceivedStopInCooldown = false;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
    }

    void FixedUpdate()
    {
        ApplyMovement();
        HandleStateTransitions();
    }

    void ApplyMovement()
    {
        // Movimiento local transform.forward / right
        Vector3 worldVelocity = transform.TransformDirection(targetVelocity) * moveSpeed;
        Vector3 delta = (worldVelocity + Vector3.up * targetVertical) * Time.fixedDeltaTime;
        Vector3 newPos = rb.position + delta;
        rb.MovePosition(newPos);

        // Rotación yaw
        if (Mathf.Abs(targetYaw) > 0.001f)
        {
            float yawStep = targetYaw * yawSpeed * Time.fixedDeltaTime;
            currentYaw += yawStep;
            rb.MoveRotation(Quaternion.Euler(0f, currentYaw, 0f));
        }
    }

    void HandleStateTransitions()
    {
        switch (state)
        {
            case DroneState.TakingOff:
                if (rb.position.y < takeoffHeight - 0.05f)
                {
                    targetVertical = verticalSpeed;
                }
                else
                {
                    targetVertical = 0f;
                    state = DroneState.Hover;
                }
                break;
            case DroneState.Landing:
                if (rb.position.y > 0.05f)
                {
                    targetVertical = -verticalSpeed;
                }
                else
                {
                    targetVertical = 0f;
                    state = DroneState.MotorsOff;
                }
                break;
            case DroneState.Hover:
                if (currentCommand == MovementCommand.None)
                {
                    targetVelocity = Vector3.zero;
                    targetVertical = 0f;
                    targetYaw = 0f;
                }
                break;
            case DroneState.Flying:
                // En flying, se aplica targetVelocity / targetVertical
                break;
            case DroneState.MotorsOff:
                targetVertical = 0f;
                targetVelocity = Vector3.zero;
                targetYaw = 0f;
                break;
        }
    }

    // --- Métodos atómicos públicos ---
    public void Takeoff()
    {
        if (state == DroneState.MotorsOff)
        {
            state = DroneState.TakingOff;
        }
    }

    public void Land()
    {
        if (state != DroneState.MotorsOff)
        {
            state = DroneState.Landing;
        }
    }

    public void Hover()
    {
        // Si está en cooldown y aún no ha recibido stop, no procesar
        if (isInStopCooldown && !hasReceivedStopInCooldown)
        {
            hasReceivedStopInCooldown = true;
            return;
        }

        // Si ya recibió stop durante el cooldown, ignorar llamadas adicionales
        if (isInStopCooldown && hasReceivedStopInCooldown)
        {
            return;
        }

        if (state == DroneState.Flying || state == DroneState.TakingOff || state == DroneState.Hover)
        {
            // Cancelar movimiento actual
            if (currentMovementCoroutine != null)
            {
                StopCoroutine(currentMovementCoroutine);
                currentMovementCoroutine = null;
            }

            currentCommand = MovementCommand.None;
            pendingCommand = MovementCommand.None;
            state = DroneState.Hover;
            targetVelocity = Vector3.zero;
            targetVertical = 0f;
            targetYaw = 0f;

            // Iniciar cooldown de 5 segundos
            if (!isInStopCooldown)
            {
                StartCoroutine(StopCooldownCoroutine());
            }
        }
    }

    public void MoveForward() { RequestMovement(MovementCommand.Forward); }
    public void MoveBackward() { RequestMovement(MovementCommand.Backward); }
    public void MoveLeft() { RequestMovement(MovementCommand.Left); }
    public void MoveRight() { RequestMovement(MovementCommand.Right); }
    public void Ascend() { RequestMovement(MovementCommand.Ascend); }
    public void Descend() { RequestMovement(MovementCommand.Descend); }
    public void RotateYaw() { RequestMovement(MovementCommand.RotateYaw); }

    private void RequestMovement(MovementCommand command)
    {
        // Bloquear movimientos durante el cooldown de stop
        if (isInStopCooldown)
        {
            return;
        }

        if (!EnsureFlying())
        {
            return; // No se puede mover si los motores están apagados
        }

        // Si es el mismo comando que el actual, extender la duración
        if (currentCommand == command && currentMovementCoroutine != null)
        {
            // Cancelar la coroutine actual para reiniciar el temporizador
            StopCoroutine(currentMovementCoroutine);
            currentMovementCoroutine = StartCoroutine(TimedMovementCoroutine(command));
            return;
        }

        // Si hay un movimiento en curso pero es diferente, ponerlo en espera
        if (currentMovementCoroutine != null && currentCommand != MovementCommand.None)
        {
            pendingCommand = command;
            return;
        }

        // Iniciar nuevo movimiento
        ExecuteMovementCommand(command);
    }

    private void ExecuteMovementCommand(MovementCommand command)
    {
        currentCommand = command;
        
        // Aplicar el movimiento según el comando
        switch (command)
        {
            case MovementCommand.Forward:
                targetVelocity = new Vector3(0f, 0f, 1f);
                break;
            case MovementCommand.Backward:
                targetVelocity = new Vector3(0f, 0f, -1f);
                break;
            case MovementCommand.Left:
                targetVelocity = new Vector3(-1f, 0f, 0f);
                break;
            case MovementCommand.Right:
                targetVelocity = new Vector3(1f, 0f, 0f);
                break;
            case MovementCommand.Ascend:
                targetVelocity = Vector3.zero;
                targetVertical = verticalSpeed;
                break;
            case MovementCommand.Descend:
                targetVelocity = Vector3.zero;
                targetVertical = -verticalSpeed;
                break;
            case MovementCommand.RotateYaw:
                targetVelocity = Vector3.zero;
                targetYaw = 1f;
                break;
        }

        // Iniciar temporizador de 500ms
        currentMovementCoroutine = StartCoroutine(TimedMovementCoroutine(command));
    }

    private IEnumerator TimedMovementCoroutine(MovementCommand command)
    {
        yield return new WaitForSeconds(movementDuration);

        // Detener el movimiento
        targetVelocity = Vector3.zero;
        targetVertical = 0f;
        targetYaw = 0f;
        
        currentCommand = MovementCommand.None;

        if (state == DroneState.Flying)
        {
            state = DroneState.Hover;
        }

        currentMovementCoroutine = null;

        // Si hay un comando pendiente, ejecutarlo ahora
        if (pendingCommand != MovementCommand.None)
        {
            MovementCommand next = pendingCommand;
            pendingCommand = MovementCommand.None;
            ExecuteMovementCommand(next);
        }
    }

    private IEnumerator StopCooldownCoroutine()
    {
        isInStopCooldown = true;
        hasReceivedStopInCooldown = true;

        yield return new WaitForSeconds(stopCooldown);

        isInStopCooldown = false;
        hasReceivedStopInCooldown = false;
    }

    private bool EnsureFlying()
    {
        if (state == DroneState.Hover)
        {
            state = DroneState.Flying;
            return true;
        }
        
        // No permitir movimientos si los motores están apagados
        if (state == DroneState.MotorsOff)
        {
            return false;
        }

        return state == DroneState.Flying;
    }
}
