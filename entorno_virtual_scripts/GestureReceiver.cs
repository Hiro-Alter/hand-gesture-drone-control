using NativeWebSocket;
using System;
using UnityEngine;

[Serializable]
public class GestureMsg { public string gesture; public double ts; }

public class GestureReceiver : MonoBehaviour
{
    [Header("WebSocket")]
    public string websocketUrl = "ws://127.0.0.1:8765";
    public DroneController drone;

    private NativeWebSocket.WebSocket websocket;

    async void Start()
    {
        websocket = new NativeWebSocket.WebSocket(websocketUrl);

        websocket.OnOpen += () => Debug.Log("[WS] Connected to " + websocketUrl);
        websocket.OnError += (e) => Debug.LogError("[WS] Error: " + e);
        websocket.OnClose += (e) => Debug.LogWarning("[WS] Closed: " + e);

        websocket.OnMessage += (bytes) => {
            var msg = System.Text.Encoding.UTF8.GetString(bytes);   
            ProcessMessage(msg);
        };

        try
        {
            await websocket.Connect();
        }
        catch (Exception ex)
        {
            Debug.LogError("[WS] Connection failed: " + ex.Message);
        }
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (websocket != null) websocket.DispatchMessageQueue();
#endif
    }

    private void ProcessMessage(string json)
    {
        try
        {
            var g = JsonUtility.FromJson<GestureMsg>(json);
            if (g != null && !string.IsNullOrEmpty(g.gesture))
            {
                ApplyGesture(g.gesture);
            }
            else
            {
                Debug.LogWarning("[GestureReceiver] Empty gesture JSON: " + json);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[GestureReceiver] JSON parse error: " + ex.Message + " raw:" + json);
        }
    }

    private void ApplyGesture(string g)
    {
        switch (g)
        {
            case "fist": drone.Takeoff(); break;
            case "palm": drone.Land(); break;
            case "stop": drone.Hover(); break;
            case "like": drone.MoveForward(); break;
            case "dislike": drone.MoveBackward(); break;
            case "peace": drone.MoveLeft(); break;
            case "peace_inverted": drone.MoveRight(); break;
            case "two_up": drone.Ascend(); break;
            case "two_up_inverted": drone.Descend(); break;
            case "rock": drone.RotateYaw(); break;
            default: Debug.LogWarning("[GestureReceiver] Unknown gesture: " + g); break;
        }
    }

    private async void OnDestroy()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }
}
