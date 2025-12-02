# Python
import asyncio
import websockets
import json
import time

CONNECTED = set()

async def handler(ws):
    print("Client connected:", ws.remote_address)
    CONNECTED.add(ws)
    try:
        async for message in ws:
            print("Received from client:", message)
            # Reenviar el mensaje a todos los clientes conectados (broadcast)
            if CONNECTED:
                await asyncio.gather(
                    *(client.send(message) for client in CONNECTED if client != ws),
                    return_exceptions=True
                )
    except websockets.ConnectionClosed:
        pass
    finally:
        CONNECTED.remove(ws)
        print("Client disconnected")

async def send_gesture(gesture, conf=None):
    if not CONNECTED:
        print("No clients connected. Discarded:", gesture)
        return
    payload = {"gesture": gesture, "ts": time.time()}
    if conf is not None:
        payload["conf"] = conf
    msg = json.dumps(payload)
    print(f"Enviando gesto: {payload}")
    await asyncio.gather(*(ws.send(msg) for ws in CONNECTED))

async def main():
    host = "127.0.0.1"
    port = 8765
    async with websockets.serve(handler, host, port):
        print(f"WebSocket server started at ws://{host}:{port}")
        await asyncio.Future()  # mantiene el servidor corriendo

if __name__ == "__main__":
    asyncio.run(main())
