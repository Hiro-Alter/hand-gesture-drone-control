@echo off
echo Iniciando sistema de reconocimiento de gestos...

REM Activar entorno virtual
call .venv\Scripts\activate.bat

REM Iniciar servidor WebSocket en nueva ventana
start "WebSocket Server" cmd /k python scripts\prototipo\gesture_ws_server.py

REM Esperar 2 segundos para que el servidor inicie
timeout /t 2 /nobreak

REM Iniciar cámara en ventana actual
python scripts\prototipo\camara_prediccion.py

REM Cuando se cierre la cámara, cerrar el servidor
taskkill /FI "WINDOWTITLE eq WebSocket Server*" /T /F