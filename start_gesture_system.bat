@echo off
echo Iniciando aplicacion de control gestual...

REM Activar entorno virtual
call .venv\Scripts\activate.bat

REM Ejecutar aplicacion de escritorio
python -m src.main