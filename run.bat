@echo off
cd /d "%~dp0"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
    echo Chua co moi truong Python. Hay chay setup.bat truoc.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" screen_overlay.py
if errorlevel 1 pause
