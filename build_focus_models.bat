@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime-env\Scripts\python.exe" (
    echo Run setup_directml.bat first.
    exit /b 1
)
"runtime-env\Scripts\python.exe" -c "import onnx" >nul 2>&1
if errorlevel 1 (
    "runtime-env\Scripts\python.exe" -m pip install "onnx>=1.16,<2"
    if errorlevel 1 exit /b 1
)
"runtime-env\Scripts\python.exe" build_focus_models.py %*
exit /b %errorlevel%
