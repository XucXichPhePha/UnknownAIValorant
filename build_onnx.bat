@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime-env\Scripts\python.exe" (
    echo Run setup_directml.bat first.
    pause
    exit /b 1
)
"runtime-env\Scripts\python.exe" -m pip install --no-cache-dir "torch==2.14.1+cpu" "torchvision==0.29.1+cpu" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
"runtime-env\Scripts\python.exe" -m pip install --no-cache-dir -r requirements-export.txt
if errorlevel 1 exit /b 1
"runtime-env\Scripts\python.exe" export_onnx.py %*
exit /b %errorlevel%
