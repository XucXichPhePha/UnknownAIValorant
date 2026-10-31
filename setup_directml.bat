@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime-env\Scripts\python.exe" (
    if "%DIRECTML_PYTHON%"=="" (
        echo Set DIRECTML_PYTHON to the full path of a 64-bit Python 3.12 executable.
        echo Example: set "DIRECTML_PYTHON=C:\Python312\python.exe"
        exit /b 1
    )
    "%DIRECTML_PYTHON%" -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Python 3.12 x64 required'"
    if errorlevel 1 exit /b 1
    "%DIRECTML_PYTHON%" -m venv runtime-env
    if errorlevel 1 exit /b 1
)
"runtime-env\Scripts\python.exe" -m pip install --no-cache-dir -r requirements-directml.txt
if errorlevel 1 exit /b 1
if not exist "best.onnx" (
    call build_onnx.bat
    if errorlevel 1 exit /b 1
)
"runtime-env\Scripts\python.exe" check_directml.py --profile --runs 30
if errorlevel 1 exit /b 1
if not exist "best-focus-256x192.onnx" (
    call build_focus_models.bat
    if errorlevel 1 exit /b 1
)
if not exist "best-focus-192x128.onnx" (
    call build_focus_models.bat
    if errorlevel 1 exit /b 1
)
echo Ready. Run run_directml.bat for the fast center-view defaults.
