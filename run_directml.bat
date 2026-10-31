@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist "runtime-env\Scripts\python.exe" (
    echo Run setup_directml.bat first.
    pause
    exit /b 1
)
"runtime-env\Scripts\python.exe" screen_overlay.py %*
set "directmlExit=%errorlevel%"
if not "%directmlExit%"=="0" pause
exit /b %directmlExit%
