@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Dang tao moi truong Python rieng...
    python -m venv .venv
    if errorlevel 1 (
        echo Khong tao duoc moi truong. Hay cai Python 3.10 tro len va thu lai.
        pause
        exit /b 1
    )
)

echo Dang cap nhat pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Khong the cap nhat pip.
    pause
    exit /b 1
)

echo Dang cai PyTorch CUDA 13.0 va torchvision cho RTX 4050...
".venv\Scripts\python.exe" -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo Khong cai duoc PyTorch CUDA. Kiem tra ket noi Internet va driver NVIDIA.
    pause
    exit /b 1
)

echo Dang cai cac thu vien cua ung dung...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Khong cai duoc cac thu vien cua ung dung.
    pause
    exit /b 1
)

echo Dang kiem tra CUDA...
".venv\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'khong tim thay')"
if errorlevel 1 (
    echo Kiem tra PyTorch/CUDA that bai.
    pause
    exit /b 1
)

echo Hoan tat. Chay run.bat de bat dau nhan dien.
pause
