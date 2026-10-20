@echo off
cd /d "%~dp0"
chcp 65001 >nul

if not exist ".venv\Scripts\python.exe" (
    echo Chưa có môi trường Python. Hãy chạy setup.bat trước.
    pause
    exit /b 1
)

echo Đang kiểm tra TensorRT CUDA 13...
".venv\Scripts\python.exe" -c "import tensorrt"
if errorlevel 1 (
    echo Đang cài TensorRT CUDA 13, dung lượng tải khoảng 2 GB...
    ".venv\Scripts\python.exe" -m pip install tensorrt-cu13==11.4.0.106 --extra-index-url https://pypi.nvidia.com
    if errorlevel 1 (
        echo Không cài được TensorRT. Kiểm tra kết nối Internet rồi thử lại.
        pause
        exit /b 1
    )
)

echo Đang cài các thư viện xuất ONNX...
".venv\Scripts\python.exe" -m pip install "onnx>=1.12.0,<2.0.0" onnxruntime-gpu "onnxslim>=0.1.82"
if errorlevel 1 (
    echo Không cài được thư viện ONNX cần cho TensorRT.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" build_engine.py
if errorlevel 1 (
    echo Build TensorRT engine thất bại. Ứng dụng vẫn có thể dùng best.pt.
    pause
    exit /b 1
)
pause
