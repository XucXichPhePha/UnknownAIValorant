from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_MODEL = PROJECT_DIR / "best.pt"
ENGINE_MODEL = PROJECT_DIR / "best.engine"


def main() -> int:
    if not SOURCE_MODEL.is_file():
        print(f"Không tìm thấy model: {SOURCE_MODEL}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("Không tìm thấy CUDA GPU để build TensorRT engine.", file=sys.stderr)
        return 1

    print(f"Đang build TensorRT cho {torch.cuda.get_device_name(0)}.")
    print("Quá trình này có thể mất vài phút ở lần đầu.")

    with tempfile.TemporaryDirectory(prefix="screen-overlay-engine-") as temp:
        temp_model = Path(temp) / SOURCE_MODEL.name
        shutil.copy2(SOURCE_MODEL, temp_model)
        exported = YOLO(str(temp_model)).export(
            format="engine",
            imgsz=640,
            quantize=None,
            dynamic=False,
            workspace=4,
            device=0,
        )
        engine_path = Path(exported)
        if not engine_path.is_file():
            print(
                f"Ultralytics không tạo được engine tại {engine_path}.",
                file=sys.stderr,
            )
            return 1
        shutil.copy2(engine_path, ENGINE_MODEL)

    print(f"Đã tạo: {ENGINE_MODEL}")
    print("Chạy run.bat để dùng engine TensorRT.")
    print("Engine dùng độ chính xác FP32, dành cho GPU hiện tại.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
