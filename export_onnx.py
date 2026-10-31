"""One-time CPU conversion of best.pt; no CUDA or TensorRT downloads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(320, 416, 640), default=320)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / "runtime-data"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO(str(project / "best.pt"))
    print(f"Exporting {model.names} at {args.size}px on CPU (one-time operation).")
    path = model.export(
        format="onnx", imgsz=args.size, batch=1, device="cpu",
        dynamic=False, simplify=True, opset=17, quantize=None, nms=False,
    )
    print(f"DirectML model ready: {path}")


if __name__ == "__main__":
    main()
