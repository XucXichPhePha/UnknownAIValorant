"""Benchmark the full preprocessing/inference/postprocessing path without capturing the screen."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from directml_backend import DirectMLDetector, frame_delay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--fps", type=int, choices=range(30, 241), default=120)
    parser.add_argument("--profile", action="store_true", help="Also verify GPU node placement")
    parser.add_argument("--resolution", choices=("1280x882", "1280x720"), default="1280x882")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    project = Path(__file__).resolve().parent
    detector = DirectMLDetector(
        project / "best.onnx", args.device_id,
        profile_prefix=project / "directml-profile" if args.profile else None,
    )
    width, height = map(int, args.resolution.split("x"))
    # Deterministic synthetic input at the requested native display resolution.
    frame = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    for _ in range(5):
        detector.predict(frame)
        time.sleep(1 / args.fps)
    times = []
    started = time.perf_counter()
    for _ in range(args.runs):
        frame_started = time.perf_counter()
        detector.predict(frame)
        elapsed = time.perf_counter() - frame_started
        times.append(elapsed * 1000)
        time.sleep(frame_delay(args.fps, elapsed))
    elapsed = time.perf_counter() - started
    report = {
        "onnxruntime": ort.__version__, "providers": detector.session.get_providers(),
        "device_id": args.device_id, "model_input": detector.size,
        "classes": detector.names, "target_fps": args.fps, "runs": args.runs,
        "mean_processing_ms": round(float(np.mean(times)), 2),
        "p95_processing_ms": round(float(np.percentile(times, 95)), 2),
        "paced_fps": round(args.runs / elapsed, 2),
        "cpu_inference_fallback": False,
        "input": f"synthetic {width}x{height} BGR; not a gameplay benchmark",
    }
    if args.profile:
        path = Path(detector.session.end_profiling())
        events = json.loads(path.read_text(encoding="utf-8"))
        placements = {}
        for event in events:
            provider = event.get("args", {}).get("provider")
            if provider:
                placements[provider] = placements.get(provider, 0) + 1
        report["profile_node_events"] = placements
        if not placements.get("DmlExecutionProvider") or placements.get("CPUExecutionProvider"):
            raise RuntimeError(f"GPU-only verification failed: {placements}")
    print(json.dumps(report, indent=2))
    (project / f"benchmark-{args.resolution}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
