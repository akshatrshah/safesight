"""
Benchmark inference latency and FPS across the model sizes listed in
configs/detection.yaml -> benchmark.models_to_compare.

WHY THIS SCRIPT EXISTS
-------------------------
Your project spec (and any real perception role) requires reporting
accuracy AND latency together, never accuracy alone — a highly accurate
model that runs at 2 FPS is often useless for a real-time safety system.
This script produces the numbers for that comparison. Every number here
is measured on your actual hardware, not estimated.

Usage:
    python scripts/benchmark_latency.py --source path/to/sample_images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `perception` is importable without setting PYTHONPATH

import cv2
import yaml

from perception.detection.detector import Detector


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def benchmark_model(model_name: str, config: dict, image) -> dict:
    detector = Detector(
        model_name=model_name,
        target_classes=config["target_classes"],
        confidence_threshold=config["confidence_threshold"],
    )

    warmup = config["benchmark"]["num_warmup_frames"]
    timed = config["benchmark"]["num_timed_frames"]

    # Warmup: first few inference calls are slower (lazy CUDA init, etc.)
    # and would distort the average if included.
    for _ in range(warmup):
        detector.detect(image)

    start = time.perf_counter()
    for _ in range(timed):
        detector.detect(image)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / timed) * 1000
    fps = timed / elapsed

    return {
        "model_name": model_name,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 2),
        "num_timed_frames": timed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="One representative image")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "detection.yaml")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "experiments" / "detection_latency_results.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    image = cv2.imread(str(args.source))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.source}")

    results = [benchmark_model(name, config, image) for name in config["benchmark"]["models_to_compare"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Model':<15}{'Latency (ms)':<15}{'FPS':<10}")
    for r in results:
        print(f"{r['model_name']:<15}{r['avg_latency_ms']:<15}{r['fps']:<10}")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
