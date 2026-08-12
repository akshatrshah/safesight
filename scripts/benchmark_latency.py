"""Measures real latency/FPS per model size on my own hardware. Accuracy alone isn't enough for a safety system."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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

    # first few calls are slower (lazy init etc), so I warm up before timing
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
