"""Real precision/recall/mAP evaluation using Ultralytics' own validated implementation, not a hand-rolled one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(model_name: str, data_yaml: str) -> dict:
    model = YOLO(model_name)
    metrics = model.val(data=data_yaml, verbose=False)

    return {
        "model_name": model_name,
        "dataset": data_yaml,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "detection.yaml")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Ultralytics dataset yaml")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "experiments" / "detection_eval_results.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = [evaluate(name, args.data) for name in config["benchmark"]["models_to_compare"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Model':<15}{'Precision':<12}{'Recall':<10}{'mAP@50':<10}{'mAP@50:95':<10}")
    for r in results:
        print(f"{r['model_name']:<15}{r['precision']:<12}{r['recall']:<10}{r['map50']:<10}{r['map50_95']:<10}")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
