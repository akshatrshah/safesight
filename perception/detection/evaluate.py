"""
Evaluation harness: real precision / recall / mAP@50 / mAP@50:95, not a
visual "it looks like it works" check.

WHAT THESE METRICS MEAN (practical level)
--------------------------------------------
- Precision: of all the boxes the model predicted, what fraction were
  correct? Low precision = lots of false alarms.
- Recall: of all the real objects that existed, what fraction did the
  model find? Low recall = the model is missing real hazards — the
  worse failure mode for a safety system.
- mAP@50: "mean Average Precision" using IoU >= 0.5 to count a prediction
  as correct. A fairly forgiving threshold — a box that's roughly in the
  right place counts as a hit.
- mAP@50:95: the SAME idea, but averaged across IoU thresholds from 0.5
  to 0.95 in steps of 0.05. This punishes loosely-fitted boxes much more
  and is the standard, harder metric reported in most CV papers. Always
  report both — a model can have a high mAP@50 and a much lower
  mAP@50:95 if its boxes are roughly-but-not-precisely placed.

WHY WE USE ULTRALYTICS' BUILT-IN val(), NOT OUR OWN IMPLEMENTATION
----------------------------------------------------------------------
We hand-rolled IoU/NMS in perception/utils/geometry.py purely so you
understand the mechanism. Computing mAP correctly (matching predictions
to ground truth across all classes and IoU thresholds, handling
duplicate matches, etc.) has enough edge cases that re-implementing it
is a distraction from the actual learning goal of this milestone. We use
Ultralytics' validated implementation here, the same way you'd use a
well-tested library function in backend work rather than re-writing your
own JSON parser.

DATASET NOTE
--------------
This first pass evaluates on Ultralytics' small built-in COCO8 sample
dataset (8 images, COCO-format labels, auto-downloaded) purely to prove
the evaluation harness itself is wired correctly end-to-end. This is a
placeholder, not a warehouse dataset — Milestone 2 swaps in a real
labeled warehouse/forklift set. Say so explicitly in any report; never
present COCO8 numbers as if they reflect warehouse performance.
"""

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
    """
    Run Ultralytics' validation pipeline and pull out the metrics we care
    about. `data_yaml` is any Ultralytics-format dataset config
    ("coco8.yaml" ships with the library and auto-downloads).
    """
    model = YOLO(model_name)
    metrics = model.val(data=data_yaml, verbose=False)

    return {
        "model_name": model_name,
        "dataset": data_yaml,
        "precision": round(float(metrics.box.mp), 4),   # mean precision across classes
        "recall": round(float(metrics.box.mr), 4),       # mean recall across classes
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
