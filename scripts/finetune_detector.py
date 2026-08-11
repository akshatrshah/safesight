"""
Fine-tune a pretrained YOLO detector on a custom, real, labeled dataset
(e.g. a Roboflow-exported forklift/warehouse dataset).

WHAT "FINE-TUNING" ACTUALLY MEANS, CONCRETELY
--------------------------------------------------
`Detector`/`evaluate.py` so far have only ever LOADED pretrained COCO
weights and run inference/evaluation — no training happened, and the
model has literally never seen a forklift, because COCO doesn't have
that class.

Fine-tuning means: start from those same pretrained weights (which
already know general visual patterns — edges, shapes, textures) and
continue training, running real gradient descent, on YOUR new labeled
images. This is much faster and needs far less data than training from
random weights, because the network isn't learning to see from zero —
it's adapting what it already knows to recognize new categories.

EXPECTED INPUT FORMAT
--------------------------
This script expects a Roboflow-exported "YOLOv8" format dataset:
    your_dataset/
      data.yaml          <- class names + train/val/test image paths
      train/images/, train/labels/
      valid/images/, valid/labels/
      test/images/, test/labels/   (optional)

To get this: create a free Roboflow account, pick a dataset (see
docs/FINE_TUNING.md for real dataset recommendations), and use the
"Download Dataset" button, choosing the YOLOv8 format. Point
--data at the data.yaml file it gives you.

WHAT TO WATCH FOR (OVERFITTING)
------------------------------------
After training, Ultralytics saves a results.png / results.csv under
runs/detect/train*/ showing train loss AND validation loss per epoch.
If train loss keeps dropping but validation loss starts climbing back
up, that's the overfitting signature (see FOUNDATIONS.md) — the model
is memorizing the training images rather than generalizing. If you see
that, stop training earlier (fewer epochs) or get more training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset's data.yaml (from Roboflow export)")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None,
                         help="Force a device, e.g. 'mps' for Apple Silicon GPU, 'cpu', or '0' for a CUDA GPU. "
                              "Default (None) lets Ultralytics auto-detect, which doesn't always pick MPS correctly.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "finetune_results.json")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {args.data}. "
            "Export a dataset from Roboflow in YOLOv8 format and point --data at its data.yaml — "
            "see docs/FINE_TUNING.md for dataset recommendations and step-by-step instructions."
        )

    print(f"Fine-tuning {args.base_model} on {args.data} for {args.epochs} epochs...")
    model = YOLO(args.base_model)

    # This is the actual training call — real gradient descent runs here,
    # adjusting the pretrained weights based on your labeled images.
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,  # stop early if validation performance stalls for 10 epochs (a simple overfitting guard)
        device=args.device,
    )

    # Evaluate the fine-tuned model on its held-out validation split —
    # same metrics as perception/detection/evaluate.py, now on YOUR classes.
    metrics = model.val()
    summary = {
        "base_model": args.base_model,
        "dataset": str(args.data),
        "epochs": args.epochs,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "best_weights_path": str(Path(results.save_dir) / "weights" / "best.pt"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Fine-tuning complete ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFine-tuned weights saved at: {summary['best_weights_path']}")
    print("Use this path as `model_name` in configs/detection.yaml to use your fine-tuned model everywhere else in the project.")


if __name__ == "__main__":
    main()