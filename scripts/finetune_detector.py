"""Fine-tunes a pretrained YOLO checkpoint on my own labeled dataset instead of just running inference on COCO weights."""

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
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset's data.yaml")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None,
                         help="'mps' for Apple Silicon, 'cpu', or '0' for a CUDA GPU. Default lets Ultralytics auto-detect.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "finetune_results.json")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {args.data}. Run download_loco.py and convert_loco_to_yolo.py first."
        )

    print(f"Fine-tuning {args.base_model} on {args.data} for {args.epochs} epochs...")
    model = YOLO(args.base_model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,
        device=args.device,
    )

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
    print("Use this path as model_name in configs/detection.yaml to use it everywhere else in the project.")


if __name__ == "__main__":
    main()
