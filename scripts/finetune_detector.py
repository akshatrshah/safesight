"""Fine-tunes a pretrained YOLO checkpoint on my own labeled dataset, with resume support.

Ultralytics already saves last.pt and best.pt after every single epoch by
default, that part isn't something I had to build. What was actually
missing was a way to RESUME from an interrupted run instead of starting
over from epoch 0, and a way to save periodic checkpoints beyond just
last/best, in case I want to inspect an earlier epoch specifically.

Usage:
    Fresh run:
        python scripts/finetune_detector.py --data path/to/data.yaml --epochs 40

    Resume after a Ctrl+C or a crash:
        python scripts/finetune_detector.py --resume --resume-from runs/detect/train/weights/last.pt
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
    parser.add_argument("--data", type=Path, help="Path to the dataset's data.yaml (not needed when resuming)")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8,
                         help="Parallel dataloader processes. Ultralytics defaults to 8, which competes hard for CPU on a laptop. "
                              "Drop to 2-4 if running in the background while using the machine for other things.")
    parser.add_argument("--device", type=str, default=None,
                         help="'mps' for Apple Silicon, 'cpu', or '0' for a CUDA GPU. Default lets Ultralytics auto-detect.")
    parser.add_argument("--save-period", type=int, default=5,
                         help="Save a numbered checkpoint every N epochs, in addition to last.pt/best.pt every epoch. -1 disables this.")
    parser.add_argument("--erasing", type=float, default=0.5,
                         help="Probability of randomly erasing a patch of each training image, this directly simulates occlusion, "
                              "like something loaded on a forklift's prongs changing its usual silhouette. Default 0.4 in Ultralytics, I bumped it slightly.")
    parser.add_argument("--scale", type=float, default=0.7,
                         help="Random scale augmentation range, helps the model generalize across a forklift appearing at different "
                              "sizes/distances, and indirectly across silhouette changes from cargo. Default 0.5, I increased it for more variation.")
    parser.add_argument("--mixup", type=float, default=0.1,
                         help="Probability of blending two training images together. A real, standard technique for improving "
                              "robustness to partial occlusion. Default 0.0 (off), I turned it on modestly.")
    parser.add_argument("--patience", type=int, default=20,
                         help="Stop early if validation performance hasn't improved in this many epochs. Was hardcoded to 10, "
                              "raised the default since more patience genuinely helps when I have the time to spend.")
    parser.add_argument("--freeze", type=int, default=None,
                         help="Freeze this many backbone layers for the whole run. For a proper two-stage approach, run once "
                              "with --freeze 10 --epochs 15 first (fast, trains only the detection head), then run again "
                              "WITHOUT --freeze, using --base-model pointed at that run's best.pt, to unfreeze and fine-tune "
                              "the whole network. Two-stage transfer learning, genuinely helps on a small dataset like mine.")
    parser.add_argument("--tta", action="store_true",
                         help="Use test-time augmentation for the final validation pass, averages predictions across multiple "
                              "augmented views of each image. Free accuracy improvement, no retraining needed, slightly slower to evaluate.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run instead of starting fresh")
    parser.add_argument("--resume-from", type=Path, default=None,
                         help="Path to the last.pt to resume from. Required if --resume is set.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "finetune_results.json")
    args = parser.parse_args()

    if args.resume:
        if args.resume_from is None or not args.resume_from.exists():
            raise FileNotFoundError(
                f"--resume needs --resume-from pointing at an existing last.pt, got: {args.resume_from}"
            )
        print(f"Resuming training from {args.resume_from} ...")
        model = YOLO(str(args.resume_from))
        # resume=True tells Ultralytics to read the original run's saved
        # config (data, epochs, imgsz, etc) and continue from the last
        # completed epoch, rather than starting a new run from scratch.
        results = model.train(resume=True)

    else:
        if args.data is None or not args.data.exists():
            raise FileNotFoundError(
                f"Dataset config not found at {args.data}. Run download_loco_full.py and prepare_poc_subset.py first, "
                "or pass --resume --resume-from to continue an interrupted run instead."
            )
        print(f"Fine-tuning {args.base_model} on {args.data} for {args.epochs} epochs...")
        model = YOLO(args.base_model)

        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            freeze=args.freeze,
            device=args.device,
            save_period=args.save_period,
            workers=args.workers,
            erasing=args.erasing,
            scale=args.scale,
            mixup=args.mixup,
        )

    metrics = model.val(augment=args.tta)
    weights_dir = Path(results.save_dir) / "weights"
    summary = {
        "base_model": args.base_model,
        "dataset": str(args.data) if args.data else "resumed run, see save_dir",
        "epochs": args.epochs,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "best_weights_path": str(weights_dir / "best.pt"),
        "last_weights_path": str(weights_dir / "last.pt"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Fine-tuning complete ===")
    print(json.dumps(summary, indent=2))
    print(f"\nBest weights: {summary['best_weights_path']}")
    print(f"Last checkpoint (for resuming further): {summary['last_weights_path']}")
    print("If this run gets interrupted next time, resume with:")
    print(f"  python scripts/finetune_detector.py --resume --resume-from {summary['last_weights_path']}")


if __name__ == "__main__":
    main()
