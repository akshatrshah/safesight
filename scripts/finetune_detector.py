"""Fine-tunes a pretrained YOLO checkpoint on my own labeled dataset, with resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--erasing", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "finetune_results.json")
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--auto-push-minutes", type=int, default=None,
                         help="Push a checkpoint to GitHub every N minutes, checked at each epoch boundary.")
    args = parser.parse_args()

    def log_progress(trainer) -> None:
        if args.progress_file is not None:
            epoch = min(trainer.epoch + 1, trainer.epochs)
            metrics = trainer.metrics
            elapsed_min = (time.time() - start_time) / 60
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"epoch {epoch}/{trainer.epochs}  "
                f"mAP50={metrics.get('metrics/mAP50(B)', 0):.4f}  "
                f"precision={metrics.get('metrics/precision(B)', 0):.4f}  "
                f"recall={metrics.get('metrics/recall(B)', 0):.4f}  "
                f"elapsed={elapsed_min:.1f}min"
            )
            args.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.progress_file, "a") as f:
                f.write(line + "\n")
            print(f"[progress] {line}")

        if args.auto_push_minutes is not None:
            minutes_since_last_push = (time.time() - last_push_time[0]) / 60
            if minutes_since_last_push >= args.auto_push_minutes:
                print(f"[auto-push] {minutes_since_last_push:.1f} min since last push, pushing checkpoint now...")
                push_script = REPO_ROOT / "scripts" / "push_checkpoint_now.sh"
                result = subprocess.run(["bash", str(push_script)], capture_output=True, text=True)
                print(result.stdout)
                if result.returncode != 0:
                    print(f"[auto-push] push failed, will retry at the next epoch boundary: {result.stderr}")
                last_push_time[0] = time.time()

    start_time = time.time()
    last_push_time = [start_time]

    if args.resume:
        if args.resume_from is None or not args.resume_from.exists():
            raise FileNotFoundError(f"--resume needs --resume-from, got: {args.resume_from}")
        print(f"Resuming training from {args.resume_from} ...")
        model = YOLO(str(args.resume_from))
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(resume=True)
    else:
        if args.data is None or not args.data.exists():
            raise FileNotFoundError(f"Dataset config not found at {args.data}.")
        print(f"Fine-tuning {args.base_model} on {args.data} for {args.epochs} epochs...")
        model = YOLO(args.base_model)
        model.add_callback("on_fit_epoch_end", log_progress)
        results = model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, freeze=args.freeze, device=args.device,
            save_period=args.save_period, workers=args.workers,
            erasing=args.erasing, scale=args.scale, mixup=args.mixup,
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


if __name__ == "__main__":
    main()
