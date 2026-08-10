"""
Run the detector on a folder of images or a video file and save annotated
output. This is the "does it actually work, visually" sanity check —
NOT the evaluation harness (that's evaluate.py, which produces real
precision/recall/mAP numbers).

Usage:
    python scripts/run_detection.py --source path/to/images_or_video --out output_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `perception` is importable without setting PYTHONPATH

import cv2
import yaml

from perception.detection.detector import Detector


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def draw_detections(image, frame_detections) -> None:
    """Draw boxes + labels directly onto `image` (in place)."""
    for det in frame_detections.detections:
        x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def run_on_images(detector: Detector, source_dir: Path, out_dir: Path) -> None:
    image_paths = sorted(
        p for p in source_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        frame_detections = detector.detect(image)
        draw_detections(image, frame_detections)

        out_path = out_dir / f"annotated_{image_path.name}"
        cv2.imwrite(str(out_path), image)
        print(f"{image_path.name}: {len(frame_detections.detections)} detections -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Folder of images")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "annotated")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "detection.yaml")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    detector = Detector(
        model_name=config["model_name"],
        target_classes=config["target_classes"],
        confidence_threshold=config["confidence_threshold"],
    )

    run_on_images(detector, args.source, args.out)


if __name__ == "__main__":
    main()
