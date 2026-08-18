"""Runs the full pipeline over a video file from the command line, saves annotated output plus a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

from perception.video.video_processor import VideoProcessor


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Path to a video file")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "detection.yaml")
    parser.add_argument("--person-model", type=str, default="yolov8n.pt")
    parser.add_argument("--vehicle-model", type=str, default=None, help="Path to fine-tuned forklift/pallet weights, omit to run person-only")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame, 1 = every frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many processed frames, for a quick test")
    parser.add_argument("--out-video", type=Path, default=REPO_ROOT / "outputs" / "analyzed_video.mp4")
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "experiments" / "video_analysis.json")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Video not found at {args.source}")

    config = load_config(args.config)
    processor = VideoProcessor(
        person_model=args.person_model,
        vehicle_model=args.vehicle_model,
        confidence_threshold=config["confidence_threshold"],
        frame_skip=args.frame_skip,
    )

    print(f"Processing {args.source} ...")
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    result = processor.process_video(args.source, annotated_output_path=args.out_video, max_frames=args.max_frames)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": result.summary(),
        "detection_breakdown": result.detection_breakdown(),
        "frames": [
            {
                "frame_number": fr.frame_number,
                "timestamp_seconds": round(fr.timestamp_seconds, 2),
                "max_risk_level": fr.max_risk_level,
                "tracked_objects": fr.tracked_objects,
                "interactions": fr.interactions,
            }
            for fr in result.frame_results
        ],
    }
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Video analysis complete ===")
    print(json.dumps(result.summary(), indent=2))
    print("\n=== Detection breakdown, per class (debugging view) ===")
    print(json.dumps(result.detection_breakdown(), indent=2))
    print(f"\nAnnotated video: {args.out_video}")
    print(f"Full report: {args.out_json}")


if __name__ == "__main__":
    main()
