"""
End-to-end integration demo: tracking -> pose -> zone check -> risk
scoring, all wired together on real detections from a real image.

IMPORTANT HONESTY NOTE
---------------------------
We do not yet have a fine-tuned forklift detector (that's Milestone 2 —
"forklift" isn't a native COCO class). So this demo uses two tracked
PEOPLE as a stand-in for a "worker" and a "vehicle" purely to prove the
pipeline wiring works end-to-end — position -> velocity -> zone
membership -> interaction risk score. This is explicitly a plumbing
demo, not a claim that the system currently detects real forklift
hazards. Milestone 2's fine-tuned detector is what would make the
"vehicle" role real instead of a stand-in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `perception` is importable without setting PYTHONPATH

import cv2
import yaml

from perception.tracking.tracker import Tracker
from perception.pose.pose_estimator import PoseEstimator
from perception.spatial.zones import Zone, ZoneManager
from perception.risk.risk_model import compute_interaction_features, HeuristicRiskModel


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Single image to run the demo on")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "detection.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "pipeline_demo_output.json")
    args = parser.parse_args()

    config = load_config(args.config)
    frame = cv2.imread(str(args.source))
    if frame is None:
        raise FileNotFoundError(f"Could not read image at {args.source}")

    # --- Tracking (feed the same frame 3x to build up velocity history —
    # a real video would feed genuinely consecutive frames here) ---
    tracker = Tracker(model_name=config["model_name"], target_classes=["person"], confidence_threshold=config["confidence_threshold"])
    for i in range(3):
        tracked = tracker.update(frame, timestamp=float(i))

    print(f"Tracked {len(tracked)} people: ids={[t.track_id for t in tracked]}")

    # --- Pose estimation, per tracked person ---
    pose_estimator = PoseEstimator(confidence_threshold=config["confidence_threshold"])
    poses = pose_estimator.estimate(frame)
    print(f"Estimated pose for {len(poses)} people")

    # --- Zone check: an illustrative "restricted area" covering the left third of the frame ---
    frame_height, frame_width = frame.shape[:2]
    restricted_zone = Zone(
        name="restricted_area",
        polygon=[(0, 0), (frame_width / 3, 0), (frame_width / 3, frame_height), (0, frame_height)],
        zone_type="restricted",
    )
    zone_manager = ZoneManager([restricted_zone])

    events = []
    for t in tracked:
        zones = zone_manager.zone_names_containing(t.center)
        if zones:
            print(f"  track {t.track_id} is inside zone(s): {zones}")
        events.append({"track_id": t.track_id, "center": t.center, "zones": zones})

    # --- Risk: pairwise between every 2 tracked people, one stand-in as "vehicle" ---
    # (See module docstring — this is a wiring demo, not a real forklift risk claim.)
    risk_model = HeuristicRiskModel()
    risk_events = []
    if len(tracked) >= 2:
        for i in range(len(tracked)):
            for j in range(i + 1, len(tracked)):
                a, b = tracked[i], tracked[j]
                vel_a = tracker.history.velocity(a.track_id) or (0.0, 0.0)
                vel_b = tracker.history.velocity(b.track_id) or (0.0, 0.0)

                features = compute_interaction_features(
                    forklift_position=a.center, forklift_velocity=vel_a,
                    person_position=b.center, person_velocity=vel_b,
                    person_in_forklift_zone=bool(zone_manager.zone_names_containing(b.center)),
                )
                risk_level = risk_model.classify(features)
                print(f"  interaction track {a.track_id} <-> track {b.track_id}: "
                      f"distance={features.distance:.1f}px, risk={risk_level.value}")
                risk_events.append({
                    "track_a": a.track_id, "track_b": b.track_id,
                    "distance_px": round(features.distance, 1),
                    "risk_level": risk_level.value,
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"tracked_objects": events, "interactions": risk_events}, f, indent=2)
    print(f"\nFull output written to {args.out}")


if __name__ == "__main__":
    main()
