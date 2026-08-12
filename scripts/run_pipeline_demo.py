"""Wires tracking, pose, zones, and risk together end to end.

Runs two models side by side: pretrained yolov8n.pt for person detection,
plus my fine-tuned forklift/pallet model, merged with MultiModelTracker.
This is a real worker-vehicle interaction now, not the two-people stand-in
I used before I had a working forklift detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import yaml

from perception.tracking.multi_model_tracker import MultiModelTracker, ModelSource
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

    multi_tracker = MultiModelTracker([
        ModelSource(model_name="yolov8n.pt", target_classes=["person"], confidence_threshold=config["confidence_threshold"], id_offset=0),
        ModelSource(model_name=config["model_name"], target_classes=["forklift", "pallet"], confidence_threshold=config["confidence_threshold"], id_offset=100000),
    ])

    for i in range(3):
        tracked = multi_tracker.update(frame, timestamp=float(i))

    people = [t for t in tracked if t.class_name == "person"]
    forklifts = [t for t in tracked if t.class_name == "forklift"]
    print(f"Tracked {len(people)} people, {len(forklifts)} forklifts, {len(tracked)} objects total")
    for t in tracked:
        print(f"  id={t.track_id} class={t.class_name} conf={t.confidence:.2f}")

    pose_estimator = PoseEstimator(confidence_threshold=config["confidence_threshold"])
    poses = pose_estimator.estimate(frame)
    print(f"Estimated pose for {len(poses)} people")

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
            print(f"  track {t.track_id} ({t.class_name}) is inside zone(s): {zones}")
        events.append({"track_id": t.track_id, "class_name": t.class_name, "center": t.center, "zones": zones})

    # Real interactions now: every forklift paired with every person, not person-vs-person.
    risk_model = HeuristicRiskModel()
    risk_events = []
    for forklift in forklifts:
        for person in people:
            vel_forklift = multi_tracker.velocity(forklift.track_id) or (0.0, 0.0)
            vel_person = multi_tracker.velocity(person.track_id) or (0.0, 0.0)

            features = compute_interaction_features(
                forklift_position=forklift.center, forklift_velocity=vel_forklift,
                person_position=person.center, person_velocity=vel_person,
                person_in_forklift_zone=bool(zone_manager.zone_names_containing(person.center)),
            )
            risk_level = risk_model.classify(features)
            print(f"  forklift {forklift.track_id} <-> person {person.track_id}: "
                  f"distance={features.distance:.1f}px, risk={risk_level.value}")
            risk_events.append({
                "forklift_id": forklift.track_id, "person_id": person.track_id,
                "distance_px": round(features.distance, 1),
                "risk_level": risk_level.value,
            })

    if not forklifts:
        print("  no forklifts detected in this frame, no interactions to score")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"tracked_objects": events, "interactions": risk_events}, f, indent=2)
    print(f"\nFull output written to {args.out}")


if __name__ == "__main__":
    main()
