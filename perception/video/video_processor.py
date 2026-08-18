"""Runs the full perception pipeline over an actual video stream, not one static image.

This is the missing piece from everything before it. Detection, tracking,
pose, zones, and risk all already work on a single frame, what was
missing was the actual video loop: cv2.VideoCapture reading frames one
at a time, feeding each into the SAME tracker instance so track IDs and
velocity genuinely persist across real time, not a repeated still frame.

This is also where time to collision finally means something real. Every
earlier demo fed the same image to the tracker multiple times just to
get SOME velocity value out of it, here the velocity comes from actual
motion between real, different, consecutive frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from perception.tracking.multi_model_tracker import MultiModelTracker, ModelSource
from perception.pose.pose_estimator import PoseEstimator
from perception.spatial.zones import Zone, ZoneManager
from perception.risk.risk_model import compute_interaction_features, HeuristicRiskModel


@dataclass
class FrameResult:
    frame_number: int
    timestamp_seconds: float
    tracked_objects: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    max_risk_level: str = "LOW"


@dataclass
class VideoAnalysisResult:
    source_path: str
    fps: float
    total_frames_in_video: int
    frames_processed: int
    frame_results: list[FrameResult] = field(default_factory=list)

    def risk_timeline(self) -> list[tuple[float, str]]:
        """(timestamp_seconds, max_risk_level) for every processed frame, for plotting."""
        return [(fr.timestamp_seconds, fr.max_risk_level) for fr in self.frame_results]

    def high_risk_frames(self) -> list[FrameResult]:
        return [fr for fr in self.frame_results if fr.max_risk_level == "HIGH"]

    def detection_breakdown(self) -> dict:
        """
        Per-class summary across the whole video: how many distinct objects
        were tracked, how often each class appeared frame to frame, and
        confidence stats. This is the debugging view, does the forklift
        count look right, is confidence suspiciously low, is a class
        flickering in and out (high frame_appearances but low
        unique_objects_tracked would mean one real object, tracked
        stably, appearing in many frames, which is what you WANT to see;
        the opposite pattern, many unique IDs but few frame appearances
        each, usually means the tracker is losing and re-acquiring the
        same object repeatedly, worth investigating if you see it).
        """
        from collections import defaultdict

        class_stats: dict[str, dict] = defaultdict(lambda: {"track_ids": set(), "confidences": [], "frame_appearances": 0})
        for fr in self.frame_results:
            for obj in fr.tracked_objects:
                stats = class_stats[obj["class_name"]]
                stats["track_ids"].add(obj["track_id"])
                stats["confidences"].append(obj["confidence"])
                stats["frame_appearances"] += 1

        breakdown = {}
        for class_name, stats in class_stats.items():
            confs = stats["confidences"]
            breakdown[class_name] = {
                "unique_objects_tracked": len(stats["track_ids"]),
                "total_frame_appearances": stats["frame_appearances"],
                "confidence_min": round(min(confs), 3) if confs else None,
                "confidence_mean": round(sum(confs) / len(confs), 3) if confs else None,
                "confidence_max": round(max(confs), 3) if confs else None,
            }
        return breakdown

    def per_frame_object_counts(self) -> list[dict]:
        """(timestamp, class -> count) per frame, for plotting object count stability over time."""
        rows = []
        for fr in self.frame_results:
            counts: dict[str, int] = {}
            for obj in fr.tracked_objects:
                counts[obj["class_name"]] = counts.get(obj["class_name"], 0) + 1
            rows.append({"timestamp_seconds": fr.timestamp_seconds, **counts})
        return rows

    def summary(self) -> dict:
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for fr in self.frame_results:
            risk_counts[fr.max_risk_level] += 1
        return {
            "source": self.source_path,
            "frames_processed": self.frames_processed,
            "duration_seconds": round(self.frames_processed / self.fps, 1) if self.fps else 0,
            "risk_frame_counts": risk_counts,
            "high_risk_moments": [round(fr.timestamp_seconds, 1) for fr in self.high_risk_frames()],
        }


class VideoProcessor:
    """
    Wraps the full pipeline (tracking, pose, zones, risk) and runs it over
    every frame of a real video file. frame_skip lets me process every
    Nth frame instead of every single one, since full-rate processing on
    a CPU/laptop can be slower than the video's own playback speed.
    """

    def __init__(
        self,
        person_model: str = "yolov8n.pt",
        vehicle_model: str | None = None,
        vehicle_classes: list[str] | None = None,
        confidence_threshold: float = 0.35,
        frame_skip: int = 1,
        restricted_zone_fraction: float = 1 / 3,
    ) -> None:
        sources = [ModelSource(model_name=person_model, target_classes=["person"], confidence_threshold=confidence_threshold, id_offset=0)]
        if vehicle_model:
            sources.append(
                ModelSource(
                    model_name=vehicle_model,
                    target_classes=vehicle_classes or ["forklift", "pallet"],
                    confidence_threshold=confidence_threshold,
                    id_offset=100000,
                )
            )
        self.tracker = MultiModelTracker(sources)
        self.pose_estimator = PoseEstimator(confidence_threshold=confidence_threshold)
        self.risk_model = HeuristicRiskModel()
        self.frame_skip = max(1, frame_skip)
        self.restricted_zone_fraction = restricted_zone_fraction
        self._zone_manager: ZoneManager | None = None

    def _build_zone_manager(self, frame_width: int, frame_height: int) -> ZoneManager:
        restricted = Zone(
            name="restricted_area",
            polygon=[
                (0, 0),
                (frame_width * self.restricted_zone_fraction, 0),
                (frame_width * self.restricted_zone_fraction, frame_height),
                (0, frame_height),
            ],
            zone_type="restricted",
        )
        return ZoneManager([restricted])

    def process_video(
        self,
        source_path: str | Path,
        annotated_output_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> VideoAnalysisResult:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._zone_manager = self._build_zone_manager(frame_width, frame_height)

        writer = None
        if annotated_output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (frame_width, frame_height))

        result = VideoAnalysisResult(source_path=str(source_path), fps=fps, total_frames_in_video=total_frames, frames_processed=0)

        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % self.frame_skip == 0:
                frame_result = self._process_single_frame(frame, frame_index, frame_index / fps)
                result.frame_results.append(frame_result)
                result.frames_processed += 1

                if writer is not None:
                    annotated = self._draw_annotations(frame, frame_result)
                    writer.write(annotated)
            elif writer is not None:
                writer.write(frame)  # pass through unannotated frames we skipped, so output video stays full length

            frame_index += 1
            if max_frames is not None and result.frames_processed >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()

        return result

    def _process_single_frame(self, frame, frame_index: int, timestamp: float) -> FrameResult:
        tracked = self.tracker.update(frame, timestamp=timestamp)
        people = [t for t in tracked if t.class_name == "person"]
        vehicles = [t for t in tracked if t.class_name in ("forklift", "truck")]

        objects_out = []
        for t in tracked:
            zones = self._zone_manager.zone_names_containing(t.center) if self._zone_manager else []
            objects_out.append({
                "track_id": t.track_id, "class_name": t.class_name,
                "confidence": round(t.confidence, 3), "center": t.center, "zones": zones,
            })

        interactions_out = []
        max_risk = "LOW"
        risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        for vehicle in vehicles:
            for person in people:
                vel_vehicle = self.tracker.velocity(vehicle.track_id) or (0.0, 0.0)
                vel_person = self.tracker.velocity(person.track_id) or (0.0, 0.0)
                in_zone = bool(self._zone_manager.zone_names_containing(person.center)) if self._zone_manager else False

                features = compute_interaction_features(
                    forklift_position=vehicle.center, forklift_velocity=vel_vehicle,
                    person_position=person.center, person_velocity=vel_person,
                    person_in_forklift_zone=in_zone,
                )
                risk_level = self.risk_model.classify(features)
                interactions_out.append({
                    "vehicle_id": vehicle.track_id, "person_id": person.track_id,
                    "distance_px": round(features.distance, 1),
                    "time_to_collision_s": round(features.time_to_collision, 2) if features.time_to_collision else None,
                    "risk_level": risk_level.value,
                })
                if risk_rank[risk_level.value] > risk_rank[max_risk]:
                    max_risk = risk_level.value

        return FrameResult(
            frame_number=frame_index, timestamp_seconds=timestamp,
            tracked_objects=objects_out, interactions=interactions_out, max_risk_level=max_risk,
        )

    def _draw_annotations(self, frame, frame_result: FrameResult):
        canvas = frame.copy()
        risk_colors = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}

        for obj in frame_result.tracked_objects:
            x, y = int(obj["center"][0]), int(obj["center"][1])
            label = f'{obj["class_name"]} #{obj["track_id"]}'
            cv2.circle(canvas, (x, y), 5, (255, 255, 0), -1)
            cv2.putText(canvas, label, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        banner_color = risk_colors.get(frame_result.max_risk_level, (0, 200, 0))
        cv2.putText(canvas, f"risk: {frame_result.max_risk_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2)
        return canvas
