"""Runs two (or more) Tracker instances side by side and merges their output into one list.

I need this because fine-tuning on a 2-class dataset (forklift, pallet)
replaced the model's detection head, so best.pt genuinely cannot detect
people anymore, that knowledge didn't get "added to", it got overwritten.
The pretrained yolov8n.pt never lost anything though, it still detects
people fine. So instead of one model doing everything, I run both models
on the same frame and combine their results.

Each underlying Tracker assigns its own track IDs starting from small
numbers, so two separate models would both hand out id=1, id=2, etc,
colliding with each other once merged. I fix that by adding a distinct
offset per model, so IDs stay globally unique across the merged output.
"""

from __future__ import annotations

from perception.tracking.tracker import Tracker, TrackedDetection


class ModelSource:
    """One model's config: which weights, which classes, and its ID offset."""

    def __init__(self, model_name: str, target_classes: list[str], confidence_threshold: float, id_offset: int) -> None:
        self.tracker = Tracker(model_name=model_name, target_classes=target_classes, confidence_threshold=confidence_threshold)
        self.id_offset = id_offset


class MultiModelTracker:
    """Runs several Tracker instances on the same frame, merges results with non-colliding IDs."""

    def __init__(self, sources: list[ModelSource]) -> None:
        self.sources = sources

    def update(self, frame, timestamp: float | None = None) -> list[TrackedDetection]:
        merged: list[TrackedDetection] = []
        for source in self.sources:
            tracked = source.tracker.update(frame, timestamp=timestamp)
            for t in tracked:
                # Offset the ID so it can never collide with another source's IDs,
                # but the underlying tracker's own history still uses the RAW id
                # internally, so I only shift it on the way out.
                merged.append(
                    TrackedDetection(
                        track_id=t.track_id + source.id_offset,
                        class_name=t.class_name,
                        confidence=t.confidence,
                        box_xyxy=t.box_xyxy,
                        center=t.center,
                    )
                )
        return merged

    def velocity(self, offset_track_id: int) -> tuple[float, float] | None:
        """Same idea as Tracker.history.velocity(), but works with the merged, offset track_id."""
        for source in self.sources:
            if offset_track_id >= source.id_offset and (offset_track_id - source.id_offset) in source.tracker.history.active_track_ids():
                raw_id = offset_track_id - source.id_offset
                return source.tracker.history.velocity(raw_id)
        return None
