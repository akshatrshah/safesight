"""
Multi-object tracking: turns per-frame detections into persistent
identities with a position history over time.

WHY THIS SITS ON TOP OF DETECTION, NOT INSIDE IT
----------------------------------------------------
Detection (perception/detection/detector.py) answers "what's in this
frame" with zero memory of any other frame. Tracking answers "which
object is this, across many frames" by matching new detections to
existing tracks. We use Ultralytics' built-in `.track()` method, which
runs the same YOLO detector under the hood and then applies ByteTrack
(a tracking-by-detection algorithm) to assign persistent IDs.

WHAT BYTETRACK ACTUALLY DOES (practical level)
--------------------------------------------------
For each new frame:
  1. Predict where each existing track SHOULD be, based on its recent
     motion (this prediction step uses a Kalman filter internally).
  2. Compare that prediction against the new frame's actual detections
     using IoU — the closest match gets matched to that track ID.
  3. Unmatched new detections start new tracks. Unmatched existing
     tracks that go too long without a match are dropped.
ByteTrack's specific contribution: it does this matching in TWO passes —
high-confidence detections first, then a second pass using LOW-confidence
detections that would normally be thrown away. This helps recover tracks
during brief occlusion (e.g. a worker briefly blocked by a forklift),
where the detector often still produces a low-confidence box rather than
no box at all.

WHY WE ALSO KEEP OUR OWN TrackHistory
-----------------------------------------
Ultralytics gives us an ID per frame, but for computing velocity,
trajectory, and time-to-collision (used by perception/risk/risk_model.py)
we need each track's recent POSITION HISTORY, not just its current
position. TrackHistory is a small in-memory store we maintain ourselves
for exactly that purpose.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class TrackedDetection:
    """One tracked object's state in the CURRENT frame."""

    track_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass
class TrackSnapshot:
    """One historical (position, time) sample for a single track."""

    position: tuple[float, float]
    timestamp: float


class TrackHistory:
    """
    Keeps a rolling window of recent (position, timestamp) samples per
    track ID. This is the "memory" that turns single-point positions
    into a usable trajectory.
    """

    def __init__(self, max_history: int = 30) -> None:
        self.max_history = max_history
        self._tracks: dict[int, deque[TrackSnapshot]] = {}

    def update(self, track_id: int, position: tuple[float, float], timestamp: float | None = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        if track_id not in self._tracks:
            self._tracks[track_id] = deque(maxlen=self.max_history)
        self._tracks[track_id].append(TrackSnapshot(position=position, timestamp=timestamp))

    def get_history(self, track_id: int) -> list[TrackSnapshot]:
        return list(self._tracks.get(track_id, []))

    def velocity(self, track_id: int) -> tuple[float, float] | None:
        """
        Estimate current (vx, vy) in pixels/second from the two most
        recent samples. Returns None if there isn't enough history yet.
        """
        history = self._tracks.get(track_id)
        if history is None or len(history) < 2:
            return None

        prev, curr = history[-2], history[-1]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            return None

        vx = (curr.position[0] - prev.position[0]) / dt
        vy = (curr.position[1] - prev.position[1]) / dt
        return (vx, vy)

    def active_track_ids(self) -> list[int]:
        return list(self._tracks.keys())

    def prune(self, active_ids: set[int]) -> None:
        """Drop history for tracks that are no longer being tracked."""
        for track_id in list(self._tracks.keys()):
            if track_id not in active_ids:
                del self._tracks[track_id]


class Tracker:
    """
    Wraps Ultralytics' YOLO + ByteTrack. Call `update(frame)` once per
    video frame, in order — track IDs only stay consistent if frames are
    fed in sequence (this is stateful, unlike Detector.detect()).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        target_classes: list[str] | None = None,
        confidence_threshold: float = 0.35,
        max_history: int = 30,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.target_classes = set(target_classes) if target_classes else None
        self._class_names: dict[int, str] = self.model.names
        self.history = TrackHistory(max_history=max_history)

    def update(self, frame, timestamp: float | None = None) -> list[TrackedDetection]:
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            persist=True,          # keep ID assignment state between calls
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]

        tracked: list[TrackedDetection] = []
        active_ids: set[int] = set()

        if result.boxes is None or result.boxes.id is None:
            # No tracks yet this frame (e.g. nothing detected, or IDs not
            # assigned on the very first frame in rare cases).
            return tracked

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = self._class_names[class_id]
            if self.target_classes is not None and class_name not in self.target_classes:
                continue

            track_id = int(box.id.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            tracked.append(
                TrackedDetection(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    box_xyxy=(x1, y1, x2, y2),
                    center=center,
                )
            )
            active_ids.add(track_id)
            self.history.update(track_id, center, timestamp)

        self.history.prune(active_ids)
        return tracked
