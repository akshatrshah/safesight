"""ByteTrack wrapper, plus my own position history so I can compute velocity for the risk model."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class TrackedDetection:
    track_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass
class TrackSnapshot:
    position: tuple[float, float]
    timestamp: float


class TrackHistory:
    """Rolling window of recent (position, timestamp) samples per track ID."""

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
        """Pixels/second from the two most recent samples. None if there's not enough history yet."""
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
        for track_id in list(self._tracks.keys()):
            if track_id not in active_ids:
                del self._tracks[track_id]


class Tracker:
    """Call update() once per frame, in order, IDs only stay consistent if frames come in sequence."""

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
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]

        tracked: list[TrackedDetection] = []
        active_ids: set[int] = set()

        if result.boxes is None or result.boxes.id is None:
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
