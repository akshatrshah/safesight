"""Wraps Ultralytics' YOLO so the rest of my code depends on my own stable schema, not their internal format."""

from __future__ import annotations

from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass
class FrameDetections:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)


class Detector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        target_classes: list[str] | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.model = YOLO(model_name)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_classes = set(target_classes) if target_classes else None
        self._class_names: dict[int, str] = self.model.names

    def detect(self, image) -> FrameDetections:
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )
        result = results[0]

        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = self._class_names[class_id]

            if self.target_classes is not None and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf.item())

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    box_xyxy=(x1, y1, x2, y2),
                )
            )

        return FrameDetections(frame_index=0, detections=detections)
