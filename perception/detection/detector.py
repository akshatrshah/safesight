"""
Thin wrapper around an Ultralytics YOLO model.

WHY WRAP IT AT ALL
--------------------
Ultralytics' `model(image)` call returns a library-specific `Results`
object with its own API. If every other module in this project (tracking,
pose, risk scoring) depends directly on Ultralytics' object shape, then
swapping detectors later (e.g. to a fine-tuned model, or a different
architecture entirely) means touching every downstream module.

Instead, this wrapper converts YOLO's output into one simple, boring,
stable data structure (`Detection`) that the rest of the system depends
on. This is the same reason you'd put an interface between a service and
a specific database driver in backend work — same idea, applied to a
model instead of a database.

WHAT YOLO IS ACTUALLY DOING (practical level)
-----------------------------------------------
YOLO ("You Only Look Once") is a single-stage detector: it looks at the
whole image once and, in a single forward pass, predicts a set of
candidate boxes with class probabilities and "objectness" (confidence
that *something* is there) directly, rather than first proposing regions
and then classifying each region separately (that two-stage approach is
what older detectors like Faster R-CNN do). Single-stage is faster and
generally slightly less accurate on small/crowded objects than two-stage
alternatives, which is exactly the accuracy-vs-speed tradeoff worth being
able to discuss in an interview.

The raw output is many overlapping candidate boxes; confidence
thresholding + NMS (see perception/utils/geometry.py) reduces that down
to one box per real object. Ultralytics does this internally for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class Detection:
    """One detected object in one frame, in our own stable schema."""

    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coords


@dataclass
class FrameDetections:
    """All detections for a single frame."""

    frame_index: int
    detections: list[Detection] = field(default_factory=list)


class Detector:
    """
    Loads a YOLO model and runs inference, filtered to a target set of
    classes and a minimum confidence.

    `model_name` accepts any Ultralytics checkpoint name or path, e.g.
    "yolov8n.pt" (nano, fastest/least accurate) or "yolov8m.pt" (medium,
    slower/more accurate). Weights download automatically on first use.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        target_classes: list[str] | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.model = YOLO(model_name)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        # None -> keep all classes the model knows about.
        self.target_classes = set(target_classes) if target_classes else None
        # Ultralytics exposes the class-id -> class-name mapping on the model.
        self._class_names: dict[int, str] = self.model.names

    def detect(self, image) -> FrameDetections:
        """
        Run detection on a single image (numpy array, HWC, BGR — the
        format OpenCV reads images in) or an image file path.
        """
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
