"""YOLOv8-pose wrapper, plus a joint-angle helper built on basic vector math."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PersonPose:
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    box_xyxy: tuple[float, float, float, float] | None = None

    def get(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)


def joint_angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float | None:
    """Angle at joint b, formed by a-b-c, via the dot product formula. hip/knee/ankle gives knee bend."""
    MIN_CONFIDENCE = 0.3
    if min(a.confidence, b.confidence, c.confidence) < MIN_CONFIDENCE:
        return None

    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


class PoseEstimator:
    def __init__(self, model_name: str = "yolov8n-pose.pt", confidence_threshold: float = 0.35) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def estimate(self, image) -> list[PersonPose]:
        results = self.model.predict(source=image, conf=self.confidence_threshold, verbose=False)
        result = results[0]

        poses: list[PersonPose] = []
        if result.keypoints is None:
            return poses

        for person_idx in range(len(result.keypoints)):
            kpts_xy = result.keypoints.xy[person_idx].tolist()
            kpts_conf = result.keypoints.conf[person_idx].tolist() if result.keypoints.conf is not None else [1.0] * 17

            keypoints = {
                name: Keypoint(name=name, x=xy[0], y=xy[1], confidence=conf)
                for name, xy, conf in zip(KEYPOINT_NAMES, kpts_xy, kpts_conf)
            }

            box_xyxy = None
            if result.boxes is not None and person_idx < len(result.boxes):
                box_xyxy = tuple(result.boxes[person_idx].xyxy[0].tolist())

            poses.append(PersonPose(keypoints=keypoints, box_xyxy=box_xyxy))

        return poses
