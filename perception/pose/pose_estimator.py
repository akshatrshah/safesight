"""
Human pose estimation: for each detected person, estimate body keypoint
locations (head, shoulders, elbows, hips, knees, ankles, etc).

WHY YOLOv8-POSE, AND WHAT IT ACTUALLY OUTPUTS
--------------------------------------------------
We use Ultralytics' pose variant (yolov8n-pose.pt) rather than a
separate pose library (e.g. MediaPipe, RTMPose/MMPose) for this
milestone specifically because it shares the same detection backbone
and API we already built Detector/Tracker around — one less new
dependency, one less new output format to wrap. It's trained on COCO's
17-keypoint skeleton format:

    0: nose            6: right shoulder    12: right hip
    1: left eye        7: left elbow        13: left knee
    2: right eye       8: right elbow       14: right knee
    3: left ear        9: left wrist        15: left ankle
    4: right ear       10: right wrist      16: right ankle
    5: left shoulder   11: left hip

Each keypoint comes with its own (x, y) pixel location AND its own
confidence — the model might be very sure where someone's nose is but
much less sure where an occluded ankle is, and that per-joint
uncertainty is useful information downstream, not something to discard.

A NOTE ON SWAPPING IN A DIFFERENT POSE MODEL LATER
-------------------------------------------------------
If RTMPose/MediaPipe are swapped in later for accuracy or speed reasons,
only this file needs to change — same "stable wrapper" pattern as
Detector and Tracker. Downstream code (activity recognition) depends on
our own `PersonPose` schema, never on Ultralytics' raw output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ultralytics import YOLO

# COCO 17-keypoint index names, in the order Ultralytics returns them.
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
    """All keypoints for one detected person in one frame."""

    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    box_xyxy: tuple[float, float, float, float] | None = None

    def get(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)


def joint_angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float | None:
    """
    Angle at joint `b`, formed by the three points a-b-c, in degrees.

    Example: joint_angle(hip, knee, ankle) gives the knee bend angle —
    close to 180 degrees means a straight leg, a much smaller angle
    means a sharply bent knee (useful signal for "crouching"/"lifting").

    Returns None if any of the three keypoints has near-zero confidence
    (the model essentially didn't find that joint, so the angle would be
    meaningless).
    """
    MIN_CONFIDENCE = 0.3
    if min(a.confidence, b.confidence, c.confidence) < MIN_CONFIDENCE:
        return None

    # Two vectors from the middle point b, out to a and out to c.
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))  # clamp for float safety
    return math.degrees(math.acos(cos_angle))


class PoseEstimator:
    """Wraps a YOLOv8-pose model, returns one PersonPose per detected person."""

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
            kpts_xy = result.keypoints.xy[person_idx].tolist()       # [[x,y], ...] x17
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
