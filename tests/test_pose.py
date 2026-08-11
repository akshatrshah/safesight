import pytest

from perception.pose.pose_estimator import Keypoint, joint_angle


def test_joint_angle_straight_line_is_180_degrees():
    # a - b - c in a perfectly straight vertical line -> 180 degree angle at b.
    a = Keypoint(name="hip", x=0, y=0, confidence=1.0)
    b = Keypoint(name="knee", x=0, y=10, confidence=1.0)
    c = Keypoint(name="ankle", x=0, y=20, confidence=1.0)
    angle = joint_angle(a, b, c)
    assert abs(angle - 180.0) < 1e-6


def test_joint_angle_right_angle_bend():
    # b at origin, a straight up, c straight right -> 90 degree angle at b.
    a = Keypoint(name="hip", x=0, y=-10, confidence=1.0)
    b = Keypoint(name="knee", x=0, y=0, confidence=1.0)
    c = Keypoint(name="ankle", x=10, y=0, confidence=1.0)
    angle = joint_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-6


def test_joint_angle_none_when_low_confidence():
    a = Keypoint(name="hip", x=0, y=0, confidence=0.05)  # below the 0.3 threshold
    b = Keypoint(name="knee", x=0, y=10, confidence=1.0)
    c = Keypoint(name="ankle", x=0, y=20, confidence=1.0)
    assert joint_angle(a, b, c) is None


def test_pose_estimator_returns_expected_schema():
    import cv2
    from pathlib import Path
    from perception.pose.pose_estimator import PoseEstimator

    sample_image = Path(__file__).resolve().parent.parent / "sample_data" / "bus.jpg"
    image = cv2.imread(str(sample_image))

    estimator = PoseEstimator(model_name="yolov8n-pose.pt", confidence_threshold=0.35)
    poses = estimator.estimate(image)

    assert len(poses) > 0   # bus.jpg has multiple people in it
    for pose in poses:
        assert "nose" in pose.keypoints
        assert "left_ankle" in pose.keypoints
        nose = pose.get("nose")
        assert 0.0 <= nose.confidence <= 1.0
