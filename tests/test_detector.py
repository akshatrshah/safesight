import numpy as np
import pytest

from perception.detection.detector import Detection, FrameDetections, Detector


@pytest.fixture(scope="module")
def detector() -> Detector:
    # yolov8n is the smallest/fastest checkpoint -> keeps tests fast.
    return Detector(model_name="yolov8n.pt", target_classes=["person", "truck"], confidence_threshold=0.35)


def test_detect_returns_expected_schema(detector: Detector):
    # A blank synthetic frame: we're not asserting WHAT is detected here,
    # only that the wrapper returns our stable schema and doesn't crash
    # on a frame with no real objects in it.
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(blank_frame)

    assert isinstance(result, FrameDetections)
    assert isinstance(result.detections, list)
    for det in result.detections:
        assert isinstance(det, Detection)
        assert det.class_name in {"person", "truck"}
        assert 0.0 <= det.confidence <= 1.0
        x1, y1, x2, y2 = det.box_xyxy
        assert x2 > x1
        assert y2 > y1


def test_detect_filters_to_target_classes_only(detector: Detector):
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank_frame)
    assert all(det.class_name in {"person", "truck"} for det in result.detections)


def test_detector_with_no_target_classes_allows_all():
    detector_all_classes = Detector(model_name="yolov8n.pt", target_classes=None, confidence_threshold=0.9)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # High confidence threshold on a blank frame -> should safely return zero
    # detections rather than crashing.
    result = detector_all_classes.detect(blank_frame)
    assert result.detections == []
