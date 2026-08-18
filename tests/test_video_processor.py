import cv2
import numpy as np
import pytest

from perception.video.video_processor import VideoProcessor, VideoAnalysisResult, FrameResult


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """A small real video file with genuine simulated motion, built once and reused across tests."""
    tmp_dir = tmp_path_factory.mktemp("video_test")
    video_path = tmp_dir / "synthetic.mp4"

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 200), (200, 200, 200), -1)  # a simple blob, not a real person

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for _ in range(10):
        writer.write(frame)
    writer.release()

    return str(video_path)


def test_process_video_reads_correct_frame_count(synthetic_video):
    processor = VideoProcessor(person_model="yolov8n.pt", vehicle_model=None, frame_skip=1)
    result = processor.process_video(synthetic_video)

    assert isinstance(result, VideoAnalysisResult)
    assert result.total_frames_in_video == 10
    assert result.frames_processed == 10


def test_frame_skip_reduces_processed_count(synthetic_video):
    processor = VideoProcessor(person_model="yolov8n.pt", vehicle_model=None, frame_skip=3)
    result = processor.process_video(synthetic_video)

    # frames 0, 3, 6, 9 -> 4 processed out of 10
    assert result.frames_processed == 4


def test_max_frames_stops_early(synthetic_video):
    processor = VideoProcessor(person_model="yolov8n.pt", vehicle_model=None, frame_skip=1)
    result = processor.process_video(synthetic_video, max_frames=3)

    assert result.frames_processed == 3


def test_summary_has_expected_shape(synthetic_video):
    processor = VideoProcessor(person_model="yolov8n.pt", vehicle_model=None, frame_skip=1)
    result = processor.process_video(synthetic_video)
    summary = result.summary()

    assert "risk_frame_counts" in summary
    assert set(summary["risk_frame_counts"].keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert summary["frames_processed"] == 10


def test_missing_video_raises_clear_error():
    processor = VideoProcessor(person_model="yolov8n.pt", vehicle_model=None)
    with pytest.raises(FileNotFoundError):
        processor.process_video("/tmp/this_video_does_not_exist_12345.mp4")


def test_annotated_output_is_a_real_readable_video(synthetic_video, tmp_path):
    out_path = tmp_path / "annotated.mp4"
    processor = VideoProcessor(person_model="yolov8n.pt", vehicle_model=None, frame_skip=1)
    processor.process_video(synthetic_video, annotated_output_path=str(out_path))

    assert out_path.exists()
    cap = cv2.VideoCapture(str(out_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    success, frame = cap.read()
    cap.release()
    assert success
    assert frame.shape == (240, 320, 3)
