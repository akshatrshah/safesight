import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.db import Database
from backend.events import EventBus
from backend.main import create_app
from perception.video.video_processor import FrameResult, VideoAnalysisResult


class FakeProcessor:
    """Scripted stand-in for VideoProcessor, so backend tests don't load any YOLO model."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def process_video(self, source_path, annotated_output_path=None, max_frames=None, on_frame=None):
        if self.fail:
            raise RuntimeError("model exploded")
        result = VideoAnalysisResult(source_path=str(source_path), fps=10.0, total_frames_in_video=3, frames_processed=0)
        script = [
            ("LOW", [{"vehicle_id": 100001, "person_id": 1, "distance_px": 400.0, "time_to_collision_s": None, "risk_level": "LOW"}]),
            ("MEDIUM", [{"vehicle_id": 100001, "person_id": 1, "distance_px": 150.0, "time_to_collision_s": 4.2, "risk_level": "MEDIUM"}]),
            ("HIGH", [{"vehicle_id": 100001, "person_id": 1, "distance_px": 40.0, "time_to_collision_s": 0.8, "risk_level": "HIGH"}]),
        ]
        for i, (risk, interactions) in enumerate(script):
            fr = FrameResult(
                frame_number=i, timestamp_seconds=i / 10.0, max_risk_level=risk, interactions=interactions,
                tracked_objects=[
                    {"track_id": 1, "class_name": "person", "confidence": 0.9, "center": (10, 10), "zones": []},
                    {"track_id": 100001, "class_name": "forklift", "confidence": 0.8, "center": (50, 10), "zones": []},
                ],
            )
            result.frame_results.append(fr)
            result.frames_processed += 1
            if on_frame:
                on_frame(fr, 3)
        if annotated_output_path:
            open(annotated_output_path, "wb").write(b"fake-mp4-bytes")
        return result


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, vehicle_model="fake_forklift.pt")


@pytest.fixture
def client(settings):
    app = create_app(settings, processor_factory=lambda job_settings: FakeProcessor())
    with TestClient(app) as c:
        yield c


def upload(client, name="clip.mp4", **fields):
    return client.post("/api/jobs", files={"file": (name, b"not-really-a-video", "video/mp4")}, data=fields)


def wait_for(client, job_id, status, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {status}, last: {job}")


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_job_lifecycle_and_stored_results(client):
    resp = upload(client, frame_skip="3", max_frames="50")
    assert resp.status_code == 202
    job = resp.json()
    assert job["settings"] == {"frame_skip": 3, "max_frames": 50, "use_vehicle_model": False}

    done = wait_for(client, job["id"], "done")
    assert done["frames_processed"] == 3
    assert done["summary"]["risk_frame_counts"] == {"LOW": 1, "MEDIUM": 1, "HIGH": 1}
    assert done["detection_breakdown"]["forklift"]["unique_objects_tracked"] == 1

    timeline = client.get(f"/api/jobs/{job['id']}/timeline").json()
    assert [t["max_risk_level"] for t in timeline] == ["LOW", "MEDIUM", "HIGH"]
    assert timeline[0]["object_counts"] == {"person": 1, "forklift": 1}

    assert len(client.get(f"/api/jobs/{job['id']}/events").json()) == 3
    high_only = client.get(f"/api/jobs/{job['id']}/events", params={"min_risk": "HIGH"}).json()
    assert [e["risk_level"] for e in high_only] == ["HIGH"]

    assert client.get(f"/api/jobs/{job['id']}/video").content == b"fake-mp4-bytes"


def test_alerts_span_jobs_and_exclude_low(client):
    ids = [upload(client).json()["id"] for _ in range(2)]
    for job_id in ids:
        wait_for(client, job_id, "done")

    alerts = client.get("/api/alerts").json()
    assert len(alerts) == 4  # MEDIUM + HIGH from each of two jobs
    assert {a["job_id"] for a in alerts} == set(ids)
    assert all(a["risk_level"] in ("MEDIUM", "HIGH") for a in alerts)
    assert client.get("/api/alerts", params={"min_risk": "HIGH"}).json().__len__() == 2

    stats = client.get("/api/stats").json()
    assert stats["jobs"]["done"] == 2
    assert stats["interactions"] == {"LOW": 2, "MEDIUM": 2, "HIGH": 2}


def test_failed_job_records_error(settings):
    app = create_app(settings, processor_factory=lambda s: FakeProcessor(fail=True))
    with TestClient(app) as client:
        job_id = upload(client).json()["id"]
        failed = wait_for(client, job_id, "failed")
        assert "model exploded" in failed["error"]
        assert client.get(f"/api/jobs/{job_id}/video").status_code == 404


def test_rejects_bad_uploads(client, settings):
    assert upload(client, name="notes.txt").status_code == 415
    assert client.post("/api/jobs", files={"file": ("a.mp4", b"x")}, data={"frame_skip": "0"}).status_code == 422
    settings.vehicle_model = None
    assert upload(client, use_vehicle_model="true").status_code == 400
    assert client.get("/api/jobs").json() == []  # nothing was created by any rejected upload
    assert list(settings.uploads_dir.iterdir()) == []


def test_upload_size_limit_cleans_up(settings):
    settings.max_upload_mb = 0
    app = create_app(settings, processor_factory=lambda s: FakeProcessor())
    with TestClient(app) as client:
        assert upload(client).status_code == 413
        assert list(settings.uploads_dir.iterdir()) == []


def test_unknown_job_404s(client):
    for path in ("", "/timeline", "/events", "/video", "/stream"):
        assert client.get(f"/api/jobs/nope{path}").status_code == 404


def test_delete_job_removes_data_and_files(client, settings):
    job_id = upload(client).json()["id"]
    wait_for(client, job_id, "done")
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get("/api/alerts").json() == []  # cascade removed the events
    assert list(settings.uploads_dir.iterdir()) == [] and list(settings.annotated_dir.iterdir()) == []


def test_list_jobs_filter_and_order(client):
    first = upload(client, name="a.mp4").json()["id"]
    wait_for(client, first, "done")
    second = upload(client, name="b.mp4").json()["id"]
    wait_for(client, second, "done")
    assert [j["id"] for j in client.get("/api/jobs").json()] == [second, first]
    assert client.get("/api/jobs", params={"status": "failed"}).json() == []
    assert len(client.get("/api/jobs", params={"limit": 1}).json()) == 1


def test_stream_of_finished_job_returns_terminal_event_and_closes(client):
    job_id = upload(client).json()["id"]
    wait_for(client, job_id, "done")
    with client.stream("GET", f"/api/jobs/{job_id}/stream") as resp:
        body = "".join(resp.iter_text())
    assert "event: job_done" in body
    data = json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])
    assert data["summary"]["frames_processed"] == 3


def test_restart_fails_interrupted_jobs(settings):
    db = Database(settings.db_path)
    db.create_job("stuck", "a.mp4", {})
    db.mark_running("stuck")
    with TestClient(create_app(settings, processor_factory=lambda s: FakeProcessor())) as client:
        job = client.get("/api/jobs/stuck").json()
    assert job["status"] == "failed" and "restarted" in job["error"]


def test_event_bus_delivers_from_other_thread_to_job_and_global_subscribers():
    async def scenario():
        import threading

        bus = EventBus()
        job_sub, all_sub, other_sub = bus.subscribe("job1"), bus.subscribe(EventBus.ALL), bus.subscribe("job2")
        threading.Thread(target=bus.publish, args=("job1", {"type": "frame", "n": 1})).start()
        assert (await asyncio.wait_for(job_sub.queue.get(), 2))["n"] == 1
        assert (await asyncio.wait_for(all_sub.queue.get(), 2))["n"] == 1
        assert other_sub.queue.empty()

        bus.unsubscribe(job_sub)
        bus.publish("job1", {"type": "frame", "n": 2})
        await asyncio.sleep(0.05)
        assert job_sub.queue.empty()

    asyncio.run(scenario())


def test_event_bus_slow_subscriber_drops_oldest_instead_of_blocking():
    async def scenario():
        from backend import events

        bus = EventBus()
        sub = bus.subscribe("j")
        for n in range(events.QUEUE_SIZE + 10):
            bus.publish("j", {"type": "frame", "n": n})
        await asyncio.sleep(0.1)
        assert sub.queue.qsize() == events.QUEUE_SIZE
        assert (await sub.queue.get())["n"] == 10  # the 10 oldest were dropped

    asyncio.run(scenario())
