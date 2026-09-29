"""Background job runner: one worker thread pulling analysis jobs off a queue.

The pipeline is CPU/GPU bound and holds the models in memory, so jobs run one at a
time. A fresh processor is built per job on purpose, the tracker keeps track IDs and
velocity history, which must not leak from one video into the next.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from backend.config import Settings
from backend.db import Database
from backend.events import EventBus

log = logging.getLogger("safesight.worker")

# (job settings dict) -> object with .process_video(path, annotated_output_path=, max_frames=, on_frame=)
ProcessorFactory = Callable[[dict], Any]


def default_processor_factory(settings: Settings) -> ProcessorFactory:
    def build(job_settings: dict):
        from perception.video.video_processor import VideoProcessor  # heavy import (torch), only when a job actually runs

        return VideoProcessor(
            person_model=settings.person_model,
            vehicle_model=settings.vehicle_model if job_settings.get("use_vehicle_model") else None,
            confidence_threshold=settings.confidence_threshold,
            frame_skip=job_settings.get("frame_skip", 2),
        )

    return build


class JobWorker:
    def __init__(self, db: Database, bus: EventBus, settings: Settings, processor_factory: ProcessorFactory | None = None) -> None:
        self.db = db
        self.bus = bus
        self.settings = settings
        self.processor_factory = processor_factory or default_processor_factory(settings)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="safesight-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._process(job_id)
            except Exception:  # never let one bad job kill the worker thread
                log.exception("unhandled error in job %s", job_id)

    def _process(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is None:  # deleted while queued
            return

        video_path = self.settings.uploads_dir / f"{job_id}{Path(job['filename']).suffix or '.mp4'}"
        annotated_path = self.settings.annotated_dir / f"{job_id}.mp4"
        self.db.mark_running(job_id)
        self.bus.publish(job_id, {"type": "job_started", "job_id": job_id})

        state = {"processed": 0}

        def on_frame(frame_result, total_frames: int) -> None:
            counts: dict[str, int] = {}
            for obj in frame_result.tracked_objects:
                counts[obj["class_name"]] = counts.get(obj["class_name"], 0) + 1
            self.db.add_frame(
                job_id, frame_result.frame_number, frame_result.timestamp_seconds,
                frame_result.max_risk_level, counts, frame_result.interactions,
            )
            state["processed"] += 1
            self.db.update_progress(job_id, state["processed"], total_frames)
            self.bus.publish(job_id, {
                "type": "frame", "job_id": job_id,
                "frame_number": frame_result.frame_number,
                "timestamp_seconds": round(frame_result.timestamp_seconds, 2),
                "max_risk_level": frame_result.max_risk_level,
                "object_counts": counts,
                "frames_processed": state["processed"], "total_frames": total_frames,
            })
            for interaction in frame_result.interactions:
                if interaction["risk_level"] in ("MEDIUM", "HIGH"):
                    self.bus.publish(job_id, {
                        "type": "alert", "job_id": job_id,
                        "frame_number": frame_result.frame_number,
                        "timestamp_seconds": round(frame_result.timestamp_seconds, 2),
                        **interaction,
                    })

        try:
            processor = self.processor_factory(job["settings"])
            result = processor.process_video(
                video_path,
                annotated_output_path=annotated_path,
                max_frames=job["settings"].get("max_frames"),
                on_frame=on_frame,
            )
            summary = result.summary()
            self.db.mark_done(job_id, summary, result.detection_breakdown())
            self.bus.publish(job_id, {"type": "job_done", "job_id": job_id, "summary": summary})
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self.db.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
            self.bus.publish(job_id, {"type": "job_failed", "job_id": job_id, "error": str(exc)})
