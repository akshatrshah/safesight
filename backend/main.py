"""SafeSight backend API.

Run with:
    uvicorn backend.main:create_app --factory --port 8000
Interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import Settings
from backend.db import Database
from backend.events import EventBus
from backend.worker import JobWorker, ProcessorFactory

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
TERMINAL_EVENTS = {"job_done", "job_failed"}
HEARTBEAT_SECONDS = 15


def create_app(settings: Settings | None = None, processor_factory: ProcessorFactory | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    bus = EventBus()
    worker = JobWorker(db, bus, settings, processor_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db.fail_interrupted_jobs()
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="SafeSight API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.db, app.state.bus, app.state.worker, app.state.settings = db, bus, worker, settings

    def require_job(job_id: str) -> dict:
        job = db.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"No job with id {job_id}")
        return job

    # ---- health / overview ----

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "queue_depth": worker.queue_depth()}

    @app.get("/api/stats")
    def stats() -> dict:
        return db.stats()

    # ---- jobs ----

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        file: UploadFile = File(...),
        frame_skip: int = Form(2, ge=1, le=30),
        max_frames: int | None = Form(150, ge=1),
        use_vehicle_model: bool = Form(False),
    ) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type '{suffix}', expected one of {sorted(ALLOWED_EXTENSIONS)}")
        if use_vehicle_model and not settings.vehicle_model:
            raise HTTPException(400, "use_vehicle_model requested but SAFESIGHT_VEHICLE_MODEL is not configured on the server")

        job_id = uuid.uuid4().hex[:12]
        dest = settings.uploads_dir / f"{job_id}{suffix}"
        limit, written = settings.max_upload_mb * 1024 * 1024, 0
        try:
            with open(dest, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB limit")
                    out.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise

        job = db.create_job(
            job_id, file.filename or dest.name,
            {"frame_skip": frame_skip, "max_frames": max_frames, "use_vehicle_model": use_vehicle_model},
        )
        worker.submit(job_id)
        return job

    @app.get("/api/jobs")
    def list_jobs(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        status: Literal["queued", "running", "done", "failed"] | None = None,
    ) -> list[dict]:
        return db.list_jobs(limit, offset, status)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return require_job(job_id)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        job = require_job(job_id)
        if job["status"] in ("queued", "running"):
            raise HTTPException(409, "Cannot delete a job that is queued or running")
        db.delete_job(job_id)
        for path in settings.uploads_dir.glob(f"{job_id}.*"):
            path.unlink(missing_ok=True)
        (settings.annotated_dir / f"{job_id}.mp4").unlink(missing_ok=True)

    @app.get("/api/jobs/{job_id}/timeline")
    def job_timeline(job_id: str) -> list[dict]:
        require_job(job_id)
        return db.get_timeline(job_id)

    @app.get("/api/jobs/{job_id}/events")
    def job_events(
        job_id: str,
        min_risk: Literal["LOW", "MEDIUM", "HIGH"] = "LOW",
        limit: int = Query(200, ge=1, le=2000),
        offset: int = Query(0, ge=0),
    ) -> list[dict]:
        require_job(job_id)
        return db.list_events(job_id, min_risk, limit, offset)

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str) -> FileResponse:
        job = require_job(job_id)
        path = settings.annotated_dir / f"{job_id}.mp4"
        if job["status"] != "done" or not path.exists():
            raise HTTPException(404, "Annotated video not available (job not finished, or it failed)")
        return FileResponse(path, media_type="video/mp4", filename=f"{Path(job['filename']).stem}_annotated.mp4")

    # ---- alerts across all jobs ----

    @app.get("/api/alerts")
    def alerts(
        min_risk: Literal["MEDIUM", "HIGH"] = "MEDIUM",
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> list[dict]:
        return db.list_events(None, min_risk, limit, offset)

    # ---- live event streams (Server-Sent Events) ----

    def sse(event: dict) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    async def stream(topic: str, stop_on_terminal: bool, initial_fn=lambda: []) -> AsyncIterator[str]:
        # Subscribe BEFORE reading current state, so a job finishing in between can't slip past both.
        sub = bus.subscribe(topic)
        try:
            for event in initial_fn():
                yield sse(event)
                if stop_on_terminal and event["type"] in TERMINAL_EVENTS:
                    return
            while True:
                try:
                    event = await asyncio.wait_for(sub.queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield sse(event)
                if stop_on_terminal and event["type"] in TERMINAL_EVENTS:
                    return
        finally:
            bus.unsubscribe(sub)

    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    @app.get("/api/jobs/{job_id}/stream")
    async def job_stream(job_id: str) -> StreamingResponse:
        """Live frame / alert events for one job. Ends after job_done or job_failed.
        Connecting to an already-finished job returns its terminal event immediately."""
        require_job(job_id)

        def initial() -> list[dict]:
            job = db.get_job(job_id) or {}
            if job.get("status") == "done":
                return [{"type": "job_done", "job_id": job_id, "summary": job["summary"]}]
            if job.get("status") == "failed":
                return [{"type": "job_failed", "job_id": job_id, "error": job["error"]}]
            return []

        return StreamingResponse(stream(job_id, True, initial), media_type="text/event-stream", headers=sse_headers)

    @app.get("/api/stream")
    async def global_stream() -> StreamingResponse:
        """Every event from every job, for a live wall-display style feed. Never ends on its own."""
        return StreamingResponse(stream(EventBus.ALL, False), media_type="text/event-stream", headers=sse_headers)

    return app

