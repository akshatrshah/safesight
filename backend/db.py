"""SQLite storage for analysis jobs, per-frame stats, and person-vehicle interaction events.

Stdlib sqlite3 only, no ORM, the schema is three small tables. One connection per
call keeps this safe to use from both the API threads and the worker thread;
WAL mode lets the dashboard read while the worker is writing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,              -- queued | running | done | failed
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    settings TEXT NOT NULL,            -- JSON
    frames_processed INTEGER NOT NULL DEFAULT 0,
    total_frames INTEGER NOT NULL DEFAULT 0,
    summary TEXT,                      -- JSON, set when done
    detection_breakdown TEXT,          -- JSON, set when done
    error TEXT
);
CREATE TABLE IF NOT EXISTS frame_stats (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    max_risk_level TEXT NOT NULL,
    object_counts TEXT NOT NULL,       -- JSON class -> count
    PRIMARY KEY (job_id, frame_number)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    vehicle_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    distance_px REAL NOT NULL,
    time_to_collision_s REAL,
    risk_level TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, frame_number);
CREATE INDEX IF NOT EXISTS idx_events_risk ON events(risk_level);
"""

_JSON_COLUMNS = ("settings", "summary", "detection_breakdown")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._init_lock = threading.Lock()
        with self._init_lock, self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:  # commit / rollback
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _job_row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        for col in _JSON_COLUMNS:
            job[col] = json.loads(job[col]) if job[col] else None
        return job

    # ---- jobs ----

    def create_job(self, job_id: str, filename: str, settings: dict) -> dict:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, filename, status, created_at, settings) VALUES (?, ?, 'queued', ?, ?)",
                (job_id, filename, utcnow(), json.dumps(settings)),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            return self._job_row(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def list_jobs(self, limit: int = 50, offset: int = 0, status: str | None = None) -> list[dict]:
        query, params = "SELECT * FROM jobs", []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._conn() as conn:
            return [self._job_row(r) for r in conn.execute(query, params).fetchall()]

    def mark_running(self, job_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (utcnow(), job_id))

    def update_progress(self, job_id: str, frames_processed: int, total_frames: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET frames_processed=?, total_frames=? WHERE id=?",
                (frames_processed, total_frames, job_id),
            )

    def mark_done(self, job_id: str, summary: dict, detection_breakdown: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='done', finished_at=?, summary=?, detection_breakdown=? WHERE id=?",
                (utcnow(), json.dumps(summary), json.dumps(detection_breakdown), job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?", (utcnow(), error, job_id))

    def fail_interrupted_jobs(self) -> int:
        """On startup, anything still queued/running belongs to a process that died, so it can never finish."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error='Server restarted before this job finished' "
                "WHERE status IN ('queued', 'running')",
                (utcnow(),),
            )
            return cur.rowcount

    def delete_job(self, job_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,)).rowcount > 0

    # ---- per-frame data ----

    def add_frame(self, job_id: str, frame_number: int, timestamp: float, max_risk: str, object_counts: dict, interactions: list[dict]) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO frame_stats VALUES (?, ?, ?, ?, ?)",
                (job_id, frame_number, timestamp, max_risk, json.dumps(object_counts)),
            )
            conn.executemany(
                "INSERT INTO events (job_id, frame_number, timestamp_seconds, vehicle_id, person_id, distance_px, time_to_collision_s, risk_level) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (job_id, frame_number, timestamp, i["vehicle_id"], i["person_id"], i["distance_px"], i["time_to_collision_s"], i["risk_level"])
                    for i in interactions
                ],
            )

    def get_timeline(self, job_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT frame_number, timestamp_seconds, max_risk_level, object_counts FROM frame_stats WHERE job_id=? ORDER BY frame_number",
                (job_id,),
            ).fetchall()
        return [{**dict(r), "object_counts": json.loads(r["object_counts"])} for r in rows]

    def list_events(self, job_id: str | None = None, min_risk: str = "LOW", limit: int = 200, offset: int = 0) -> list[dict]:
        levels = {"LOW": ("LOW", "MEDIUM", "HIGH"), "MEDIUM": ("MEDIUM", "HIGH"), "HIGH": ("HIGH",)}[min_risk]
        query = f"SELECT * FROM events WHERE risk_level IN ({','.join('?' * len(levels))})"
        params: list = list(levels)
        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def stats(self) -> dict:
        with self._conn() as conn:
            job_counts = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
            event_counts = {r["risk_level"]: r["n"] for r in conn.execute("SELECT risk_level, COUNT(*) n FROM events GROUP BY risk_level")}
        return {
            "jobs": {s: job_counts.get(s, 0) for s in ("queued", "running", "done", "failed")},
            "interactions": {r: event_counts.get(r, 0) for r in ("LOW", "MEDIUM", "HIGH")},
        }
