"""SafeSight dashboard: a client of the backend API, not a second copy of the pipeline.

Start the backend first, then:
    streamlit run dashboard/app.py
Point it at a non-default backend with SAFESIGHT_API_URL.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("SAFESIGHT_API_URL", "http://localhost:8000").rstrip("/")
RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]
RISK_TO_NUM = {r: i for i, r in enumerate(RISK_ORDER)}

st.set_page_config(page_title="SafeSight", layout="wide")


def api(method: str, path: str, **kwargs):
    resp = requests.request(method, f"{API}{path}", timeout=kwargs.pop("timeout", 15), **kwargs)
    resp.raise_for_status()
    return resp.json() if resp.content and resp.headers.get("content-type", "").startswith("application/json") else resp


def sse_events(path: str) -> Iterator[dict]:
    """Yield decoded events from a Server-Sent Events endpoint until the server closes it."""
    with requests.get(f"{API}{path}", stream=True, timeout=(5, 60)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[len("data: "):])


try:
    stats = api("GET", "/api/stats")
except requests.RequestException:
    st.title("SafeSight")
    st.error(f"Can't reach the backend at {API}. Start it with: `uvicorn backend.main:create_app --factory --port 8000`")
    st.stop()

st.title("SafeSight")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jobs done", stats["jobs"]["done"])
c2.metric("Running / queued", stats["jobs"]["running"] + stats["jobs"]["queued"])
c3.metric("Failed", stats["jobs"]["failed"])
c4.metric("HIGH risk interactions", stats["interactions"]["HIGH"])
c5.metric("MEDIUM risk interactions", stats["interactions"]["MEDIUM"])

tab_analyze, tab_jobs, tab_alerts = st.tabs(["Analyze a video", "Jobs", "Alerts"])


def show_job_results(job: dict) -> None:
    if job["status"] == "failed":
        st.error(f"Job failed: {job['error']}")
        return
    if job["status"] != "done":
        st.info(f"Job is {job['status']}. Results appear here once it finishes.")
        return

    summary = job["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames analyzed", summary["frames_processed"])
    m2.metric("Duration (s)", summary["duration_seconds"])
    m3.metric("HIGH risk frames", summary["risk_frame_counts"]["HIGH"])
    m4.metric("MEDIUM risk frames", summary["risk_frame_counts"]["MEDIUM"])

    video = requests.get(f"{API}/api/jobs/{job['id']}/video", timeout=60)
    if video.ok:
        st.subheader("Annotated video")
        st.video(video.content)

    timeline = api("GET", f"/api/jobs/{job['id']}/timeline")
    if timeline:
        st.subheader("Risk over time")
        risk_df = pd.DataFrame({"timestamp_seconds": [t["timestamp_seconds"] for t in timeline],
                                "risk_level": [RISK_TO_NUM[t["max_risk_level"]] for t in timeline]})
        st.line_chart(risk_df.set_index("timestamp_seconds"))
        st.caption("0 = LOW, 1 = MEDIUM, 2 = HIGH")

        st.subheader("Object count over time")
        counts_df = pd.DataFrame([{"timestamp_seconds": t["timestamp_seconds"], **t["object_counts"]} for t in timeline])
        st.line_chart(counts_df.fillna(0).set_index("timestamp_seconds"))

    st.subheader("Detection breakdown, per class")
    if job["detection_breakdown"]:
        st.dataframe(pd.DataFrame([{"class": k, **v} for k, v in job["detection_breakdown"].items()]), hide_index=True)
    else:
        st.write("No objects detected in this video, check the model and confidence threshold.")

    st.subheader("Person-vehicle interactions")
    min_risk = st.radio("Show", RISK_ORDER, index=1, horizontal=True, key=f"minrisk-{job['id']}",
                        format_func=lambda r: f"{r} and above")
    events = api("GET", f"/api/jobs/{job['id']}/events", params={"min_risk": min_risk, "limit": 2000})
    if events:
        st.dataframe(pd.DataFrame(events).drop(columns=["id", "job_id"]), hide_index=True)
    else:
        st.write("None at this level (either no vehicle model was loaded, or no vehicle came near a person).")


def follow_live(job_id: str) -> dict:
    """Show real progress and alerts streamed from the backend while a job runs; returns the final job record."""
    bar, status, alert_box = st.progress(0.0), st.empty(), st.container()
    alerts_seen = 0
    try:
        for event in sse_events(f"/api/jobs/{job_id}/stream"):
            if event["type"] == "frame":
                total = max(event["total_frames"], 1)
                bar.progress(min(event["frames_processed"] / total, 1.0))
                status.write(f"t = {event['timestamp_seconds']}s, risk **{event['max_risk_level']}**, objects {event['object_counts']}")
            elif event["type"] == "alert":
                alerts_seen += 1
                alert_box.warning(f"{event['risk_level']} at {event['timestamp_seconds']}s: vehicle #{event['vehicle_id']} and person #{event['person_id']}, "
                                  f"{event['distance_px']}px apart, time to collision {event['time_to_collision_s']}")
    except requests.RequestException as exc:
        st.warning(f"Live stream interrupted ({exc}), showing whatever the backend has stored.")
    bar.empty()
    status.empty()
    return api("GET", f"/api/jobs/{job_id}")


with tab_analyze:
    st.caption("Uploads go to the backend, which queues the job, runs the real pipeline, stores everything in its database, and streams progress back live.")
    with st.form("analyze"):
        upload = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        f1, f2, f3 = st.columns(3)
        frame_skip = f1.slider("Process every Nth frame", 1, 10, 2, help="Higher = faster but coarser.")
        max_frames = f2.number_input("Max frames (0 = no limit)", min_value=0, value=150, step=10)
        use_vehicle = f3.checkbox("Use fine-tuned forklift/pallet model", value=False,
                                  help="Needs SAFESIGHT_VEHICLE_MODEL set on the backend.")
        submitted = st.form_submit_button("Analyze")

    if submitted:
        if upload is None:
            st.warning("Choose a video first.")
        else:
            try:
                job = api("POST", "/api/jobs", files={"file": (upload.name, upload.getvalue())}, timeout=300,
                          data={"frame_skip": frame_skip, "use_vehicle_model": str(use_vehicle).lower(),
                                **({"max_frames": max_frames} if max_frames > 0 else {})})
            except requests.HTTPError as exc:
                st.error(exc.response.json().get("detail", str(exc)))
            else:
                st.session_state["analyze_job_id"] = job["id"]
                st.session_state["analyze_followed"] = False

    job_id = st.session_state.get("analyze_job_id")
    if job_id:
        if not st.session_state.get("analyze_followed"):
            st.session_state["analyze_followed"] = True
            follow_live(job_id)
        show_job_results(api("GET", f"/api/jobs/{job_id}"))

with tab_jobs:
    if st.button("Refresh", key="refresh-jobs"):
        st.rerun()
    jobs = api("GET", "/api/jobs", params={"limit": 100})
    if not jobs:
        st.write("No jobs yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "id": j["id"], "file": j["filename"], "status": j["status"], "created": j["created_at"],
            "frames": f"{j['frames_processed']}/{j['total_frames']}",
            "HIGH frames": (j["summary"] or {}).get("risk_frame_counts", {}).get("HIGH"),
        } for j in jobs]), hide_index=True)

        chosen = st.selectbox("Open a job", [j["id"] for j in jobs],
                              format_func=lambda i: next(f"{j['filename']} ({j['status']}, {i})" for j in jobs if j["id"] == i))
        show_job_results(next(j for j in jobs if j["id"] == chosen))
        if st.button("Delete this job", key="delete-job"):
            try:
                api("DELETE", f"/api/jobs/{chosen}")
                st.rerun()
            except requests.HTTPError as exc:
                st.error(exc.response.json().get("detail", str(exc)))

with tab_alerts:
    st.caption("Every MEDIUM or HIGH person-vehicle interaction the backend has stored, across all jobs, newest first.")
    level = st.radio("Minimum level", ["MEDIUM", "HIGH"], horizontal=True, key="alert-level")
    alerts = api("GET", "/api/alerts", params={"min_risk": level, "limit": 500})
    if alerts:
        st.dataframe(pd.DataFrame(alerts).drop(columns=["id"]), hide_index=True)
    else:
        st.write("No alerts recorded yet.")
