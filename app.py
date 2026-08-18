"""Dashboard: upload a video, run it through the full pipeline, see the results.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st
import yaml

from perception.video.video_processor import VideoProcessor

st.set_page_config(page_title="SafeSight", layout="wide")
st.title("SafeSight: Video Analysis")
st.caption("Upload a video, this runs my real detection, tracking, pose, zone, and risk pipeline over every frame.")

with open(REPO_ROOT / "configs" / "detection.yaml") as f:
    config = yaml.safe_load(f)

with st.sidebar:
    st.header("Settings")
    use_vehicle_model = st.checkbox("Use fine-tuned forklift/pallet model", value=False)
    vehicle_model_path = None
    if use_vehicle_model:
        vehicle_model_path = st.text_input("Path to fine-tuned weights", value=config.get("model_name", ""))
    frame_skip = st.slider("Process every Nth frame", min_value=1, max_value=10, value=2,
                            help="Higher = faster but coarser. 1 = every single frame.")
    max_frames = st.number_input("Max frames to process (0 = no limit)", min_value=0, value=150, step=10,
                                  help="Caps processing time for a quick look. Set to 0 for the full video.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    output_path = str(Path(tempfile.gettempdir()) / "safesight_annotated.mp4")

    if st.button("Analyze"):
        with st.spinner("Processing video, this runs the real pipeline frame by frame..."):
            processor = VideoProcessor(
                person_model="yolov8n.pt",
                vehicle_model=vehicle_model_path if use_vehicle_model else None,
                confidence_threshold=config["confidence_threshold"],
                frame_skip=frame_skip,
            )
            result = processor.process_video(
                input_path,
                annotated_output_path=output_path,
                max_frames=max_frames if max_frames > 0 else None,
            )

        st.success(f"Processed {result.frames_processed} frames")

        summary = result.summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frames analyzed", summary["frames_processed"])
        col2.metric("Duration (s)", summary["duration_seconds"])
        col3.metric("High risk frames", summary["risk_frame_counts"]["HIGH"])
        col4.metric("Medium risk frames", summary["risk_frame_counts"]["MEDIUM"])

        st.subheader("Annotated video")
        st.video(output_path)

        st.subheader("Detection breakdown, per class")
        st.caption("This is the debugging view. Unique objects tracked should stay stable and low if the same real object is being tracked correctly across frames. A high unique count relative to frame appearances usually means the tracker is losing and re-acquiring the same object repeatedly, worth investigating.")
        breakdown = result.detection_breakdown()
        if breakdown:
            breakdown_rows = [
                {"class": class_name, **stats}
                for class_name, stats in breakdown.items()
            ]
            st.dataframe(pd.DataFrame(breakdown_rows))
        else:
            st.write("No objects detected in this video at all, check the model and confidence threshold.")

        st.subheader("Object count over time")
        st.caption("Per-frame counts for each detected class. A steady, non-jumpy line is a good sign, sudden spikes or drops can mean flickering detections.")
        count_rows = result.per_frame_object_counts()
        if count_rows:
            counts_df = pd.DataFrame(count_rows).fillna(0).set_index("timestamp_seconds")
            st.line_chart(counts_df)

        st.subheader("Risk over time")
        timeline = result.risk_timeline()
        risk_to_num = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        df = pd.DataFrame(
            {"timestamp_seconds": [t for t, _ in timeline], "risk_level": [risk_to_num[r] for _, r in timeline]}
        )
        st.line_chart(df.set_index("timestamp_seconds"))
        st.caption("0 = LOW, 1 = MEDIUM, 2 = HIGH")

        st.subheader("Detected interactions")
        all_interactions = []
        for fr in result.frame_results:
            for interaction in fr.interactions:
                all_interactions.append({"timestamp_s": round(fr.timestamp_seconds, 1), **interaction})
        if all_interactions:
            st.dataframe(pd.DataFrame(all_interactions))
        else:
            st.write("No person-vehicle interactions detected in this video (either no vehicle model loaded, or none appeared close to a person).")

        st.subheader("High risk moments")
        if summary["high_risk_moments"]:
            st.write(f"Flagged at: {summary['high_risk_moments']} seconds")
        else:
            st.write("None detected in this video.")

else:
    st.info("Upload a video above to get started. For testing, use a video that was NOT part of my LOCO fine-tuning data, to get an honest read on generalization rather than a memorized result.")
