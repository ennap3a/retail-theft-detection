"""Step 1: upload a CCTV video."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from retail_theft_detector.config import ALLOWED_VIDEO_EXTENSIONS, TMP_DIR, max_video_seconds
from retail_theft_detector.review import create_project, delete_project
from retail_theft_detector.video import format_duration

from . import state, theme


def render() -> None:
    st.markdown(
        '<div class="rtd-hero"><h1>Retail Theft Review Assistant</h1>'
        "<p>Turn store CCTV footage into reviewable evidence of suspected concealment events, "
        "with a person always making the final call.</p></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    features = [
        ("Track people", "YOLO11 detection with ByteTrack keeps a stable ID for every shopper."),
        ("Read body pose", "Wrist and torso keypoints show whether an item stays in the hand."),
        ("Score behaviour", "A model trained on 365 clips scores body movement; zone rules confirm a hidden item."),
        ("Collect evidence", "Annotated video, crops, clips and CSV reports for every alert."),
    ]
    for col, (title, body) in zip(cols, features, strict=True):
        with col:
            theme.card(title, body)

    st.write("")
    st.subheader("Upload a video")
    accepted = sorted(ext.lstrip(".") for ext in ALLOWED_VIDEO_EXTENSIONS)
    uploaded = st.file_uploader(
        "One camera, one clip. Short clips (10–60 s) give the quickest results.",
        type=accepted,
        accept_multiple_files=False,
    )
    theme.note(
        "Your video is processed on this server only for this session. "
        "Use <b>Start over</b> in the sidebar to delete it and all results."
    )

    if uploaded is None:
        return

    if st.button("Upload and continue", type="primary"):
        with st.spinner("Reading video…"):
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
                    tmp.write(chunk)
            try:
                project = create_project(Path(tmp.name), uploaded.name)
            except RuntimeError as exc:
                Path(tmp.name).unlink(missing_ok=True)
                st.error(f"This file could not be read as a video. {exc}")
                return

        duration = project["video"]["info"]["duration_seconds"]
        limit = max_video_seconds()
        if limit and duration > limit:
            delete_project(project["project_id"])
            st.error(
                f"This demo accepts videos up to {format_duration(limit)}; "
                f"yours is {format_duration(duration)}. Trim the clip and try again."
            )
            return

        st.session_state.project_id = project["project_id"]
        state.go(1)
        st.rerun()
