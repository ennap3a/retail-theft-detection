"""Step 3: run the detector with a live progress bar."""

from __future__ import annotations

import streamlit as st

from retail_theft_detector.detector import DetectorSettings, analyze_project
from retail_theft_detector.project import load_project, paths_for_project
from retail_theft_detector.video import format_duration
from retail_theft_detector.zones import load_zones, zone_counts

from . import state


def detector_settings() -> DetectorSettings:
    """Detector settings chosen in the sidebar (falls back to defaults)."""
    defaults = DetectorSettings()
    return DetectorSettings(
        object_confidence=st.session_state.get("cfg_confidence", defaults.object_confidence),
        pose_confidence=st.session_state.get("cfg_confidence", defaults.pose_confidence),
        max_analysis_fps=float(st.session_state.get("cfg_fps", defaults.max_analysis_fps)),
        evidence_seconds_before=float(st.session_state.get("cfg_evidence", defaults.evidence_seconds_before)),
        evidence_seconds_after=float(st.session_state.get("cfg_evidence", defaults.evidence_seconds_after)),
    )


def render() -> None:
    project_id = st.session_state.project_id
    project = load_project(project_id)
    zones = load_zones(paths_for_project(project_id).zones_path)
    counts = zone_counts(zones)
    info = project["video"]["info"]
    settings = detector_settings()

    st.header("Run analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Video length", format_duration(info["duration_seconds"]))
    c2.metric("Product zones", counts["product"])
    c3.metric("Customer zone", counts["customer"])
    c4.metric("Analysis rate", f"{settings.max_analysis_fps:g} fps")

    st.caption(
        "The detector tracks every person, estimates their pose, watches the product zones for "
        "changes and applies the behaviour rules frame by frame. On a laptop CPU expect roughly "
        "5 analysed frames per second; adjust the rate in the sidebar."
    )

    back, _, run = st.columns([1, 3, 1.4])
    if back.button("← Edit zones"):
        state.go(1)
        st.rerun()
    if not run.button("Run analysis", type="primary"):
        return

    bar = st.progress(0.0, text="Starting…")

    def report(fraction: float, message: str) -> None:
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    try:
        analyze_project(project_id, settings, progress=report)
    except Exception as exc:  # surface any model / video error to the user
        bar.empty()
        st.error(f"Analysis failed: {exc}")
        return

    st.session_state.package_path = None
    st.session_state.feedback_result = None
    state.go(3)
    st.rerun()
