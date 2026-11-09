"""Step 2: choose a setup frame and draw product / customer zones."""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from retail_theft_detector.project import load_project, paths_for_project, update_project
from retail_theft_detector.video import extract_frame
from retail_theft_detector.zones import load_zones, save_zones, validate_zone_document

from . import state
from .drawing import finish_zone, render_zones, zone_document

MODES = {"Product zone": "product", "Customer zone": "customer"}


def _load_saved_zones(project_id: str) -> None:
    """Restore zones from disk the first time this step is opened for a project."""
    if st.session_state.zones:
        return
    saved = load_zones(paths_for_project(project_id).zones_path)
    if saved:
        st.session_state.zones = saved.get("zones", [])


def render() -> None:
    project_id = st.session_state.project_id
    project = load_project(project_id)
    paths = paths_for_project(project_id)
    info = project["video"]["info"]
    _load_saved_zones(project_id)

    st.header("Draw zones")
    st.caption(
        f"{project['video']['original_filename']} · {info['width']}×{info['height']} · "
        f"{info['duration_seconds']:.1f} s"
    )

    frame_bgr = cv2.imread(str(paths.first_frame_path))
    if frame_bgr is None:
        st.error("The setup frame could not be loaded. Start over and upload the video again.")
        return
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = frame_rgb.shape[:2]

    left, right = st.columns([3, 1.25], gap="large")

    with right:
        _setup_frame_controls(project, paths)
        st.divider()
        mode_label = st.radio("Drawing", list(MODES), horizontal=True, key="zone_mode")
        zone_type = MODES[mode_label]
        draft = st.session_state.draft_points
        st.caption(f"{len(draft)} point(s) in the current outline. Click the image to add points.")

        c1, c2, c3 = st.columns(3)
        if c1.button("Finish", type="primary", disabled=len(draft) < 3, help="Close the outline (3+ points)"):
            st.session_state.zones = finish_zone(st.session_state.zones, draft, zone_type)
            st.session_state.draft_points = []
            st.rerun()
        if c2.button("Undo", disabled=not draft, help="Remove the last point"):
            st.session_state.draft_points = draft[:-1]
            st.rerun()
        if c3.button("Clear", disabled=not draft, help="Discard the current outline"):
            st.session_state.draft_points = []
            st.rerun()

        st.divider()
        _zone_list()

    with left:
        image = render_zones(frame_rgb, st.session_state.zones, st.session_state.draft_points)
        click = streamlit_image_coordinates(image, width="stretch", key="zone_canvas", cursor="crosshair")
        if click and click.get("unix_time") != st.session_state.last_click and click.get("width"):
            st.session_state.last_click = click["unix_time"]
            scale_x = frame_w / click["width"]
            scale_y = frame_h / click["height"]
            point = {
                "x": round(min(max(click["x"] * scale_x, 0), frame_w), 1),
                "y": round(min(max(click["y"] * scale_y, 0), frame_h), 1),
            }
            st.session_state.draft_points = [*st.session_state.draft_points, point]
            st.rerun()

        with st.expander("How to draw good zones", expanded=not st.session_state.zones):
            st.markdown(
                "- **Product zone** — trace tightly around shelves, racks or tables where items are taken from. "
                "You can add several.\n"
                "- **Customer zone** — the browsing area where a shopper normally stands. "
                "Leave exits, tills and walkways *outside* it: leaving this zone is part of the alert rule.\n"
                "- Pick a setup frame where the shelves are fully visible and nobody blocks them."
            )

    doc = zone_document(
        project_id, frame_w, frame_h, project.get("setup_frame_seconds", 0.0), st.session_state.zones
    )
    errors = validate_zone_document(doc)

    st.divider()
    back, _, forward = st.columns([1, 3, 1.4])
    if back.button("← Different video", help="Discard this video and upload another"):
        state.start_over()
        st.rerun()
    if errors and st.session_state.zones:
        for error in errors:
            st.warning(error)
    if forward.button("Save zones and continue →", type="primary", disabled=bool(errors)):
        save_zones(paths.zones_path, doc)
        update_project(project_id, {"zones_saved": True, "stage": "ready_for_detector"})
        state.go(2)
        st.rerun()


def _setup_frame_controls(project: dict, paths) -> None:
    duration = float(project["video"]["info"]["duration_seconds"] or 0)
    current = float(project.get("setup_frame_seconds", 0.0))
    if duration <= 0.2:
        return
    seconds = st.slider(
        "Setup frame (seconds)",
        min_value=0.0,
        max_value=max(0.0, round(duration - 0.1, 1)),
        value=min(current, duration - 0.1),
        step=0.1,
        help="Choose a moment where the shelves are clearly visible. "
        "The detector compares later frames against this one.",
    )
    if st.button("Use this frame", disabled=abs(seconds - current) < 0.05):
        extract_frame(Path(project["video"]["path"]), paths.first_frame_path, at_seconds=seconds)
        update_project(project["project_id"], {"setup_frame_seconds": round(seconds, 3)})
        st.rerun()


def _zone_list() -> None:
    zones = st.session_state.zones
    st.markdown("**Zones**")
    if not zones:
        st.caption("No zones yet. You need at least one product zone and one customer zone.")
        return
    for index, zone in enumerate(zones):
        name, delete = st.columns([3, 1])
        icon = "🟩" if zone["type"] == "product" else "🟦"
        name.write(f"{icon} {zone['label']} · {len(zone['points'])} pts")
        if delete.button("✕", key=f"delete_zone_{zone['id']}", help=f"Delete {zone['label']}"):
            st.session_state.zones = zones[:index] + zones[index + 1 :]
            st.rerun()
