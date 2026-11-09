"""Session state helpers: one project per browser session."""

from __future__ import annotations

import streamlit as st

from retail_theft_detector.review import delete_project

STEPS = ("Upload video", "Draw zones", "Run analysis", "Review results")

_DEFAULTS = {
    "step": 0,
    "project_id": None,
    "zones": [],
    "draft_points": [],
    "last_click": None,
    "package_path": None,
    "feedback_result": None,
}


def init() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def go(step: int) -> None:
    st.session_state.step = step


def start_over(delete_data: bool = True) -> None:
    project_id = st.session_state.get("project_id")
    if delete_data and project_id:
        delete_project(project_id)
    for key, value in _DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value
