"""Streamlit entry point: ``streamlit run streamlit_app.py``."""

from __future__ import annotations

import json
import os

import streamlit as st

from retail_theft_detector.config import BEHAVIOUR_CARD_PATH as CARD_PATH
from retail_theft_detector.config import SKELETON_WEIGHTS_PATH as WEIGHTS_PATH
from retail_theft_detector.config import retention_hours
from retail_theft_detector.project import paths_for_project
from retail_theft_detector.review import cleanup_old_projects
from ui import page_analyse, page_review, page_upload, page_zones, state, theme

REPO_URL = "https://github.com/ennap3a/retail-theft-detection"

st.set_page_config(
    page_title="Retail Theft Review Assistant",
    page_icon="🛡️",
    layout="wide",
    menu_items={"About": f"Human-in-the-loop retail security review. Source: {REPO_URL}"},
)


SECRET_SETTINGS = ("MAX_VIDEO_SECONDS", "RETENTION_HOURS")


def _secrets_to_env() -> None:
    """Expose hosting settings stored as Streamlit secrets to the config module."""
    try:
        for name in SECRET_SETTINGS:
            if name in st.secrets and name not in os.environ:
                os.environ[name] = str(st.secrets[name])
    except Exception:  # no secrets file: run with defaults
        pass


@st.cache_resource(ttl=3600)
def _housekeeping() -> int:
    """Delete expired uploads at most once an hour (only when RETENTION_HOURS is set)."""
    return cleanup_old_projects(retention_hours())


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🛡️ Theft Review")
        st.caption("YOLO11 · pose · behaviour model · zone logic")

        st.markdown("#### Detector settings")
        locked = st.session_state.step == 3
        st.slider(
            "Detection confidence",
            0.10,
            0.60,
            0.20,
            0.05,
            key="cfg_confidence",
            disabled=locked,
            help="Lower finds more people and objects but adds noise.",
        )
        st.slider(
            "Analysis frames per second",
            2,
            15,
            8,
            key="cfg_fps",
            disabled=locked,
            help="Higher is more precise but slower.",
        )
        st.slider(
            "Evidence clip padding (s)",
            1.0,
            5.0,
            2.0,
            0.5,
            key="cfg_evidence",
            disabled=locked,
            help="Seconds of footage kept before and after each alert.",
        )
        _behaviour_settings(locked)

        st.divider()
        if st.session_state.project_id and st.button("Start over and delete my data"):
            state.start_over()
            st.rerun()

        st.markdown("#### Responsible use")
        st.caption(
            "Alerts are *suspected* events for trained staff to review. They are never proof of theft "
            "and must not be used for automated decisions about people."
        )
        st.markdown(f"[Source code on GitHub]({REPO_URL})")


@st.cache_data(show_spinner=False)
def _behaviour_card() -> dict | None:
    if not CARD_PATH.exists() or not WEIGHTS_PATH.exists():
        return None
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


def _behaviour_settings(locked: bool) -> None:
    card = _behaviour_card()
    if card is None:
        return
    st.markdown("#### Behaviour model")
    enabled = st.toggle(
        "Score body movement",
        value=True,
        key="cfg_behaviour",
        disabled=locked,
        help="A second detector that learned what concealment movements look like from 365 labelled clips "
        "(skeleton network + movement features). "
        "It does not need a shelf change or a visible item, so it catches theft the zone rules miss.",
    )
    default = float(card.get("threshold", 0.5))
    st.slider(
        "Alert threshold",
        0.05,
        0.95,
        round(default, 2),
        0.01,
        key="cfg_behaviour_threshold",
        disabled=locked or not enabled,
        help=f"Lower catches more theft but raises more false alarms. The default ({default:.2f}) was chosen "
        "during training to catch most suspicious clips.",
    )


def main() -> None:
    _secrets_to_env()
    state.init()
    theme.apply()
    _housekeeping()

    # The project may have expired or been deleted from another tab.
    project_id = st.session_state.project_id
    if project_id and not paths_for_project(project_id).metadata_path.exists():
        state.start_over(delete_data=False)
        st.info("Your previous session expired. Please upload the video again.")

    _sidebar()
    theme.step_bar(st.session_state.step)
    pages = (page_upload, page_zones, page_analyse, page_review)
    pages[st.session_state.step].render()


main()
