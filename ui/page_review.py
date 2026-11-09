"""Step 4: review alerts, evidence and downloads, and record reviewer feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from retail_theft_detector.project import load_project, paths_for_project
from retail_theft_detector.review import append_evaluation_row, build_review_package, load_results
from retail_theft_detector.video import format_duration

from . import state, theme

REAL_CASES = {
    "normal": "Normal shopping / no theft",
    "suspicious": "Suspicious or shoplifting behaviour",
    "unsure": "Unsure / needs another review",
}
MODEL_REVIEWS = {
    "correct_no_alert": "Correct: no alert was needed",
    "correct_alert": "Correct: the alert matched the suspicious action",
    "false_alert": "False alert: normal behaviour was flagged",
    "missed_event": "Missed event: suspicious action was not caught",
    "wrong_timestamp": "Alert happened at the wrong time",
    "unsure": "Unsure / needs another review",
}
MISTAKE_REASONS = {
    "item_visible_in_hand": "Item was visible in hand",
    "item_returned_to_shelf": "Item was returned to shelf",
    "bag_false_alert": "Bag caused a false alert",
    "clothing_body_false_alert": "Clothing / body caused a false alert",
    "missed_small_product": "Missed a small product",
    "bad_zones": "Zones need improvement",
    "lighting_occlusion_angle": "Lighting, occlusion or camera angle",
    "tracking_issue": "Tracking ID issue",
    "other": "Other",
}
RESULT_TEXT = {
    "TP": "True positive — the model caught a real suspicious event.",
    "TN": "True negative — no alert on normal footage.",
    "FP": "False positive — the model alerted on normal behaviour.",
    "FN": "False negative — the model missed a suspicious event.",
    "REVIEW": "Marked for another review.",
}


def render() -> None:
    project_id = st.session_state.project_id
    project = load_project(project_id)
    base_dir = paths_for_project(project_id).base_dir
    summary, alerts = load_results(project_id)
    results = summary.get("results", {})
    video = summary.get("video", {})

    st.header("Review results")
    st.caption(project["video"]["original_filename"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suspected events", len(alerts))
    c2.metric("Video length", format_duration(video.get("source_duration_seconds", 0)))
    c3.metric("Frames analysed", video.get("processed_frames", 0))
    c4.metric("Processing time", f"{results.get('runtime_seconds', 0):.1f} s")

    if alerts:
        st.warning(
            f"{len(alerts)} suspected event(s) need human review. "
            "These are not accusations — check the evidence before taking any action."
        )
    else:
        st.success("No suspected concealment events were found in this video.")

    videos_tab, alerts_tab, feedback_tab, downloads_tab = st.tabs(
        ["Videos", f"Alerts ({len(alerts)})", "Reviewer feedback", "Downloads"]
    )
    with videos_tab:
        _videos(project, base_dir, results)
    with alerts_tab:
        _alerts(alerts, base_dir)
    with feedback_tab:
        _feedback(project_id, project)
    with downloads_tab:
        _downloads(project_id, project, base_dir, results, summary)

    st.divider()
    back, _, new = st.columns([1.6, 2.4, 1.4])
    if back.button("← Edit zones and re-run"):
        state.go(1)
        st.rerun()
    if new.button("Analyse another video", type="primary"):
        state.start_over()
        st.rerun()


def _videos(project: dict, base_dir: Path, results: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Original footage**")
        st.video(project["video"]["path"])
    with right:
        st.markdown("**Annotated by the model**")
        annotated = base_dir / results.get("annotated_video", "")
        if annotated.is_file():
            st.video(str(annotated))
        else:
            st.info("The annotated video is not available.")
    theme.note(
        "Box colours show each person's state: grey normal, yellow interacting, light blue holding an item, "
        "orange possible concealment, green item returned, red human review required."
    )


def _image(base_dir: Path, relative: str, caption: str) -> None:
    path = base_dir / relative if relative else None
    if path and path.is_file():
        st.image(str(path), caption=caption, width="stretch")


def _alerts(alerts: list[dict[str, Any]], base_dir: Path) -> None:
    if not alerts:
        st.info("Nothing to review for this video.")
        return
    for alert in alerts:
        with st.container(border=True):
            st.markdown(
                f"#### {alert['alert_id']} · person {alert['person_id']} · {alert['time_seconds']:.1f} s"
            )
            details = {
                "Suspicion score": alert.get("suspicion_score", "—"),
                "Product zone": alert.get("product_zone_label") or "—",
                "Container": alert.get("container_interaction") or "body / clothing",
                "Priority": alert.get("review_priority", "—"),
            }
            theme.chips(details)

            frame_col, crops_col, clip_col = st.columns([2, 1, 2])
            with frame_col:
                _image(base_dir, alert.get("full_frame_image", ""), "Frame at alert time")
            with crops_col:
                _image(base_dir, alert.get("suspected_item_crop_image", ""), "Shelf area that changed")
                _image(base_dir, alert.get("person_crop_image", ""), "Person")
            with clip_col:
                clip = base_dir / alert["evidence_clip"] if alert.get("evidence_clip") else None
                if clip and clip.is_file():
                    st.video(str(clip))
                    st.caption("Evidence clip around the alert")
            st.caption(alert.get("notes", ""))


def _feedback(project_id: str, project: dict) -> None:
    st.markdown(
        "Record whether the model was right. Each answer is added to the evaluation log and "
        "classified as TP, FP, FN or TN, which is how detector accuracy is measured over time."
    )
    with st.form("feedback", clear_on_submit=False):
        real_case = st.radio("What really happened in this video?", list(REAL_CASES), format_func=REAL_CASES.get)
        model_review = st.radio("How did the model do?", list(MODEL_REVIEWS), format_func=MODEL_REVIEWS.get)
        reasons = st.multiselect("If it was wrong, why?", list(MISTAKE_REASONS), format_func=MISTAKE_REASONS.get)
        bad_timestamp = st.text_input("Time of the missed or wrong event (optional)", placeholder="e.g. 00:07")
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Save feedback", type="primary")

    if submitted:
        row = append_evaluation_row(
            project_id,
            project,
            {
                "real_case": [real_case],
                "model_review": [model_review],
                "mistake_reason": reasons,
                "bad_timestamp": [bad_timestamp],
                "reviewer_notes": [notes],
            },
        )
        st.session_state.feedback_result = row["result_type"]

    if st.session_state.feedback_result:
        result = st.session_state.feedback_result
        st.success(f"Saved as **{result}**. {RESULT_TEXT.get(result, '')}")


def _downloads(project_id: str, project: dict, base_dir: Path, results: dict, summary: dict) -> None:
    stem = Path(project["video"]["original_filename"]).stem
    c1, c2, c3 = st.columns(3)
    alerts_csv = base_dir / results.get("alerts_csv", "")
    if alerts_csv.is_file():
        c1.download_button("Alerts (CSV)", alerts_csv.read_bytes(), f"{stem}_alerts.csv", "text/csv")
    c2.download_button(
        "Summary (JSON)",
        json.dumps(summary, indent=2).encode(),
        f"{stem}_summary.json",
        "application/json",
    )
    annotated = base_dir / results.get("annotated_video", "")
    if annotated.is_file():
        c3.download_button("Annotated video (MP4)", annotated.read_bytes(), f"{stem}_annotated.mp4", "video/mp4")

    st.divider()
    st.markdown("**Review package** — original and annotated video, reports, zone setup and all evidence in one ZIP.")
    package = st.session_state.package_path
    if package is None or not Path(package).is_file():
        if st.button("Prepare review package"):
            with st.spinner("Building ZIP…"):
                st.session_state.package_path = str(build_review_package(project_id, project))
            st.rerun()
    else:
        path = Path(package)
        st.download_button(
            f"Download {path.name} ({path.stat().st_size / 1e6:.1f} MB)",
            path.read_bytes(),
            path.name,
            "application/zip",
            type="primary",
        )
