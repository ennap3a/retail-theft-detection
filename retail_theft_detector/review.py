"""Review workflow shared by the web app and the Streamlit app.

Creating projects from uploaded videos, logging reviewer decisions
(TP / FP / FN / TN) and exporting review packages.
"""

from __future__ import annotations

import csv
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from .config import (
    EVALUATION_LOG_PATH,
    LOCAL_REVIEW_DIR,
    NORMAL_TEST_VIDEOS_DIR,
    SUSPICIOUS_TEST_VIDEOS_DIR,
)
from .project import (
    ensure_project_dirs,
    new_project_id,
    paths_for_project,
    read_json,
    utc_now_iso,
    write_json,
)
from .video import extract_frame, probe_video, safe_filename


def create_project(video_file: Path, original_filename: str, *, move: bool = True) -> dict[str, Any]:
    """Store an uploaded video in a new project and extract its first setup frame."""
    project_id = new_project_id()
    paths = paths_for_project(project_id)
    ensure_project_dirs(paths)

    video_path = paths.input_dir / safe_filename(original_filename)
    if move:
        shutil.move(str(video_file), video_path)
    else:
        shutil.copyfile(video_file, video_path)

    try:
        info = probe_video(video_path)
        extract_frame(video_path, paths.first_frame_path, at_seconds=0.0)
    except Exception:
        shutil.rmtree(paths.base_dir, ignore_errors=True)
        raise

    now = utc_now_iso()
    project = {
        "schema_version": 1,
        "project_id": project_id,
        "created_at": now,
        "updated_at": now,
        "setup_frame_seconds": 0.0,
        "zones_saved": False,
        "video": {
            "original_filename": Path(original_filename).name,
            "stored_filename": video_path.name,
            "path": str(video_path),
            "info": info.to_dict(),
        },
        "stage": "setup",
    }
    write_json(paths.metadata_path, project)
    return project


def delete_project(project_id: str) -> None:
    shutil.rmtree(paths_for_project(project_id).base_dir, ignore_errors=True)


def load_results(project_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(summary, alerts)`` for a project the detector has finished."""
    results_dir = paths_for_project(project_id).results_dir
    summary = read_json(results_dir / "summary.json")
    alerts_path = results_dir / "alerts.json"
    alerts = read_json(alerts_path).get("alerts", []) if alerts_path.exists() else []
    return summary, alerts


EVALUATION_FIELDNAMES = [
    "reviewed_at",
    "project_id",
    "video_name",
    "real_case",
    "model_review",
    "result_type",
    "alerts_count",
    "model_alerted",
    "bad_timestamp",
    "mistake_reasons",
    "reviewer_notes",
    "review_package_path",
]


def ensure_local_review_workspace() -> None:
    for folder in (LOCAL_REVIEW_DIR, NORMAL_TEST_VIDEOS_DIR, SUSPICIOUS_TEST_VIDEOS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    if not EVALUATION_LOG_PATH.exists():
        with EVALUATION_LOG_PATH.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=EVALUATION_FIELDNAMES)
            writer.writeheader()


def classify_evaluation(real_case: str, model_review: str) -> str:
    if real_case == "normal":
        if model_review == "correct_no_alert":
            return "TN"
        if model_review in {"false_alert", "wrong_timestamp", "correct_alert"}:
            return "FP"
    if real_case == "suspicious":
        if model_review == "correct_alert":
            return "TP"
        if model_review in {"missed_event", "wrong_timestamp", "false_alert", "correct_no_alert"}:
            return "FN"
    return "REVIEW"


def append_evaluation_row(project_id: str, project: dict, form: dict[str, list[str]]) -> dict[str, str]:
    ensure_local_review_workspace()
    paths = paths_for_project(project_id)
    summary_path = paths.results_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"results": {}}
    results = summary.get("results", {})
    alerts_count = int(results.get("total_alerts", 0) or 0)

    real_case = (form.get("real_case", [""])[0] or "").strip()
    model_review = (form.get("model_review", [""])[0] or "").strip()
    bad_timestamp = (form.get("bad_timestamp", [""])[0] or "").strip()
    reviewer_notes = (form.get("reviewer_notes", [""])[0] or "").strip()
    mistake_reasons = [reason.strip() for reason in form.get("mistake_reason", []) if reason.strip()]
    result_type = classify_evaluation(real_case, model_review)

    package_path = paths.results_dir / safe_filename(
        f"{Path(project.get('video', {}).get('original_filename', 'uploaded_video')).stem}_{project_id}_review_package.zip"
    )

    row = {
        "reviewed_at": utc_now_iso(),
        "project_id": project_id,
        "video_name": project.get("video", {}).get("original_filename", ""),
        "real_case": real_case,
        "model_review": model_review,
        "result_type": result_type,
        "alerts_count": str(alerts_count),
        "model_alerted": "yes" if alerts_count > 0 else "no",
        "bad_timestamp": bad_timestamp,
        "mistake_reasons": "; ".join(mistake_reasons),
        "reviewer_notes": reviewer_notes,
        "review_package_path": str(package_path) if package_path.exists() else "",
    }
    with EVALUATION_LOG_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EVALUATION_FIELDNAMES)
        writer.writerow(row)
    return row



def build_review_package(project_id: str, project: dict) -> Path:
    paths = paths_for_project(project_id)
    summary_path = paths.results_dir / "summary.json"
    alerts_path = paths.results_dir / "alerts.json"
    if not summary_path.exists() or not alerts_path.exists():
        raise RuntimeError("Run the detector before exporting a review package.")

    summary = read_json(summary_path)
    results = summary.get("results", {})
    original_name = project.get("video", {}).get("original_filename", "uploaded_video")
    package_name = safe_filename(f"{Path(original_name).stem}_{project_id}_review_package.zip")
    package_path = paths.results_dir / package_name

    def add_file(zf: zipfile.ZipFile, source: Path, arcname: str) -> None:
        if source.exists() and source.is_file() and source.resolve() != package_path.resolve():
            zf.write(source, arcname)

    notes = f"""Retail Theft Review Package

Project ID: {project_id}
Original video: {original_name}
Total human-review alerts: {results.get("total_alerts", 0)}
Generated at: {utc_now_iso()}

Contents:
- videos/original_*: uploaded camera video
- videos/annotated_review.mp4: model working/review video
- reports/: CSV and JSON result files
- setup/zones.json: product/customer zone setup used for analysis
- evidence/: saved alert images and clips when alerts exist

Responsible AI note:
These outputs are suspected events for trained human review only. They are not automatic accusations or final proof of theft.
"""

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", notes)

        original_path = Path(project.get("video", {}).get("path", ""))
        add_file(zf, original_path, f"videos/original_{safe_filename(original_name)}")

        annotated_rel = results.get("annotated_video", "")
        if annotated_rel:
            add_file(zf, paths.base_dir / annotated_rel, "videos/annotated_review.mp4")

        add_file(zf, paths.results_dir / "alerts.csv", "reports/alerts.csv")
        add_file(zf, paths.results_dir / "review_summary.csv", "reports/review_summary.csv")
        add_file(zf, paths.results_dir / "alerts.json", "reports/alerts.json")
        add_file(zf, paths.results_dir / "summary.json", "reports/summary.json")
        add_file(zf, paths.zones_path, "setup/zones.json")

        evidence_dir = paths.results_dir / "evidence"
        if evidence_dir.exists():
            for evidence_file in evidence_dir.rglob("*"):
                if evidence_file.is_file():
                    add_file(zf, evidence_file, f"evidence/{evidence_file.relative_to(evidence_dir).as_posix()}")

    return package_path


def cleanup_old_projects(max_age_hours: float) -> int:
    """Delete project folders older than ``max_age_hours``. Returns how many were removed.

    Used by hosted demos so uploaded footage is not kept longer than needed.
    A value of 0 or less disables cleanup.
    """
    if max_age_hours <= 0:
        return 0
    from . import project as project_module

    root = project_module.PROJECTS_DIR
    if not root.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for folder in root.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed
