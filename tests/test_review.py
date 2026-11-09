from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

import pytest

from retail_theft_detector import review
from retail_theft_detector.project import paths_for_project, write_json


def test_create_project_copies_video_and_extracts_frame(sample_video: Path, isolated_projects: Path) -> None:
    project = review.create_project(sample_video, "Store Cam 2.avi", move=False)

    paths = paths_for_project(project["project_id"])
    assert sample_video.exists()
    assert (paths.input_dir / "Store_Cam_2.avi").exists()
    assert paths.first_frame_path.exists()
    assert project["video"]["info"]["frames"] == 20


def test_create_project_rejects_non_video(tmp_path: Path, isolated_projects: Path) -> None:
    bad = tmp_path / "fake.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError):
        review.create_project(bad, "fake.mp4")
    assert not isolated_projects.exists() or not any(isolated_projects.iterdir())


def test_delete_project(sample_video: Path, isolated_projects: Path) -> None:
    project = review.create_project(sample_video, "a.avi", move=False)
    review.delete_project(project["project_id"])
    assert not paths_for_project(project["project_id"]).base_dir.exists()


def test_cleanup_old_projects(isolated_projects: Path) -> None:
    old, new = isolated_projects / "old", isolated_projects / "new"
    old.mkdir(parents=True)
    new.mkdir()
    two_days_ago = time.time() - 48 * 3600
    os.utime(old, (two_days_ago, two_days_ago))

    assert review.cleanup_old_projects(0) == 0
    assert review.cleanup_old_projects(24) == 1
    assert not old.exists() and new.exists()


def test_review_package_and_results(sample_video: Path, isolated_projects: Path) -> None:
    project = review.create_project(sample_video, "clip.avi", move=False)
    paths = paths_for_project(project["project_id"])
    write_json(paths.results_dir / "alerts.json", {"alerts": [{"alert_id": "A1"}]})
    write_json(paths.results_dir / "summary.json", {"results": {"total_alerts": 1}})

    summary, alerts = review.load_results(project["project_id"])
    assert summary["results"]["total_alerts"] == 1
    assert alerts == [{"alert_id": "A1"}]

    package = review.build_review_package(project["project_id"], project)
    names = zipfile.ZipFile(package).namelist()
    assert "README.txt" in names
    assert "videos/original_clip.avi" in names
    assert "reports/alerts.json" in names
