from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECTS_DIR


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_project_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class ProjectPaths:
    project_id: str
    base_dir: Path
    input_dir: Path
    setup_dir: Path
    results_dir: Path
    metadata_path: Path
    zones_path: Path
    first_frame_path: Path


def paths_for_project(project_id: str) -> ProjectPaths:
    base = PROJECTS_DIR / project_id
    return ProjectPaths(
        project_id=project_id,
        base_dir=base,
        input_dir=base / "input",
        setup_dir=base / "setup",
        results_dir=base / "results",
        metadata_path=base / "project.json",
        zones_path=base / "setup" / "zones.json",
        first_frame_path=base / "setup" / "setup_frame.jpg",
    )


def ensure_project_dirs(paths: ProjectPaths) -> None:
    for folder in (paths.base_dir, paths.input_dir, paths.setup_dir, paths.results_dir):
        folder.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_project(project_id: str) -> dict[str, Any]:
    paths = paths_for_project(project_id)
    if not paths.metadata_path.exists():
        raise FileNotFoundError(f"Project not found: {project_id}")
    return read_json(paths.metadata_path)


def update_project(project_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    project = load_project(project_id)
    project.update(updates)
    project["updated_at"] = utc_now_iso()
    write_json(paths_for_project(project_id).metadata_path, project)
    return project


def list_projects() -> list[dict[str, Any]]:
    if not PROJECTS_DIR.exists():
        return []
    projects = []
    for metadata_path in PROJECTS_DIR.glob("*/project.json"):
        try:
            projects.append(read_json(metadata_path))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(projects, key=lambda item: item.get("created_at", ""), reverse=True)
