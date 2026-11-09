from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]], boundary: str) -> bytes:
    """Build a multipart/form-data body the way a browser would."""
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += value.encode() + b"\r\n"
    for name, (filename, data) in files.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        out += b"Content-Type: application/octet-stream\r\n\r\n"
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A 2-second 160x120 MJPG video with a moving square."""
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    assert writer.isOpened()
    for i in range(20):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (5 + i * 5, 40), (25 + i * 5, 60), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def isolated_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every module that stores projects at a temporary folder."""
    from retail_theft_detector import project, web_app

    projects_dir = tmp_path / "projects"
    monkeypatch.setattr(project, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(web_app, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(web_app, "TMP_DIR", tmp_path / "tmp")
    return projects_dir
