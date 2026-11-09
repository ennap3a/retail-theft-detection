from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import build_multipart

from retail_theft_detector import web_app
from retail_theft_detector.review import classify_evaluation
from retail_theft_detector.web_app import RetailTheftHandler

BOUNDARY = "----TestBoundary7MA4YWxk"


@pytest.mark.parametrize(
    ("real_case", "model_review", "expected"),
    [
        ("normal", "correct_no_alert", "TN"),
        ("normal", "false_alert", "FP"),
        ("suspicious", "correct_alert", "TP"),
        ("suspicious", "missed_event", "FN"),
        ("suspicious", "wrong_timestamp", "FN"),
        ("", "", "REVIEW"),
    ],
)
def test_classify_evaluation(real_case: str, model_review: str, expected: str) -> None:
    assert classify_evaluation(real_case, model_review) == expected


@pytest.fixture
def server(isolated_projects: Path) -> Iterator[tuple[str, int]]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), RetailTheftHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[0], httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def post_upload(server: tuple[str, int], body: bytes) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request(
        "POST",
        "/upload",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    return conn.getresponse()


def test_home_page(server: tuple[str, int]) -> None:
    conn = http.client.HTTPConnection(*server, timeout=10)
    conn.request("GET", "/")
    response = conn.getresponse()
    assert response.status == 200
    assert b'name="video"' in response.read()


def test_upload_creates_project(server: tuple[str, int], sample_video: Path, isolated_projects: Path) -> None:
    body = build_multipart({}, {"video": ("store cam.avi", sample_video.read_bytes())}, BOUNDARY)

    response = post_upload(server, body)

    assert response.status == 303
    project_id = response.getheader("Location").rsplit("/", 1)[-1]
    project = json.loads((isolated_projects / project_id / "project.json").read_text())
    assert project["video"]["original_filename"] == "store cam.avi"
    assert project["video"]["info"]["width"] == 160
    stored = isolated_projects / project_id / "input" / "store_cam.avi"
    assert stored.read_bytes() == sample_video.read_bytes()
    assert (isolated_projects / project_id / "setup" / "setup_frame.jpg").exists()
    assert list(Path(web_app.TMP_DIR).glob("upload_*")) == []


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"video": ("notes.txt", b"hello")}, b"Unsupported video type"),
        ({"video": ("", b"")}, b"Choose a video file"),
        ({"other": ("a.mp4", b"x")}, b"did not include a video file"),
    ],
)
def test_upload_rejects_bad_input(
    server: tuple[str, int], isolated_projects: Path, files: dict, message: bytes
) -> None:
    response = post_upload(server, build_multipart({}, files, BOUNDARY))

    assert response.status == 400
    assert message in response.read()
    assert not isolated_projects.exists() or not any(isolated_projects.iterdir())
    assert list(Path(web_app.TMP_DIR).glob("upload_*")) == []


def test_upload_rejects_malformed_body(server: tuple[str, int]) -> None:
    response = post_upload(server, b"this is not multipart")
    assert response.status == 400
