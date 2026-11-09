from __future__ import annotations

import io
from pathlib import Path

import pytest
from conftest import build_multipart

from retail_theft_detector import multipart
from retail_theft_detector.multipart import MultipartError, parse_multipart

BOUNDARY = "----WebKitFormBoundaryAbC123"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"


def parse(body: bytes, tmp_path: Path, content_type: str = CONTENT_TYPE):
    return parse_multipart(io.BytesIO(body), content_type, len(body), tmp_dir=tmp_path)


def test_parses_fields_and_file(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 50
    body = build_multipart({"note": "hello", "real_case": "normal"}, {"video": ("clip.mp4", payload)}, BOUNDARY)

    form = parse(body, tmp_path)

    assert form.fields == {"note": ["hello"], "real_case": ["normal"]}
    upload = form.first_file("video")
    assert upload is not None
    assert upload.filename == "clip.mp4"
    assert upload.size == len(payload)
    assert upload.path.read_bytes() == payload


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
def test_delimiter_split_across_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_size: int) -> None:
    monkeypatch.setattr(multipart, "CHUNK_SIZE", chunk_size)
    # Content that contains a near-miss of the delimiter must survive intact.
    payload = b"abc\r\n--" + BOUNDARY[:-1].encode() + b"x\r\nend"
    body = build_multipart({}, {"video": ("a.mp4", payload)}, BOUNDARY)

    form = parse(body, tmp_path)

    assert form.first_file("video").path.read_bytes() == payload


def test_empty_file_input(tmp_path: Path) -> None:
    body = build_multipart({}, {"video": ("", b"")}, BOUNDARY)
    upload = parse(body, tmp_path).first_file("video")
    assert upload.filename == ""
    assert upload.size == 0


def test_utf8_filename(tmp_path: Path) -> None:
    body = build_multipart({}, {"video": ("كاميرا 1.mp4", b"data")}, BOUNDARY)
    assert parse(body, tmp_path).first_file("video").filename == "كاميرا 1.mp4"


def test_quoted_boundary(tmp_path: Path) -> None:
    body = build_multipart({"a": "1"}, {}, BOUNDARY)
    form = parse(body, tmp_path, content_type=f'multipart/form-data; boundary="{BOUNDARY}"')
    assert form.fields == {"a": ["1"]}


def test_truncated_upload_removes_temp_file(tmp_path: Path) -> None:
    body = build_multipart({}, {"video": ("a.mp4", b"x" * 5000)}, BOUNDARY)
    with pytest.raises(MultipartError):
        parse(body[:3000], tmp_path)
    assert list(tmp_path.glob("upload_*")) == []


def test_cleanup_deletes_unmoved_files(tmp_path: Path) -> None:
    body = build_multipart({}, {"video": ("a.mp4", b"data")}, BOUNDARY)
    form = parse(body, tmp_path)
    form.cleanup()
    assert list(tmp_path.glob("upload_*")) == []


@pytest.mark.parametrize(
    "content_type",
    ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain; boundary=x"],
)
def test_rejects_wrong_content_type(tmp_path: Path, content_type: str) -> None:
    with pytest.raises(MultipartError):
        parse(b"--x--\r\n", tmp_path, content_type=content_type)


def test_rejects_body_without_boundary(tmp_path: Path) -> None:
    with pytest.raises(MultipartError):
        parse(b"no multipart here", tmp_path)


def test_rejects_empty_body(tmp_path: Path) -> None:
    with pytest.raises(MultipartError):
        parse_multipart(io.BytesIO(b""), CONTENT_TYPE, 0, tmp_dir=tmp_path)
