from __future__ import annotations

from pathlib import Path

import pytest

from retail_theft_detector.video import extract_frame, is_allowed_video, probe_video, safe_filename
from retail_theft_detector.zones import polygon_area, validate_zone_document, zone_counts


def square(x: float, y: float, size: float) -> list[dict[str, float]]:
    return [{"x": x, "y": y}, {"x": x + size, "y": y}, {"x": x + size, "y": y + size}, {"x": x, "y": y + size}]


def zone_doc(*zones: dict) -> dict:
    return {"frame": {"width": 640, "height": 480}, "zones": list(zones)}


PRODUCT = {"id": "shelf-1", "type": "product", "points": square(10, 10, 100)}
CUSTOMER = {"id": "aisle", "type": "customer", "points": square(200, 100, 200)}


def test_polygon_area() -> None:
    assert polygon_area(square(0, 0, 10)) == 100
    assert polygon_area(square(0, 0, 10)[:2]) == 0


def test_valid_zone_document() -> None:
    doc = zone_doc(PRODUCT, CUSTOMER)
    assert validate_zone_document(doc) == []
    assert zone_counts(doc) == {"product": 1, "customer": 1}


@pytest.mark.parametrize(
    ("zones", "message"),
    [
        ([PRODUCT], "exactly one customer zone"),
        ([CUSTOMER], "at least one product zone"),
        ([PRODUCT, CUSTOMER, {**CUSTOMER, "id": "aisle-2"}], "exactly one customer zone"),
        ([PRODUCT, {**PRODUCT}, CUSTOMER], "duplicate id"),
        ([PRODUCT, {**CUSTOMER, "points": square(600, 10, 100)}], "outside the frame"),
        ([PRODUCT, {**CUSTOMER, "points": square(10, 10, 2)}], "too small"),
        ([PRODUCT, {**CUSTOMER, "type": "exit"}], "type must be product or customer"),
    ],
)
def test_invalid_zone_documents(zones: list[dict], message: str) -> None:
    errors = validate_zone_document(zone_doc(*zones))
    assert any(message in error for error in errors), errors


def test_missing_frame_size() -> None:
    assert "Missing frame width/height." in validate_zone_document({"zones": [PRODUCT, CUSTOMER]})


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("store cam (1).mp4", "store_cam__1_.mp4"),
        ("../../etc/passwd", "passwd"),
        ("...", "uploaded_video.mp4"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


def test_allowed_video_extensions() -> None:
    assert is_allowed_video("CAM4.MP4")
    assert is_allowed_video("clip.webm")
    assert not is_allowed_video("notes.txt")


def test_probe_and_extract_frame(sample_video: Path, tmp_path: Path) -> None:
    info = probe_video(sample_video)
    assert (info.width, info.height) == (160, 120)
    assert info.frames == 20
    assert info.duration_seconds == pytest.approx(2.0, abs=0.2)

    frame_path = extract_frame(sample_video, tmp_path / "frame.jpg", at_seconds=1.0)
    assert frame_path.exists() and frame_path.stat().st_size > 0
