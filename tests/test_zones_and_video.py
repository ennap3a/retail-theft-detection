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
CHECKOUT = {"id": "till", "type": "checkout", "points": square(200, 100, 200)}
EXIT = {"id": "door", "type": "exit", "points": square(450, 300, 100)}
LEGACY_CUSTOMER = {"id": "aisle", "type": "customer", "points": square(200, 100, 200)}


def test_polygon_area() -> None:
    assert polygon_area(square(0, 0, 10)) == 100
    assert polygon_area(square(0, 0, 10)[:2]) == 0


def test_product_zone_alone_is_enough() -> None:
    doc = zone_doc(PRODUCT)
    assert validate_zone_document(doc) == []
    assert zone_counts(doc) == {"product": 1, "checkout": 0, "exit": 0}


def test_optional_checkout_and_exit_zones() -> None:
    doc = zone_doc(PRODUCT, CHECKOUT, EXIT, {**EXIT, "id": "door-2"})
    assert validate_zone_document(doc) == []
    assert zone_counts(doc) == {"product": 1, "checkout": 1, "exit": 2}


def test_zone_files_from_older_versions_still_load() -> None:
    assert validate_zone_document(zone_doc(PRODUCT, LEGACY_CUSTOMER)) == []


@pytest.mark.parametrize(
    ("zones", "message"),
    [
        ([], "at least one product zone"),
        ([CHECKOUT, EXIT], "at least one product zone"),
        ([PRODUCT, {**PRODUCT}], "duplicate id"),
        ([PRODUCT, {**CHECKOUT, "points": square(600, 10, 100)}], "outside the frame"),
        ([PRODUCT, {**EXIT, "points": square(10, 10, 2)}], "too small"),
        ([PRODUCT, {**EXIT, "type": "parking"}], "type must be one of"),
    ],
)
def test_invalid_zone_documents(zones: list[dict], message: str) -> None:
    errors = validate_zone_document(zone_doc(*zones))
    assert any(message in error for error in errors), errors


def test_missing_frame_size() -> None:
    assert "Missing frame width/height." in validate_zone_document({"zones": [PRODUCT]})


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
