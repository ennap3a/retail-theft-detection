from __future__ import annotations

import numpy as np
import pytest

from retail_theft_detector.zones import validate_zone_document
from ui.drawing import finish_zone, render_zones, zone_document

TRIANGLE = [{"x": 10, "y": 10}, {"x": 100, "y": 10}, {"x": 50, "y": 90}]
BOX = [{"x": 120, "y": 20}, {"x": 300, "y": 20}, {"x": 300, "y": 220}, {"x": 120, "y": 220}]


def test_finish_zone_numbers_product_zones() -> None:
    zones = finish_zone([], TRIANGLE, "product")
    zones = finish_zone(zones, TRIANGLE, "product")
    assert [z["id"] for z in zones] == ["product_zone_1", "product_zone_2"]
    assert zones[1]["label"] == "Product Zone 2"


def test_finish_zone_reuses_freed_number() -> None:
    zones = finish_zone(finish_zone([], TRIANGLE, "product"), TRIANGLE, "product")
    zones = [zones[1]]  # delete zone 1
    zones = finish_zone(zones, TRIANGLE, "product")
    assert sorted(z["id"] for z in zones) == ["product_zone_1", "product_zone_2"]


def test_new_customer_zone_replaces_old_one() -> None:
    zones = finish_zone([], TRIANGLE, "customer")
    zones = finish_zone(zones, BOX, "customer")
    customers = [z for z in zones if z["type"] == "customer"]
    assert len(customers) == 1
    assert customers[0]["points"][0] == {"x": 120, "y": 20}


def test_finish_zone_needs_three_points() -> None:
    with pytest.raises(ValueError):
        finish_zone([], TRIANGLE[:2], "product")
    with pytest.raises(ValueError):
        finish_zone([], TRIANGLE, "exit")


def test_zone_document_is_valid_for_detector() -> None:
    zones = finish_zone(finish_zone([], TRIANGLE, "product"), BOX, "customer")
    doc = zone_document("p1", 320, 240, 1.25, zones)
    assert validate_zone_document(doc) == []
    assert doc["frame"] == {"width": 320, "height": 240, "source": "setup_frame.jpg", "seconds": 1.25}


def test_render_zones_draws_without_changing_input() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    zones = finish_zone([], TRIANGLE, "product")
    out = render_zones(frame, zones, draft_points=BOX[:2])
    assert out.shape == frame.shape
    assert out.any()
    assert not frame.any()
