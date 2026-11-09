"""Pure image helpers for drawing zones on the setup frame (no Streamlit imports)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# RGB colours
ZONE_COLORS = {
    "product": (45, 212, 191),
    "customer": (96, 165, 250),
}
DRAFT_COLOR = (250, 204, 21)


def _points(points: list[dict[str, float]]) -> np.ndarray:
    return np.array([[round(p["x"]), round(p["y"])] for p in points], dtype=np.int32)


def render_zones(
    frame_rgb: np.ndarray,
    zones: list[dict[str, Any]],
    draft_points: list[dict[str, float]] | None = None,
) -> np.ndarray:
    """Return a copy of ``frame_rgb`` with saved zones filled and the draft polygon outlined."""
    canvas = frame_rgb.copy()
    overlay = frame_rgb.copy()
    thickness = max(2, round(max(frame_rgb.shape[:2]) / 400))

    for zone in zones:
        if len(zone.get("points", [])) < 3:
            continue
        pts = _points(zone["points"])
        cv2.fillPoly(overlay, [pts], ZONE_COLORS.get(zone["type"], (255, 255, 255)))
    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

    for zone in zones:
        if len(zone.get("points", [])) < 3:
            continue
        color = ZONE_COLORS.get(zone["type"], (255, 255, 255))
        pts = _points(zone["points"])
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        _label(canvas, str(zone.get("label", zone["id"])), tuple(pts.min(axis=0)), color)

    if draft_points:
        pts = _points(draft_points)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, DRAFT_COLOR, thickness, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(canvas, (int(x), int(y)), thickness + 3, DRAFT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), thickness + 3, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    scale = max(0.45, image.shape[1] / 1600)
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(max(0, min(origin[0], image.shape[1] - w - 8)))
    y = int(max(h + 8, origin[1]))
    cv2.rectangle(image, (x, y - h - 8), (x + w + 8, y + base - 2), (15, 23, 42), -1)
    cv2.putText(image, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def zone_document(
    project_id: str,
    frame_width: int,
    frame_height: int,
    setup_seconds: float,
    zones: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the zones.json document expected by the detector."""
    return {
        "schema_version": 1,
        "project_id": project_id,
        "frame": {
            "width": frame_width,
            "height": frame_height,
            "source": "setup_frame.jpg",
            "seconds": round(float(setup_seconds), 3),
        },
        "zones": zones,
        "instructions_version": "streamlit_v1",
    }


def finish_zone(
    zones: list[dict[str, Any]],
    draft_points: list[dict[str, float]],
    zone_type: str,
) -> list[dict[str, Any]]:
    """Return a new zone list with the draft polygon added.

    A new customer zone replaces the previous one; product zones are numbered.
    """
    if len(draft_points) < 3:
        raise ValueError("A zone needs at least 3 points.")
    if zone_type == "customer":
        kept = [z for z in zones if z["type"] != "customer"]
        new = {"id": "customer_zone", "type": "customer", "label": "Customer Zone"}
    elif zone_type == "product":
        kept = list(zones)
        used = {z["id"] for z in zones}
        n = 1
        while f"product_zone_{n}" in used:
            n += 1
        new = {"id": f"product_zone_{n}", "type": "product", "label": f"Product Zone {n}"}
    else:
        raise ValueError(f"Unknown zone type: {zone_type}")
    new["points"] = [{"x": round(p["x"], 1), "y": round(p["y"], 1)} for p in draft_points]
    return [*kept, new]
