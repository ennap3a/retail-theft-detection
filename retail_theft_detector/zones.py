from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Only product zones are required: everything outside them is the shop floor.
# Checkout and exit zones are optional and only change how urgent an alert is.
# "customer" is accepted so zone files saved by older versions still load (it is ignored).
ZONE_TYPES = ("product", "checkout", "exit")
LEGACY_ZONE_TYPES = ("customer",)


def polygon_area(points: list[dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += point["x"] * nxt["y"] - nxt["x"] * point["y"]
    return abs(area) / 2.0


def validate_zone_document(doc: dict[str, Any]) -> list[str]:
    errors = []
    frame = doc.get("frame") or {}
    width = float(frame.get("width") or 0)
    height = float(frame.get("height") or 0)
    zones = doc.get("zones")

    if width <= 0 or height <= 0:
        errors.append("Missing frame width/height.")
    if not isinstance(zones, list):
        return errors + ["Zones must be a list."]

    product_count = 0
    seen_ids = set()

    for index, zone in enumerate(zones, start=1):
        zone_id = str(zone.get("id") or "").strip()
        zone_type = str(zone.get("type") or "").strip()
        points = zone.get("points")
        label = zone.get("label") or zone_id or f"Zone {index}"

        if not zone_id:
            errors.append(f"{label}: missing id.")
        elif zone_id in seen_ids:
            errors.append(f"{label}: duplicate id.")
        seen_ids.add(zone_id)

        if zone_type not in ZONE_TYPES + LEGACY_ZONE_TYPES:
            errors.append(f"{label}: type must be one of {', '.join(ZONE_TYPES)}.")
        if zone_type == "product":
            product_count += 1

        if not isinstance(points, list) or len(points) < 3:
            errors.append(f"{label}: draw at least 3 points.")
            continue

        for point in points:
            try:
                x = float(point["x"])
                y = float(point["y"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label}: every point needs numeric x/y.")
                continue
            if width and not 0 <= x <= width:
                errors.append(f"{label}: point x={x:.1f} is outside the frame.")
            if height and not 0 <= y <= height:
                errors.append(f"{label}: point y={y:.1f} is outside the frame.")

        if polygon_area(points) < 25:
            errors.append(f"{label}: polygon is too small.")

    if product_count < 1:
        errors.append("Draw at least one product zone.")

    return errors


def load_zones(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_zones(path: Path, doc: dict[str, Any]) -> None:
    errors = validate_zone_document(doc)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def zone_counts(doc: dict[str, Any] | None) -> dict[str, int]:
    counts = dict.fromkeys(ZONE_TYPES, 0)
    if not doc:
        return counts
    for zone in doc.get("zones", []):
        zone_type = zone.get("type")
        if zone_type in counts:
            counts[zone_type] += 1
    return counts


def alert_priority(visits: dict[str, list[float]] | None, after: float) -> tuple[str, str]:
    """Alert priority from what the person did after ``after`` seconds: exit without paying → HIGH."""
    visits = visits or {"checkout": [], "exit": []}
    checkout = [t for t in visits["checkout"] if t >= after]
    exits = [t for t in visits["exit"] if t >= after]
    if exits and (not checkout or min(exits) < min(checkout)):
        return "HIGH", f" Reached an exit at {min(exits):.1f}s without passing a checkout."
    if checkout:
        return "LOW", f" Went to a checkout at {min(checkout):.1f}s: check whether the item was paid for."
    return "MEDIUM", ""
