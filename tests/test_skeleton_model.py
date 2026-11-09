from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retail_theft_detector.pose_tracks import L_WRI, ObservationRecorder
from retail_theft_detector.zones import alert_priority

torch = pytest.importorskip("torch")
skeleton_model = pytest.importorskip("retail_theft_detector.skeleton_model")


def standing_pose(x: float, y: float, h: float, wrist_drop: float = 0.0) -> list[tuple[float, float, float]]:
    """A rough upright COCO skeleton inside a box of height ``h`` whose top-left is (x, y)."""
    rel = [
        (0.5, 0.08), (0.47, 0.06), (0.53, 0.06), (0.44, 0.07), (0.56, 0.07),
        (0.38, 0.2), (0.62, 0.2), (0.33, 0.35), (0.67, 0.35), (0.3, 0.5), (0.7, 0.5),
        (0.42, 0.52), (0.58, 0.52), (0.42, 0.75), (0.58, 0.75), (0.42, 0.97), (0.58, 0.97),
    ]  # fmt: skip
    pts = [(x + rx * h * 0.5, y + ry * h, 0.9) for rx, ry in rel]
    lx, ly, c = pts[L_WRI]
    pts[L_WRI] = (lx, ly + wrist_drop * h, c)
    return pts


def recorder_for(seconds: float, fps: float = 8.0, people: int = 1) -> ObservationRecorder:
    rec = ObservationRecorder()
    for i in range(int(seconds * fps)):
        t = i / fps
        for p in range(people):
            x, y, h = 50 + 100 * p + 5 * t, 40, 200
            rec.add(t, str(p + 1), (x, y, x + 100, y + h), standing_pose(x, y, h, 0.1 * np.sin(t)), [])
    return rec


def test_recorder_round_trip() -> None:
    rec = recorder_for(2, people=2)
    rec.add(2.0, "9", (0, 0, 50, 100), None, [(10, 10, 30, 30)])
    restored = ObservationRecorder.from_arrays(rec.to_arrays())
    assert len(restored.observations) == len(rec.observations)
    last = restored.observations[-1]
    assert last.track_id == "9"
    assert not last.keypoints.any()
    assert last.bag_distance == pytest.approx(np.hypot(5, 30) / 100, rel=1e-4)


def test_person_windows_shape_and_normalisation() -> None:
    windows, meta = skeleton_model.person_windows(recorder_for(10, people=2))
    per_person = int((10 - 1 / 8 - skeleton_model.WINDOW_SECONDS) // skeleton_model.STRIDE_SECONDS) + 1
    assert windows.shape == (2 * per_person, skeleton_model.C, skeleton_model.T, skeleton_model.V)
    assert np.isfinite(windows).all()
    # coordinates are relative to the hips and scaled by person height, so they stay small
    assert np.abs(windows[:, :2]).max() < 1.5
    assert {m.track_id for m in meta} == {"1", "2"}


def test_short_tracks_are_skipped() -> None:
    windows, meta = skeleton_model.person_windows(recorder_for(0.5))
    assert len(windows) == 0 and meta == []


def test_scorer_round_trip(tmp_path: Path) -> None:
    net = skeleton_model.SkeletonNet(width=8).eval()
    weights, card = tmp_path / "m.pt", tmp_path / "m.json"
    torch.save({"width": 8, "members": [net.state_dict()]}, weights)
    card.write_text(json.dumps({"threshold": 0.42}))

    scorer = skeleton_model.BehaviourScorer(weights, tmp_path / "missing.joblib", card)
    results = scorer.score(recorder_for(6, people=2))

    assert scorer.threshold == 0.42
    assert {r["track_id"] for r in results} == {"1", "2"}
    assert all(0 <= r["score"] <= 1 for r in results)
    assert results[0]["score"] >= results[1]["score"]


def test_augment_keeps_shape() -> None:
    windows, _ = skeleton_model.person_windows(recorder_for(5))
    x = torch.from_numpy(windows)
    out = skeleton_model.augment(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("visits", "after", "priority"),
    [
        (None, 3.0, "MEDIUM"),
        ({"checkout": [], "exit": [8.0]}, 3.0, "HIGH"),
        ({"checkout": [6.0], "exit": [8.0]}, 3.0, "LOW"),
        ({"checkout": [9.0], "exit": [8.0]}, 3.0, "HIGH"),
        ({"checkout": [], "exit": [1.0]}, 3.0, "MEDIUM"),  # exit before the event does not count
    ],
)
def test_alert_priority(visits: dict | None, after: float, priority: str) -> None:
    assert alert_priority(visits, after)[0] == priority


def test_scorer_averages_skeleton_and_movement_models(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    from sklearn.dummy import DummyClassifier

    from retail_theft_detector.behaviour import N_FEATURES

    net = skeleton_model.SkeletonNet(width=8).eval()
    weights, movement, card = tmp_path / "m.pt", tmp_path / "m.joblib", tmp_path / "m.json"
    torch.save({"width": 8, "members": [net.state_dict()]}, weights)
    always_suspicious = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, N_FEATURES)), [0, 1])
    joblib.dump(always_suspicious, movement)
    card.write_text(json.dumps({"threshold": 0.5}))

    results = skeleton_model.BehaviourScorer(weights, movement, card).score(recorder_for(6))

    assert len(results) == 1
    result = results[0]
    assert result["movement_score"] == 1.0
    assert result["score"] == pytest.approx((result["skeleton_score"] + 1.0) / 2)
