"""Skeleton-based action model for concealment, and the combined behaviour scorer.

Each tracked person is cut into short overlapping windows (4 s, one every second). A window is a
sequence of 17 body keypoints per frame, normalised to the person's own position and height, so
the model sees *how the body moves* (reaching, a hand going to a pocket, waistband or bag) and not
the store, camera or lighting. A small spatial-temporal graph network (ST-GCN style) scores every
window; a person's score is their most suspicious window.

Training only has clip-level labels ("this clip contains theft somewhere"), so the network is
trained with multiple-instance learning: a clip's score is the maximum over all its people and
windows, and only that maximum is compared with the label.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import BEHAVIOUR_CARD_PATH as CARD_PATH
from .config import MOVEMENT_MODEL_PATH as MOVEMENT_PATH
from .config import SKELETON_WEIGHTS_PATH as WEIGHTS_PATH
from .pose_tracks import KP_CONF, L_HIP, R_HIP, ObservationRecorder

FPS = 8.0
WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 1.0
T = int(FPS * WINDOW_SECONDS)  # frames per window
V = 17  # COCO keypoints
MIN_PRESENT = 0.5  # a window needs the person in at least half its frames
CHANNELS = ("x", "y", "conf", "dx", "dy", "body_vx", "body_vy", "hand_bag", "bag_near")
C = len(CHANNELS)

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]  # fmt: skip
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # left/right keypoint swap


# --- windows ---------------------------------------------------------------------------


@dataclass
class PersonWindow:
    track_id: str
    start: float
    end: float
    peak_time: float  # frame inside the window where the hands were closest to the hips
    box_at_peak: tuple[float, float, float, float]


def person_windows(recorder: ObservationRecorder) -> tuple[np.ndarray, list[PersonWindow]]:
    """Cut every tracked person into fixed-rate windows. Returns (N, C, T, V) float32 and metadata."""
    by_track: dict[str, list] = defaultdict(list)
    for obs in recorder.observations:
        by_track[obs.track_id].append(obs)

    tensors: list[np.ndarray] = []
    meta: list[PersonWindow] = []
    step = 1.0 / FPS
    for track_id, obs in by_track.items():
        obs.sort(key=lambda o: o.time)
        times = np.array([o.time for o in obs])
        first, last = times[0], times[-1]
        if last - first < 1.0:
            continue
        span = max(last - first, WINDOW_SECONDS)
        starts = np.arange(first, first + span - WINDOW_SECONDS + 1e-6, STRIDE_SECONDS)
        for start in starts:
            grid = start + np.arange(T) * step
            idx = np.searchsorted(times, grid)
            idx = np.clip(idx, 0, len(times) - 1)
            prev = np.clip(idx - 1, 0, len(times) - 1)
            idx = np.where(np.abs(times[prev] - grid) < np.abs(times[idx] - grid), prev, idx)
            present = np.abs(times[idx] - grid) <= 0.75 * step + 0.08
            if present.mean() < MIN_PRESENT:
                continue
            frames = [obs[i] if ok else None for i, ok in zip(idx, present, strict=True)]
            tensor, peak = _encode(frames)
            tensors.append(tensor)
            peak_obs = frames[peak] or next(f for f in frames if f is not None)
            meta.append(
                PersonWindow(track_id, float(start), float(start + WINDOW_SECONDS), peak_obs.time, peak_obs.box)
            )
    if not tensors:
        return np.zeros((0, C, T, V), dtype=np.float32), []
    return np.stack(tensors).astype(np.float32), meta


def _encode(frames: list) -> tuple[np.ndarray, int]:
    out = np.zeros((C, T, V), dtype=np.float32)
    centres = np.full((T, 2), np.nan)
    hand_hip = np.full(T, np.inf)
    for t, o in enumerate(frames):
        if o is None:
            continue
        x1, y1, x2, y2 = o.box
        h = max(y2 - y1, 1.0)
        kps = o.keypoints
        hips = [kps[i, :2] for i in (L_HIP, R_HIP) if kps[i, 2] >= KP_CONF]
        origin = np.mean(hips, axis=0) if hips else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        visible = kps[:, 2] >= KP_CONF
        xy = (kps[:, :2] - origin) / h
        out[0, t] = np.where(visible, xy[:, 0], 0)
        out[1, t] = np.where(visible, xy[:, 1], 0)
        out[2, t] = kps[:, 2]
        centres[t] = ((x1 + x2) / 2 / h, (y1 + y2) / 2 / h)
        out[7, t] = min(o.hand_bag_distance, 3.0) / 3.0 if np.isfinite(o.hand_bag_distance) else 1.0
        out[8, t] = float(o.bag_distance <= 1.0)
        wrists = [np.hypot(*xy[i]) for i in (9, 10) if visible[i]]
        if wrists and hips:
            hand_hip[t] = min(wrists)
    vis = out[2] >= KP_CONF
    both = vis[1:] & vis[:-1]
    out[3, 1:] = np.where(both, out[0, 1:] - out[0, :-1], 0)
    out[4, 1:] = np.where(both, out[1, 1:] - out[1, :-1], 0)
    vel = np.nan_to_num(np.diff(centres, axis=0))
    out[5, 1:] = np.clip(vel[:, 0], -0.5, 0.5)[:, None]
    out[6, 1:] = np.clip(vel[:, 1], -0.5, 0.5)[:, None]
    peak = int(np.argmin(hand_hip)) if np.isfinite(hand_hip).any() else T // 2
    return out, peak


def augment(batch: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Training-time augmentation: mirror, small rotation/scale, keypoint dropout."""
    x = batch.clone()
    n = x.shape[0]
    flip = torch.rand(n, generator=generator) < 0.5
    if flip.any():
        mirrored = x[flip][:, :, :, FLIP]
        for ch in (0, 3, 5):  # x-like channels change sign
            mirrored[:, ch] = -mirrored[:, ch]
        x[flip] = mirrored
    angle = (torch.rand(n, generator=generator) - 0.5) * 0.3
    scale = 1 + (torch.rand(n, generator=generator) - 0.5) * 0.2
    cos, sin = torch.cos(angle)[:, None, None] * scale[:, None, None], torch.sin(angle)[:, None, None] * scale[:, None, None]
    for cx, cy in ((0, 1), (3, 4)):
        px, py = x[:, cx].clone(), x[:, cy].clone()
        x[:, cx], x[:, cy] = cos * px - sin * py, sin * px + cos * py
    drop = torch.rand(n, 1, x.shape[2], x.shape[3], generator=generator) < 0.05
    x[:, :5] = x[:, :5].masked_fill(drop, 0)
    return x


# --- network ---------------------------------------------------------------------------


def _adjacency() -> torch.Tensor:
    a = torch.eye(V)
    for i, j in COCO_EDGES:
        a[i, j] = a[j, i] = 1
    d = a.sum(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.gcn = nn.Conv2d(c_in, c_out, 1)
        self.edge = nn.Parameter(torch.ones(V, V))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        if c_in == c_out and stride == 1:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out))
        self.register_buffer("A", _adjacency())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("nctv,vw->nctw", self.gcn(x), self.A * self.edge)
        return torch.relu(self.tcn(y) + self.residual(x))


class SkeletonNet(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(C * V)
        self.blocks = nn.Sequential(
            STGCNBlock(C, width),
            STGCNBlock(width, width),
            STGCNBlock(width, width * 2, stride=2),
            STGCNBlock(width * 2, width * 2),
        )
        self.head = nn.Linear(width * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        x = self.norm(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(n, c, v, t).permute(0, 1, 3, 2)
        x = self.blocks(x)
        return self.head(x.mean(dim=(2, 3))).squeeze(1)


# --- inference -------------------------------------------------------------------------


class BehaviourScorer:
    """Scores every person in an analysed video with the trained behaviour model.

    The behaviour model averages two components trained on the same clips: the skeleton network
    above and a gradient-boosting model on hand-crafted movement features (:mod:`behaviour`).
    If the movement model file is missing, the skeleton network is used on its own.
    """

    def __init__(
        self, weights: Path = WEIGHTS_PATH, movement: Path = MOVEMENT_PATH, card: Path = CARD_PATH
    ) -> None:
        self.card = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}
        self.threshold = float(self.card.get("threshold", 0.5))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.models = []
        for member in state["members"]:
            net = SkeletonNet(width=int(state.get("width", 32)))
            net.load_state_dict(member)
            net.eval()
            self.models.append(net)
        self.movement = None
        if movement.is_file():
            import joblib

            self.movement = joblib.load(movement)

    @staticmethod
    def available(weights: Path = WEIGHTS_PATH) -> bool:
        return weights.is_file()

    @torch.no_grad()
    def score(self, recorder: ObservationRecorder) -> list[dict[str, Any]]:
        """One result per tracked person (most suspicious moment), highest score first."""
        windows, meta = person_windows(recorder)
        if not meta:
            return []
        x = torch.from_numpy(windows)
        scores = np.mean([torch.sigmoid(m(x)).numpy() for m in self.models], axis=0)
        best: dict[str, dict[str, Any]] = {}
        for s, m in zip(scores, meta, strict=True):
            if m.track_id not in best or s > best[m.track_id]["skeleton_score"]:
                best[m.track_id] = {
                    "track_id": m.track_id,
                    "skeleton_score": float(s),
                    "start": m.start,
                    "end": m.end,
                    "peak_time": m.peak_time,
                    "box": [int(v) for v in m.box_at_peak],
                }

        movement = self._movement_scores(recorder)
        for result in best.values():
            parts = [result["skeleton_score"]]
            if result["track_id"] in movement:
                result["movement_score"] = movement[result["track_id"]]
                parts.append(movement[result["track_id"]])
            result["score"] = float(np.mean(parts))
        return sorted(best.values(), key=lambda r: r["score"], reverse=True)

    def _movement_scores(self, recorder: ObservationRecorder) -> dict[str, float]:
        if self.movement is None:
            return {}
        from .behaviour import track_windows

        track_wins = track_windows(recorder)
        if not track_wins:
            return {}
        proba = self.movement.predict_proba(np.vstack([tw.features for tw in track_wins]))[:, 1]
        scores: dict[str, float] = {}
        for tw, p in zip(track_wins, proba, strict=True):
            scores[tw.track_id] = max(scores.get(tw.track_id, 0.0), float(p))
        return scores
