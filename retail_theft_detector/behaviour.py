"""Hand-crafted movement features: one half of the behaviour model.

Each tracked person is summarised over a ~10 second window with interpretable signals (reaching,
hands at the waist or pockets, hands disappearing from view, crouching, hands near a bag, looking
around). A gradient-boosting classifier on these features is averaged with the skeleton network
(:mod:`skeleton_model`); cross-validation showed the combination beats either one alone.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .pose_tracks import (
    KP_CONF,
    L_ELB,
    L_HIP,
    L_KNEE,
    L_SHO,
    L_WRI,
    NOSE,
    R_ELB,
    R_HIP,
    R_KNEE,
    R_SHO,
    R_WRI,
    Observation,
    ObservationRecorder,
)

WINDOW_SECONDS = 10.0
WINDOW_STRIDE_SECONDS = 5.0
MIN_TRACK_SECONDS = 1.0


# --- per-frame signals -----------------------------------------------------------------


def _pt(kps: np.ndarray, i: int) -> np.ndarray | None:
    return kps[i, :2] if kps[i, 2] >= KP_CONF else None


def _mid(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


FRAME_SIGNALS = [
    "wrist_hip_dist_min",  # closest wrist to hip centre (hands at waist / pockets)
    "wrist_hip_dist_max",  # furthest wrist (reaching out)
    "wrist_above_shoulder",  # reaching up to a shelf
    "wrist_in_torso",  # a wrist inside the torso box (under jacket / waistband)
    "wrists_visible",  # 0, 0.5 or 1
    "wrist_hidden_body_visible",  # body is clearly visible but a wrist is not
    "elbow_bent",  # forearm folded back toward the body
    "crouch",  # hips close to knees
    "aspect",  # box height / width
    "shoulder_width",  # shoulder width / height (turned sideways or back to camera)
    "facing",  # nose offset from shoulder centre (head turned)
    "face_visible",
    "hand_bag",  # nearest wrist → bag
    "bag_near",  # bag within one person height
]


def frame_signals(obs: Observation) -> np.ndarray:
    kps = obs.keypoints
    x1, y1, x2, y2 = obs.box
    h = max(y2 - y1, 1.0)
    w = max(x2 - x1, 1.0)
    nan = math.nan
    l_sho, r_sho = _pt(kps, L_SHO), _pt(kps, R_SHO)
    l_hip, r_hip = _pt(kps, L_HIP), _pt(kps, R_HIP)
    hip = _mid(l_hip, r_hip)
    sho = _mid(l_sho, r_sho)
    wrists = [_pt(kps, L_WRI), _pt(kps, R_WRI)]
    elbows = [_pt(kps, L_ELB), _pt(kps, R_ELB)]
    seen = [wr for wr in wrists if wr is not None]

    if hip is not None and seen:
        dists = [float(np.hypot(*(wr - hip))) / h for wr in seen]
        hip_min, hip_max = min(dists), max(dists)
    else:
        hip_min = hip_max = nan
    above = float(any(wr[1] < sho[1] for wr in seen)) if (sho is not None and seen) else nan

    torso_pts = [p for p in (l_sho, r_sho, l_hip, r_hip) if p is not None]
    if len(torso_pts) >= 3 and seen:
        tp = np.array(torso_pts)
        tx1, ty1 = tp.min(axis=0) - 0.04 * h
        tx2, ty2 = tp.max(axis=0) + 0.04 * h
        in_torso = float(any(tx1 <= wr[0] <= tx2 and ty1 <= wr[1] <= ty2 for wr in seen))
    else:
        in_torso = nan

    body_visible = len(torso_pts) >= 3
    hidden = float(body_visible and len(seen) < 2) if body_visible else nan

    bent = []
    for sho_pt, elb, wr in ((l_sho, elbows[0], wrists[0]), (r_sho, elbows[1], wrists[1])):
        if sho_pt is not None and elb is not None and wr is not None:
            a, b = sho_pt - elb, wr - elb
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
            bent.append(cos)  # +1 = fully folded, -1 = straight arm
    elbow_bent = max(bent) if bent else nan

    knee = _mid(_pt(kps, L_KNEE), _pt(kps, R_KNEE))
    crouch = float((knee[1] - hip[1]) / h) if (knee is not None and hip is not None) else nan
    sw = float(np.hypot(*(l_sho - r_sho)) / h) if (l_sho is not None and r_sho is not None) else nan
    nose = _pt(kps, NOSE)
    if nose is not None and sho is not None and l_sho is not None and r_sho is not None:
        facing = float((nose[0] - sho[0]) / (abs(l_sho[0] - r_sho[0]) + 0.05 * h))
    else:
        facing = nan

    hand_bag = obs.hand_bag_distance if math.isfinite(obs.hand_bag_distance) else 3.0
    return np.array(
        [
            hip_min,
            hip_max,
            above,
            in_torso,
            len(seen) / 2,
            hidden,
            elbow_bent,
            crouch,
            h / w,
            sw,
            facing,
            float(nose is not None),
            min(hand_bag, 3.0),
            float(obs.bag_distance <= 1.0),
        ],
        dtype=np.float64,
    )


# --- track / window features ----------------------------------------------------------


def _stats(values: np.ndarray) -> list[float]:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return [math.nan] * 4
    return [float(v.mean()), float(v.min()), float(v.max()), float(v.std())]


def _feature_names() -> list[str]:
    names = [f"{s}_{agg}" for s in FRAME_SIGNALS for agg in ("mean", "min", "max", "std")]
    names += [
        "wrist_speed_mean",
        "wrist_speed_max",
        "body_speed_mean",
        "body_speed_max",
        "reach_then_hide",  # reach out/up followed within 2 s by hand at torso or hidden
        "hide_events",  # visible → hidden wrist transitions per second
        "torso_events",  # hand entering the torso box per second
        "facing_changes",  # head turning left/right per second (looking around)
    ]
    return names


FEATURE_NAMES = _feature_names()
N_FEATURES = len(FEATURE_NAMES)


def track_features(observations: list[Observation]) -> np.ndarray:
    """Summary features for one person over one window."""
    obs = sorted(observations, key=lambda o: o.time)
    signals = np.array([frame_signals(o) for o in obs])
    feats: list[float] = []
    for col in range(signals.shape[1]):
        feats += _stats(signals[:, col])

    times = np.array([o.time for o in obs])
    heights = np.array([max(o.box[3] - o.box[1], 1.0) for o in obs])
    centres = np.array([((o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2) for o in obs])
    dt = np.diff(times)
    dt[dt <= 0] = 1e-3
    body_speed = np.hypot(*np.diff(centres, axis=0).T) / heights[1:] / dt if len(obs) > 1 else np.array([])

    wrist_speed = []
    for i in range(1, len(obs)):
        for k in (L_WRI, R_WRI):
            a, b = obs[i - 1].keypoints[k], obs[i].keypoints[k]
            if a[2] >= KP_CONF and b[2] >= KP_CONF:
                wrist_speed.append(float(np.hypot(*(b[:2] - a[:2]))) / heights[i] / dt[i - 1])
    ws = np.array(wrist_speed)

    reach = (signals[:, 1] > 0.55) | (signals[:, 2] == 1)
    conceal = (signals[:, 3] == 1) | (signals[:, 5] == 1)
    reach_then_hide = 0.0
    for i in np.flatnonzero(reach):
        later = (times > times[i]) & (times <= times[i] + 2.0)
        if np.any(conceal & later):
            reach_then_hide += 1
    duration = max(float(times[-1] - times[0]), 1e-3)
    vis = signals[:, 4]
    hide_events = float(np.sum((vis[:-1] == 1) & (vis[1:] < 1))) if len(vis) > 1 else 0.0
    tor = np.nan_to_num(signals[:, 3])
    torso_events = float(np.sum((tor[:-1] == 0) & (tor[1:] == 1))) if len(tor) > 1 else 0.0
    facing = signals[:, 10]
    facing = np.sign(facing[~np.isnan(facing)].round(1))
    facing_changes = float(np.sum(facing[1:] != facing[:-1])) if facing.size > 1 else 0.0

    feats += [
        float(ws.mean()) if ws.size else math.nan,
        float(ws.max()) if ws.size else math.nan,
        float(body_speed.mean()) if body_speed.size else math.nan,
        float(body_speed.max()) if body_speed.size else math.nan,
        reach_then_hide / duration,
        hide_events / duration,
        torso_events / duration,
        facing_changes / duration,
    ]
    return np.array(feats, dtype=np.float64)


@dataclass
class TrackWindow:
    track_id: str
    start: float
    end: float
    features: np.ndarray
    peak_time: float  # moment the person was most "reach-then-hide" like, for the evidence frame


def windows(recorder: ObservationRecorder, duration: float | None = None) -> list[tuple[float, float]]:
    times = [o.time for o in recorder.observations]
    if not times:
        return []
    end = duration if duration else max(times)
    if end <= WINDOW_SECONDS * 1.5:
        return [(0.0, max(end, 1e-3) + 1e-6)]
    starts = np.arange(0.0, max(end - WINDOW_SECONDS, 0) + 1e-6, WINDOW_STRIDE_SECONDS)
    return [(float(s), float(s) + WINDOW_SECONDS) for s in starts]


def track_windows(recorder: ObservationRecorder, duration: float | None = None) -> list[TrackWindow]:
    """Per-person features for every window that has at least ``MIN_TRACK_SECONDS`` of that person."""
    by_track: dict[str, list[Observation]] = defaultdict(list)
    for o in recorder.observations:
        by_track[o.track_id].append(o)
    result = []
    for start, end in windows(recorder, duration):
        for track_id, obs in by_track.items():
            inside = [o for o in obs if start <= o.time < end]
            if len(inside) < 3 or inside[-1].time - inside[0].time < MIN_TRACK_SECONDS:
                continue
            signals = np.array([frame_signals(o) for o in inside])
            conceal = np.nan_to_num(signals[:, 3]) + np.nan_to_num(signals[:, 5])
            peak = inside[int(np.argmax(conceal))].time
            result.append(TrackWindow(track_id, start, end, track_features(inside), peak))
    return result


def window_matrix(track_wins: list[TrackWindow]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Clip-style features per window: the maximum of each feature over the people in it.

    Used for training on clip-level labels (a clip is suspicious if anyone in it is).
    """
    groups: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for tw in track_wins:
        groups[(tw.start, tw.end)].append(tw.features)
    keys = sorted(groups)
    if not keys:
        return np.empty((0, N_FEATURES)), []
    rows = []
    for key in keys:
        stack = np.vstack(groups[key])
        with np.errstate(all="ignore"):
            agg = np.where(np.all(np.isnan(stack), axis=0), np.nan, np.nanmax(stack, axis=0))
        rows.append(agg)
    return np.vstack(rows), keys


# Track length and people count are deliberately left out: in the training data they reveal which
# source a clip came from rather than what anyone did.
WINDOW_FEATURE_NAMES = list(FEATURE_NAMES)
