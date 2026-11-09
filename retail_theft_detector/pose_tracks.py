"""Per-person pose observations recorded while a video is analysed.

The detector and the training tools use the same recorder, so the behaviour models see input
computed exactly as it was at training time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

KP_CONF = 0.30

# COCO keypoint indexes
NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNEE, R_KNEE = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14


@dataclass
class Observation:
    """One tracked person in one analysed frame."""

    time: float
    track_id: str
    box: tuple[float, float, float, float]
    keypoints: np.ndarray  # (17, 3): x, y, confidence; NaN-free, zeros when missing
    hand_bag_distance: float  # nearest wrist → bag centre, in person heights (inf if no bag)
    bag_distance: float  # person centre → nearest bag centre, in person heights (inf if no bag)


@dataclass
class ObservationRecorder:
    """Collects per-person observations while a video is analysed."""

    observations: list[Observation] = field(default_factory=list)

    def add(
        self,
        time: float,
        track_id: str,
        box: Iterable[float],
        keypoints: list[tuple[float, float, float]] | None,
        bag_boxes: Iterable[Iterable[float]] = (),
    ) -> None:
        x1, y1, x2, y2 = (float(v) for v in box)
        height = max(y2 - y1, 1.0)
        kps = np.zeros((17, 3), dtype=np.float32)
        if keypoints:
            arr = np.asarray(keypoints, dtype=np.float32)[:17]
            kps[: len(arr)] = arr
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        wrists = [kps[i, :2] for i in (L_WRI, R_WRI) if kps[i, 2] >= KP_CONF]
        hand_bag = bag_dist = math.inf
        for bag in bag_boxes:
            bx1, by1, bx2, by2 = (float(v) for v in bag)
            bc = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
            bag_dist = min(bag_dist, float(np.hypot(*(bc - centre))) / height)
            for wrist in wrists:
                hand_bag = min(hand_bag, float(np.hypot(*(bc - wrist))) / height)
        self.observations.append(Observation(time, str(track_id), (x1, y1, x2, y2), kps, hand_bag, bag_dist))

    # --- persistence (training cache) -------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray]:
        obs = self.observations
        return {
            "time": np.array([o.time for o in obs], dtype=np.float32),
            "track": np.array([o.track_id for o in obs]),
            "box": np.array([o.box for o in obs], dtype=np.float32).reshape(-1, 4),
            "keypoints": np.array([o.keypoints for o in obs], dtype=np.float32).reshape(-1, 17, 3),
            "hand_bag": np.array([o.hand_bag_distance for o in obs], dtype=np.float32),
            "bag": np.array([o.bag_distance for o in obs], dtype=np.float32),
        }

    @classmethod
    def from_arrays(cls, data: Any) -> ObservationRecorder:
        rec = cls()
        for i in range(len(data["time"])):
            rec.observations.append(
                Observation(
                    float(data["time"][i]),
                    str(data["track"][i]),
                    tuple(float(v) for v in data["box"][i]),
                    np.asarray(data["keypoints"][i], dtype=np.float32),
                    float(data["hand_bag"][i]),
                    float(data["bag"][i]),
                )
            )
        return rec
