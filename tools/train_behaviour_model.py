"""Train the behaviour model (skeleton network + movement features) from labelled clips.

Expected layout (one clip per file, the folder is the label)::

    <clips_dir>/suspicious_videos/*.mp4
    <clips_dir>/normal_videos/*.mp4

Step 1 runs person tracking and pose estimation once per clip (exactly as the app does) and caches
the observations under ``data/behaviour_cache/``::

    python tools/train_behaviour_model.py extract --clips <clips_dir> --workers 4

Step 2 cross-validates the skeleton network, the movement-feature model and their combination, then
writes ``models/skeleton_model.pt``, ``models/movement_features.joblib`` and the model card
``models/behaviour_model.json`` with the evaluation results::

    python tools/train_behaviour_model.py train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retail_theft_detector import behaviour, skeleton_model  # noqa: E402
from retail_theft_detector.config import DATA_DIR  # noqa: E402
from retail_theft_detector.pose_tracks import ObservationRecorder  # noqa: E402

CACHE_DIR = DATA_DIR / "behaviour_cache"
LABELS = {"suspicious_videos": 1, "normal_videos": 0}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
WINDOWS_PER_CLIP = 12  # person-windows sampled from each clip per training step


# --- step 1: extraction ---------------------------------------------------------------


def record_clip(video_path: str, out_path: str, max_seconds: float, threads: int) -> tuple[str, int, float]:
    """Run tracking + pose on one clip exactly as the detector does and cache the observations."""
    import cv2
    import torch
    from ultralytics import YOLO

    from retail_theft_detector.detector import (
        DetectorSettings,
        attach_pose,
        is_bag_or_container,
        parse_detections,
        parse_poses,
    )

    torch.set_num_threads(threads)
    settings = DetectorSettings()
    object_model = YOLO(settings.object_model)
    pose_model = YOLO(settings.pose_model)
    recorder = ObservationRecorder()

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    stride = max(1, int(round(fps / settings.max_analysis_fps)))
    index = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = index / fps
        if max_seconds and t > max_seconds:
            break
        if index % stride == 0:
            res = object_model.track(
                source=frame, persist=True, tracker=settings.tracker, conf=settings.object_confidence, verbose=False
            )
            dets = parse_detections(res[0], object_model) if res else []
            people = [d for d in dets if d.cls_id in settings.person_class_ids]
            bags = [d.box for d in dets if is_bag_or_container(d, settings)]
            if people:
                pres = pose_model.predict(source=frame, conf=settings.pose_confidence, verbose=False)
                poses = parse_poses(pres[0]) if pres else []
                for n, person in enumerate(people, start=1):
                    pose = attach_pose(person, poses)
                    recorder.add(t, person.track_id or f"untracked_{n}", person.box, pose.keypoints if pose else None, bags)
        index += 1
    cap.release()
    np.savez_compressed(out_path, **recorder.to_arrays(), duration=np.float32(index / fps))
    return video_path, len(recorder.observations), time.perf_counter() - started


def cmd_extract(args: argparse.Namespace) -> None:
    jobs = []
    for folder in LABELS:
        out_dir = CACHE_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        for video in sorted((Path(args.clips) / folder).iterdir()):
            if video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            out = out_dir / f"{video.stem}.npz"
            if out.exists() and not args.force:
                continue
            jobs.append((str(video), str(out)))
    print(f"{len(jobs)} clips to process with {args.workers} workers", flush=True)
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(record_clip, v, o, args.max_seconds, threads) for v, o in jobs]
        for future in as_completed(futures):
            done += 1
            try:
                video, n_obs, secs = future.result()
                print(f"[{done}/{len(jobs)}] {Path(video).name}: {n_obs} observations in {secs:.1f}s", flush=True)
            except Exception as exc:  # keep going; a bad clip should not stop the run
                print(f"[{done}/{len(jobs)}] FAILED: {exc}", flush=True)


# --- shared evaluation helpers --------------------------------------------------------


@dataclass
class Clip:
    name: str
    label: int
    recorder: ObservationRecorder
    duration: float


def load_clips() -> list[Clip]:
    clips = []
    for folder, label in LABELS.items():
        for path in sorted((CACHE_DIR / folder).glob("*.npz")):
            data = np.load(path)
            clips.append(Clip(f"{folder}/{path.stem}", label, ObservationRecorder.from_arrays(data), float(data["duration"])))
    return clips


def metrics_at(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "false_alarm_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "accuracy": round((tp + tn) / len(y), 3),
    }


def threshold_for_recall(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Highest threshold that still catches ``target`` of the suspicious clips."""
    for threshold in sorted(np.unique(scores), reverse=True):
        if metrics_at(scores, y, threshold)["recall"] >= target:
            return float(threshold)
    return float(scores.min())


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    return max(np.unique(scores), key=lambda t: metrics_at(scores, y, t)["f1"])


def folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    from sklearn.model_selection import StratifiedKFold

    return [test for _, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y)]


# --- skeleton model -------------------------------------------------------------------


def train_skeleton(
    windows: list[np.ndarray], labels: np.ndarray, epochs: int, seed: int, device: str, width: int
):
    """Multiple-instance training: a clip's logit is the max over all its person-windows."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    net = skeleton_model.SkeletonNet(width=width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * max(1, len(windows) // 16 + 1))
    pos = max(1, int(labels.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(labels) - pos) / pos, device=device))
    usable = [i for i, w in enumerate(windows) if len(w)]
    for _ in range(epochs):
        net.train()
        order = rng.permutation(usable)
        for b in range(0, len(order), 16):
            batch = order[b : b + 16]
            parts, owners = [], []
            for k, clip in enumerate(batch):
                w = windows[clip]
                pick = rng.choice(len(w), size=min(len(w), WINDOWS_PER_CLIP), replace=False)
                parts.append(w[pick])
                owners += [k] * len(pick)
            x = skeleton_model.augment(torch.from_numpy(np.concatenate(parts)), generator).to(device)
            logits = net(x)
            owner = torch.tensor(owners, device=device)
            clip_logits = torch.stack([logits[owner == k].max() for k in range(len(batch))])
            target = torch.tensor(labels[batch], dtype=torch.float32, device=device)
            loss = loss_fn(clip_logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.cpu().eval()


def score_clips(nets: list, windows: list[np.ndarray]) -> np.ndarray:
    import torch

    scores = np.zeros(len(windows))
    with torch.no_grad():
        for i, w in enumerate(windows):
            if len(w):
                x = torch.from_numpy(w)
                scores[i] = float(np.mean([torch.sigmoid(n(x)).max().item() for n in nets]))
    return scores


def cv_skeleton(clips: list[Clip], args: argparse.Namespace) -> tuple[np.ndarray, list]:
    windows = [skeleton_model.person_windows(c.recorder)[0] for c in clips]
    y = np.array([c.label for c in clips])
    print(f"skeleton: {sum(len(w) for w in windows)} person-windows; {sum(not len(w) for w in windows)} clips with none")
    oof = np.zeros((args.repeats, len(clips)))
    members = []
    for r in range(args.repeats):
        for f, test in enumerate(folds(y, 5, r)):
            train = np.setdiff1d(np.arange(len(clips)), test)
            started = time.perf_counter()
            net = train_skeleton([windows[i] for i in train], y[train], args.epochs, 100 * r + f, args.device, args.width)
            oof[r, test] = score_clips([net], [windows[i] for i in test])
            if r == 0:
                members.append(net)
            print(f"  repeat {r + 1} fold {f + 1}: {time.perf_counter() - started:.0f}s", flush=True)
    return oof.mean(axis=0), members


# --- movement-feature model -----------------------------------------------------------


def movement_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=8,
        l2_regularization=1.0, class_weight="balanced", random_state=seed,
    )  # fmt: skip


def movement_rows(clips: list[Clip]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window feature rows, their clip labels and the clip each row came from."""
    rows, groups = [], []
    for i, c in enumerate(clips):
        matrix, keys = behaviour.window_matrix(behaviour.track_windows(c.recorder, c.duration))
        if not keys:
            matrix = np.full((1, behaviour.N_FEATURES), np.nan)
        rows.append(matrix)
        groups += [i] * len(matrix)
    groups = np.array(groups)
    y_clip = np.array([c.label for c in clips])
    return np.vstack(rows), y_clip[groups], groups


def cv_movement(clips: list[Clip], repeats: int) -> np.ndarray:
    X, y, groups = movement_rows(clips)
    y_clip = np.array([c.label for c in clips])
    oof = np.zeros((repeats, len(clips)))
    for r in range(repeats):
        for test in folds(y_clip, 5, r):
            train_rows = ~np.isin(groups, test)
            model = movement_model(r).fit(X[train_rows], y[train_rows])
            proba = model.predict_proba(X[~train_rows])[:, 1]
            for clip in test:
                oof[r, clip] = proba[groups[~train_rows] == clip].max()
    return oof.mean(axis=0)


# --- step 2: training -----------------------------------------------------------------


def summarise(scores: np.ndarray, y: np.ndarray, target_recall: float) -> dict:
    from sklearn.metrics import roc_auc_score

    return {
        "roc_auc": round(float(roc_auc_score(y, scores)), 3),
        "at_target_recall": metrics_at(scores, y, threshold_for_recall(scores, y, target_recall)),
        "at_best_f1": metrics_at(scores, y, best_f1_threshold(scores, y)),
    }


def cmd_train(args: argparse.Namespace) -> None:
    import joblib
    import sklearn
    import torch

    clips = load_clips()
    y = np.array([c.label for c in clips])
    print(f"{len(clips)} clips: {int(y.sum())} suspicious, {int((1 - y).sum())} normal")

    movement = cv_movement(clips, args.repeats)
    skeleton, members = cv_skeleton(clips, args)
    combined = (movement + skeleton) / 2
    np.savez(
        CACHE_DIR / "oof_scores.npz",
        names=np.array([c.name for c in clips]), y=y, movement=movement, skeleton=skeleton, combined=combined,
    )  # fmt: skip

    report = {
        "combined (shipped)": summarise(combined, y, args.target_recall),
        "skeleton_network_only": summarise(skeleton, y, args.target_recall),
        "movement_features_only": summarise(movement, y, args.target_recall),
    }
    for name, values in report.items():
        print(f"{name:24s} {json.dumps(values)}")

    X, y_rows, _ = movement_rows(clips)
    final_movement = movement_model(0).fit(X, y_rows)
    out = skeleton_model.WEIGHTS_PATH.parent
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"width": args.width, "members": [m.state_dict() for m in members]}, skeleton_model.WEIGHTS_PATH)
    joblib.dump(final_movement, skeleton_model.MOVEMENT_PATH, compress=3)

    threshold = threshold_for_recall(combined, y, args.target_recall)
    card = {
        "name": "behaviour-concealment-model",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "components": {
            "skeleton_network": {
                "file": skeleton_model.WEIGHTS_PATH.name,
                "architecture": f"ST-GCN, 4 blocks, width {args.width}, ensemble of {len(members)} fold models",
                "input": {
                    "fps": skeleton_model.FPS,
                    "window_seconds": skeleton_model.WINDOW_SECONDS,
                    "stride_seconds": skeleton_model.STRIDE_SECONDS,
                    "channels": list(skeleton_model.CHANNELS),
                },
                "training": {"epochs": args.epochs, "loss": "multiple-instance BCE (clip = max over person-windows)"},
            },
            "movement_features": {
                "file": skeleton_model.MOVEMENT_PATH.name,
                "algorithm": "HistGradientBoostingClassifier",
                "sklearn_version": sklearn.__version__,
                "window_seconds": behaviour.WINDOW_SECONDS,
                "features": behaviour.WINDOW_FEATURE_NAMES,
            },
        },
        "combination": "mean of the two probabilities per person",
        "threshold": round(threshold, 4),
        "threshold_rule": f"highest threshold that still catches {args.target_recall:.0%} of suspicious clips",
        "training_data": {"clips": len(clips), "suspicious": int(y.sum()), "normal": int((1 - y).sum())},
        "evaluation": {
            "method": f"5-fold stratified cross-validation repeated {args.repeats}x; every clip is scored only "
            "by models that never saw it during training; a clip's score is its most suspicious person-window",
            **report,
        },
    }
    skeleton_model.CARD_PATH.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    for path in (skeleton_model.WEIGHTS_PATH, skeleton_model.MOVEMENT_PATH, skeleton_model.CARD_PATH):
        print(f"Saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="Cache tracking + pose observations for every labelled clip")
    ex.add_argument("--clips", required=True, help="Folder containing suspicious_videos/ and normal_videos/")
    ex.add_argument("--workers", type=int, default=2)
    ex.add_argument("--max-seconds", type=float, default=0.0, help="Only analyse the first N seconds (0 = all)")
    ex.add_argument("--force", action="store_true", help="Re-extract clips that are already cached")
    tr = sub.add_parser("train", help="Cross-validate and train the behaviour model from the cache")
    tr.add_argument("--target-recall", type=float, default=0.8)
    tr.add_argument("--repeats", type=int, default=1)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--width", type=int, default=32)
    tr.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    args = parser.parse_args()
    {"extract": cmd_extract, "train": cmd_train}[args.command](args)


if __name__ == "__main__":
    main()
