from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from .config import DATASETS_DIR, REPORTS_DIR, ROOT_DIR
from .project import load_project
from .video import safe_filename

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["person", "held_product", "bag", "basket_cart"]


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def dataset_dir(name: str) -> Path:
    return DATASETS_DIR / safe_filename(name)


def dataset_yaml_path(name: str) -> Path:
    return dataset_dir(name) / "data.yaml"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dataset_yaml(base_dir: Path, classes: list[str]) -> Path:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    content = f"""# Ultralytics YOLO detection dataset
path: {base_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names}
"""
    path = base_dir / "data.yaml"
    write_text(path, content)
    return path


def init_dataset(name: str, classes: list[str]) -> dict[str, Any]:
    base = dataset_dir(name)
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    (base / "raw_frames").mkdir(parents=True, exist_ok=True)
    yaml_path = write_dataset_yaml(base, classes)
    metadata = {
        "schema_version": 1,
        "dataset": safe_filename(name),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": classes,
        "data_yaml": str(yaml_path),
        "note": "Use raw_frames for annotation, then place/export labeled data into images/labels train/val/test.",
    }
    write_json(base / "dataset.json", metadata)
    return metadata


def extract_frames_from_project(
    project_id: str,
    dataset_name: str,
    every_seconds: float,
    limit: int | None,
    start_seconds: float,
    end_seconds: float | None,
) -> dict[str, Any]:
    project = load_project(project_id)
    video_path = Path(project["video"]["path"])
    base = dataset_dir(dataset_name)
    output_dir = base / "raw_frames" / project_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0.0
    if end_seconds is None or end_seconds <= 0:
        end_seconds = duration
    end_seconds = min(end_seconds, duration) if duration else end_seconds
    start_seconds = max(0.0, start_seconds)
    every_seconds = max(0.05, every_seconds)

    saved = 0
    current = start_seconds
    while current <= end_seconds:
        if limit is not None and saved >= limit:
            break
        frame_index = int(current * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        filename = f"{safe_filename(project_id)}_{saved:06d}_{current:.2f}s.jpg"
        cv2.imwrite(str(output_dir / filename), frame)
        saved += 1
        current += every_seconds
    cap.release()

    manifest = {
        "project_id": project_id,
        "source_video": str(video_path),
        "output_dir": str(output_dir),
        "saved_frames": saved,
        "every_seconds": every_seconds,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }
    write_json(output_dir / "frames_manifest.json", manifest)
    return manifest


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def label_path_for_image(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    relative = image_path.relative_to(images_root)
    return (labels_root / relative).with_suffix(".txt")


def parse_classes_from_yaml(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    in_names = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line:
                continue
            if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
                break
            if ":" in line:
                _, value = line.split(":", 1)
                names.append(value.strip().strip("'\""))
    return names


def check_dataset(name: str) -> dict[str, Any]:
    base = dataset_dir(name)
    yaml_path = base / "data.yaml"
    classes = parse_classes_from_yaml(yaml_path)
    class_count = len(classes)
    splits: dict[str, Any] = {}
    errors: list[str] = []

    for split in ("train", "val", "test"):
        images_root = base / "images" / split
        labels_root = base / "labels" / split
        images = list_images(images_root)
        missing_labels = []
        empty_labels = 0
        boxes = 0
        invalid_lines = []

        for image_path in images:
            label_path = label_path_for_image(image_path, images_root, labels_root)
            if not label_path.exists():
                missing_labels.append(str(image_path.relative_to(base)))
                continue
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_labels += 1
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_lines.append(f"{label_path}:{line_no}: expected 5 columns")
                    continue
                try:
                    cls = int(float(parts[0]))
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    invalid_lines.append(f"{label_path}:{line_no}: non-numeric value")
                    continue
                if class_count and not 0 <= cls < class_count:
                    invalid_lines.append(f"{label_path}:{line_no}: class {cls} outside 0..{class_count - 1}")
                if any(v < 0 or v > 1 for v in values):
                    invalid_lines.append(f"{label_path}:{line_no}: box values must be normalized 0..1")
                boxes += 1

        splits[split] = {
            "images": len(images),
            "missing_labels": len(missing_labels),
            "empty_labels": empty_labels,
            "boxes": boxes,
            "invalid_lines": len(invalid_lines),
            "missing_label_examples": missing_labels[:10],
            "invalid_line_examples": invalid_lines[:10],
        }
        if split in {"train", "val"} and not images:
            errors.append(f"{split} split has no images.")
        if missing_labels:
            errors.append(f"{split} split has images without matching label files.")
        if invalid_lines:
            errors.append(f"{split} split has invalid labels.")

    report = {
        "schema_version": 1,
        "dataset": safe_filename(name),
        "data_yaml": str(yaml_path),
        "classes": classes,
        "splits": splits,
        "ready_for_training": not errors and splits["train"]["images"] > 0 and splits["val"]["images"] > 0,
        "errors": errors,
    }
    write_json(base / "dataset_check.json", report)
    return report


def split_labeled_folder(
    source_images: Path,
    source_labels: Path,
    dataset_name: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    copy: bool,
) -> dict[str, Any]:
    base = dataset_dir(dataset_name)
    images = list_images(source_images)
    rng = random.Random(seed)
    rng.shuffle(images)
    train_cut = int(len(images) * train_ratio)
    val_cut = train_cut + int(len(images) * val_ratio)
    split_map = {
        "train": images[:train_cut],
        "val": images[train_cut:val_cut],
        "test": images[val_cut:],
    }
    action = shutil.copy2 if copy else shutil.move
    moved: dict[str, int] = {}
    for split, split_images in split_map.items():
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in split_images:
            label_path = source_labels / image_path.with_suffix(".txt").name
            dest_image = base / "images" / split / image_path.name
            dest_label = base / "labels" / split / label_path.name
            action(image_path, dest_image)
            if label_path.exists():
                action(label_path, dest_label)
            count += 1
        moved[split] = count
    return {"dataset": safe_filename(dataset_name), "images": len(images), "split_counts": moved}


def train_model(
    dataset_name: str,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    run_name: str | None,
) -> dict[str, Any]:
    check = check_dataset(dataset_name)
    if not check["ready_for_training"]:
        raise RuntimeError("Dataset is not ready for training. Run check-dataset and fix errors first.")
    data_yaml = dataset_yaml_path(dataset_name)
    run_name = run_name or f"{safe_filename(dataset_name)}_{utc_timestamp()}"
    yolo = YOLO(model)
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(ROOT_DIR / "runs" / "train"),
        "name": run_name,
    }
    if device:
        kwargs["device"] = device
    results = yolo.train(**kwargs)
    save_dir = getattr(results, "save_dir", ROOT_DIR / "runs" / "train" / run_name)
    report = {
        "schema_version": 1,
        "dataset": safe_filename(dataset_name),
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device or "auto",
        "run_name": run_name,
        "save_dir": str(save_dir),
        "best_weights": str(Path(save_dir) / "weights" / "best.pt"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(REPORTS_DIR / "training" / f"{run_name}.json", report)
    return report


def evaluate_model(
    dataset_name: str,
    weights: str,
    imgsz: int,
    batch: int,
    split: str,
    device: str | None,
    run_name: str | None,
) -> dict[str, Any]:
    data_yaml = dataset_yaml_path(dataset_name)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")
    run_name = run_name or f"eval_{safe_filename(dataset_name)}_{utc_timestamp()}"
    model = YOLO(weights)
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "split": split,
        "project": str(ROOT_DIR / "runs" / "val"),
        "name": run_name,
    }
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    box = getattr(metrics, "box", None)
    report = {
        "schema_version": 1,
        "dataset": safe_filename(dataset_name),
        "weights": weights,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device or "auto",
        "run_name": run_name,
        "metrics": {
            "map50_95": float(getattr(box, "map", 0.0)) if box else None,
            "map50": float(getattr(box, "map50", 0.0)) if box else None,
            "map75": float(getattr(box, "map75", 0.0)) if box else None,
            "mean_precision": float(np_mean(getattr(box, "p", []))) if box else None,
            "mean_recall": float(np_mean(getattr(box, "r", []))) if box else None,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(REPORTS_DIR / "evaluation" / f"{run_name}.json", report)
    return report


def np_mean(values: Any) -> float:
    try:
        import numpy as np

        arr = np.array(values, dtype=float)
        return float(arr.mean()) if arr.size else 0.0
    except Exception:
        return 0.0


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail theft detector training/evaluation tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-dataset", help="Create a YOLO dataset skeleton.")
    p_init.add_argument("--name", required=True)
    p_init.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)

    p_extract = sub.add_parser("extract-frames", help="Extract raw frames from an uploaded project for annotation.")
    p_extract.add_argument("--project-id", required=True)
    p_extract.add_argument("--dataset", required=True)
    p_extract.add_argument("--every-seconds", type=float, default=0.5)
    p_extract.add_argument("--limit", type=int)
    p_extract.add_argument("--start-seconds", type=float, default=0.0)
    p_extract.add_argument("--end-seconds", type=float)

    p_split = sub.add_parser("split-labeled", help="Split a flat YOLO-labeled folder into train/val/test.")
    p_split.add_argument("--images", required=True, type=Path)
    p_split.add_argument("--labels", required=True, type=Path)
    p_split.add_argument("--dataset", required=True)
    p_split.add_argument("--train-ratio", type=float, default=0.7)
    p_split.add_argument("--val-ratio", type=float, default=0.2)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--move", action="store_true", help="Move files instead of copying them.")

    p_check = sub.add_parser("check-dataset", help="Validate YOLO labels and split readiness.")
    p_check.add_argument("--dataset", required=True)

    p_train = sub.add_parser("train", help="Train a YOLO detection model.")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--model", default="yolo11n.pt")
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--device")
    p_train.add_argument("--run-name")

    p_eval = sub.add_parser("evaluate", help="Evaluate trained weights and write a metrics report.")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--imgsz", type=int, default=640)
    p_eval.add_argument("--batch", type=int, default=8)
    p_eval.add_argument("--split", choices=["train", "val", "test"], default="val")
    p_eval.add_argument("--device")
    p_eval.add_argument("--run-name")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init-dataset":
        print_json(init_dataset(args.name, args.classes))
    elif args.command == "extract-frames":
        print_json(
            extract_frames_from_project(
                args.project_id,
                args.dataset,
                args.every_seconds,
                args.limit,
                args.start_seconds,
                args.end_seconds,
            )
        )
    elif args.command == "split-labeled":
        print_json(
            split_labeled_folder(
                args.images,
                args.labels,
                args.dataset,
                args.train_ratio,
                args.val_ratio,
                args.seed,
                copy=not args.move,
            )
        )
    elif args.command == "check-dataset":
        print_json(check_dataset(args.dataset))
    elif args.command == "train":
        print_json(train_model(args.dataset, args.model, args.epochs, args.imgsz, args.batch, args.device, args.run_name))
    elif args.command == "evaluate":
        print_json(evaluate_model(args.dataset, args.weights, args.imgsz, args.batch, args.split, args.device, args.run_name))


if __name__ == "__main__":
    main()
