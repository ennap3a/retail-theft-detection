from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

from retail_theft_detector.config import (  # noqa: E402
    ALLOWED_VIDEO_EXTENSIONS,
    NORMAL_TEST_VIDEOS_DIR,
    REPORTS_DIR,
    SUSPICIOUS_TEST_VIDEOS_DIR,
)
from retail_theft_detector.silentvision_action_model import (  # noqa: E402
    SilentVisionActionModel,
    SilentVisionUnavailable,
)

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "external_models"
    / "Shoplifting-Detection-main"
    / "CODE"
    / "weight_steals"
    / "GATE_FLOW_SLOW_FAST_RGB_ONLY"
    / "weights_at_epoch_8_75___good.h5"
)


def iter_videos(folder: Path):
    if not folder.exists():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            yield path


def classify_result(truth: str, model_alerted: bool) -> str:
    if truth == "suspicious":
        return "TP" if model_alerted else "FN"
    return "FP" if model_alerted else "TN"


def safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    valid_rows = [row for row in rows if not row.get("error")]
    tp = sum(1 for row in valid_rows if row["result_type"] == "TP")
    fp = sum(1 for row in valid_rows if row["result_type"] == "FP")
    fn = sum(1 for row in valid_rows if row["result_type"] == "FN")
    tn = sum(1 for row in valid_rows if row["result_type"] == "TN")
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_alarm_rate = safe_div(fp, fp + tn)
    miss_rate = safe_div(fn, tp + fn)
    avg_runtime = safe_div(
        sum(float(row.get("runtime_seconds") or 0) for row in valid_rows),
        len(valid_rows),
    )
    avg_processing_fps = safe_div(
        sum(float(row.get("processing_fps") or 0) for row in valid_rows),
        len(valid_rows),
    )
    return {
        "videos_evaluated": len(valid_rows),
        "videos_with_errors": len(rows) - len(valid_rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "avg_runtime_seconds": avg_runtime,
        "avg_processing_fps": avg_processing_fps,
    }


def write_outputs(output_dir: Path, rows: list[dict[str, object]], metrics: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "silentvision_predictions.csv"
    json_path = output_dir / "silentvision_metrics.json"
    details_path = output_dir / "silentvision_clip_predictions.json"

    fieldnames = [
        "video_path",
        "truth",
        "model_alerted",
        "predicted_label",
        "result_type",
        "suspicion_score",
        "suspicious_windows",
        "total_windows",
        "processed_frames",
        "runtime_seconds",
        "processing_fps",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {csv_path}")
    print(f"Wrote metrics: {json_path}")
    print(f"Wrote clip details: {details_path}")


def build_dataset(
    normal_dir: Path,
    suspicious_dir: Path,
    limit: int | None,
    per_class_limit: int | None,
) -> list[tuple[Path, str]]:
    normal_items = [(path, "normal") for path in iter_videos(normal_dir)]
    suspicious_items = [(path, "suspicious") for path in iter_videos(suspicious_dir)]
    if per_class_limit is not None:
        normal_items = normal_items[:per_class_limit]
        suspicious_items = suspicious_items[:per_class_limit]
    items = normal_items + suspicious_items
    if limit is not None:
        return items[:limit]
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the external SilentVision shoplifting model locally.")
    parser.add_argument("--normal-dir", type=Path, default=NORMAL_TEST_VIDEOS_DIR)
    parser.add_argument("--suspicious-dir", type=Path, default=SUSPICIOUS_TEST_VIDEOS_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-suspicious-windows", type=int, default=2)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-windows", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Missing weights file: {args.weights}")
        return 2

    dataset = build_dataset(args.normal_dir, args.suspicious_dir, args.limit, args.per_class_limit)
    print(f"Normal folder: {args.normal_dir}")
    print(f"Suspicious folder: {args.suspicious_dir}")
    print(f"Weights: {args.weights}")
    print(f"Videos found: {len(dataset)}")

    try:
        model = SilentVisionActionModel(args.weights)
        if args.check_only:
            model.load()
            print("SilentVision model loaded successfully.")
            return 0
    except SilentVisionUnavailable as exc:
        print(str(exc))
        print("Recommended path: run this evaluator in Google Colab, where TensorFlow is easier to use.")
        return 3

    if args.check_only:
        print("Dependency check passed.")
        return 0
    if not dataset:
        print("No videos found. Put test videos into the normal_videos and suspicious_videos folders first.")
        return 1

    rows: list[dict[str, object]] = []
    for index, (video_path, truth) in enumerate(dataset, 1):
        print(f"[{index}/{len(dataset)}] {truth}: {video_path.name}")
        try:
            prediction = model.predict_video(
                video_path,
                threshold=args.threshold,
                min_suspicious_windows=args.min_suspicious_windows,
                target_fps=args.target_fps,
                max_windows=args.max_windows,
            )
            row = prediction.to_dict()
            row.update(
                {
                    "truth": truth,
                    "model_alerted": "yes" if prediction.model_alerted else "no",
                    "result_type": classify_result(truth, prediction.model_alerted),
                    "error": "",
                }
            )
        except Exception as exc:
            row = {
                "video_path": str(video_path),
                "truth": truth,
                "model_alerted": "",
                "predicted_label": "",
                "result_type": "",
                "suspicion_score": "",
                "suspicious_windows": "",
                "total_windows": "",
                "processed_frames": "",
                "runtime_seconds": "",
                "processing_fps": "",
                "clip_predictions": [],
                "error": str(exc),
            }
        rows.append(row)

    metrics = compute_metrics(rows)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_silentvision")
    output_dir = args.output_dir or (REPORTS_DIR / "silentvision_evaluation" / run_name)
    write_outputs(output_dir, rows, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
