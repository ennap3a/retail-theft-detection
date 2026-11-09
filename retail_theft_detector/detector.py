from __future__ import annotations

import csv
import math
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .project import load_project, paths_for_project, read_json, update_project, utc_now_iso, write_json

PERSON_CLASS = 0
BAG_CLASSES = {24, 26, 28}  # backpack, handbag, suitcase
PRODUCT_LIKE_CLASSES = {
    39,  # bottle
    40,  # wine glass
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    46,  # banana
    47,  # apple
    49,  # orange
    63,  # laptop
    64,  # mouse
    66,  # keyboard
    67,  # cell phone
    73,  # book
    76,  # scissors
}

BAG_CONTAINER_NAMES = {
    "backpack",
    "handbag",
    "suitcase",
    "bag",
    "shopping bag",
    "shopping_bag",
    "basket",
    "basket cart",
    "basket_cart",
    "cart",
    "shopping cart",
    "shopping_cart",
    "tote",
    "purse",
}

VISIBLE_PRODUCT_NAMES = {
    "held product",
    "held_product",
    "product",
    "retail item",
    "retail_item",
    "item",
    "small package",
    "small_package",
    "package",
    "box",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "banana",
    "apple",
    "orange",
    "laptop",
    "mouse",
    "keyboard",
    "cell phone",
    "phone",
    "book",
    "scissors",
    "cosmetic",
    "clothing item",
    "clothing_item",
}


@dataclass(frozen=True)
class DetectorSettings:
    object_model: str = "yolo11n.pt"
    pose_model: str = "yolo11n-pose.pt"
    tracker: str = "bytetrack.yaml"
    person_class_ids: tuple[int, ...] = (0,)
    product_class_ids: tuple[int, ...] = tuple(sorted(PRODUCT_LIKE_CLASSES | {1}))
    bag_class_ids: tuple[int, ...] = (2, 3, 24, 26, 28)
    object_confidence: float = 0.20
    pose_confidence: float = 0.20
    max_analysis_fps: float = 8.0
    product_change_threshold: float = 0.018
    product_return_threshold: float = 0.010
    interaction_distance_px: float = 60.0
    hand_to_body_margin_px: float = 20.0
    hand_to_bag_distance_px: float = 90.0
    object_to_hand_distance_px: float = 85.0
    missing_confirm_seconds: float = 0.55
    concealment_confirm_seconds: float = 0.35
    leave_confirm_seconds: float = 0.55
    return_confirm_seconds: float = 0.60
    return_zone_distance_px: float = 75.0
    visible_item_grace_seconds: float = 1.25
    hidden_confirm_seconds: float = 1.00
    zone_occlusion_threshold: float = 0.12
    origin_return_confirm_seconds: float = 0.45
    origin_return_patch_threshold: float = 0.12
    origin_box_margin_px: int = 18
    evidence_seconds_before: float = 2.0
    evidence_seconds_after: float = 2.0


@dataclass
class Detection:
    box: list[int]
    cls_id: int
    class_name: str
    confidence: float
    track_id: str | None = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


@dataclass
class Pose:
    box: list[int]
    keypoints: list[tuple[float, float, float]]


def safe_name(text: object) -> str:
    value = str(text).strip() or "unknown"
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def normalize_label(text: object) -> str:
    return str(text).strip().lower().replace("-", " ").replace("_", " ")


def class_name_in(class_name: str, candidates: set[str]) -> bool:
    normalized_name = normalize_label(class_name)
    normalized_candidates = {normalize_label(candidate) for candidate in candidates}
    return normalized_name in normalized_candidates


def is_bag_or_container(det: Detection, settings: DetectorSettings) -> bool:
    if class_name_in(det.class_name, BAG_CONTAINER_NAMES):
        return True
    unknown_name = normalize_label(det.class_name) in {"", "unknown", str(det.cls_id)}
    return unknown_name and det.cls_id in settings.bag_class_ids


def is_visible_product(det: Detection, settings: DetectorSettings) -> bool:
    if is_bag_or_container(det, settings):
        return False
    if class_name_in(det.class_name, VISIBLE_PRODUCT_NAMES):
        return True
    unknown_name = normalize_label(det.class_name) in {"", "unknown", str(det.cls_id)}
    return unknown_name and det.cls_id in settings.product_class_ids


def reset_behavior_state(state: dict[str, Any], status: str) -> None:
    state.update(
        {
            "status": status,
            "interaction_frames": 0,
            "missing_frames": 0,
            "concealment_frames": 0,
            "outside_frames": 0,
            "returned_frames": 0,
            "return_candidate_frames": 0,
            "visible_item_frames": 0,
            "visible_item_lost_frames": 0,
            "hidden_evidence_frames": 0,
            "zone_occluded_frames": 0,
            "origin_returned_frames": 0,
            "origin_box": None,
            "origin_patch_score": None,
            "active_zone_id": None,
            "suspicion_score": 0,
            "last_container_type": "",
        }
    )


def points_to_np(points: list[dict[str, float]]) -> np.ndarray:
    return np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)


def create_mask(shape: tuple[int, int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def point_near_polygon(point: tuple[float, float], polygon: np.ndarray, max_distance: float) -> bool:
    if len(polygon) < 3:
        return False
    distance = cv2.pointPolygonTest(polygon.astype(np.float32), point, True)
    return distance >= -max_distance


def bbox_center(box: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_overlap_area(a: list[int], b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_mask_overlap_ratio(box: list[int], mask: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    zone_pixels = cv2.countNonZero(mask)
    if zone_pixels == 0:
        return 0.0
    return cv2.countNonZero(mask[y1:y2, x1:x2]) / zone_pixels


def iou(a: list[int], b: list[int]) -> float:
    inter = bbox_overlap_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_inside_box(point: tuple[float, float], box: list[int], margin: float = 0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def expand_box(box: list[int], shape: tuple[int, int, int], margin: int = 20) -> list[int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w, x2 + margin),
        min(h, y2 + margin),
    ]


def crop_box(frame: np.ndarray, box: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def hand_points(keypoints: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    hands = []
    for index in (9, 10):  # COCO left/right wrists
        if index < len(keypoints):
            x, y, conf = keypoints[index]
            if conf >= 0.15 and x > 0 and y > 0:
                hands.append((float(x), float(y)))
    return hands


def torso_box(person_box: list[int], keypoints: list[tuple[float, float, float]] | None) -> list[int]:
    if keypoints:
        indexes = [5, 6, 11, 12]  # shoulders and hips
        points = [(keypoints[i][0], keypoints[i][1]) for i in indexes if i < len(keypoints) and keypoints[i][2] >= 0.15]
        if len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    return [
        int(x1 + 0.18 * width),
        int(y1 + 0.20 * height),
        int(x2 - 0.18 * width),
        int(y1 + 0.78 * height),
    ]


def zone_change_score(reference_frame: np.ndarray, current_frame: np.ndarray, mask: np.ndarray) -> float:
    zone_pixels = cv2.countNonZero(mask)
    if zone_pixels == 0:
        return 0.0
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, cur_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresholded = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    changed = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    return cv2.countNonZero(changed) / zone_pixels


def find_changed_box(reference_frame: np.ndarray, current_frame: np.ndarray, mask: np.ndarray) -> tuple[list[int] | None, float]:
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, cur_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    changed = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 35:
        return None, area
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h], area


def patch_change_score(reference_frame: np.ndarray, current_frame: np.ndarray, box: list[int], margin: int = 0) -> float:
    expanded = expand_box(box, reference_frame.shape, margin)
    ref_crop = crop_box(reference_frame, expanded)
    cur_crop = crop_box(current_frame, expanded)
    if ref_crop is None or cur_crop is None or ref_crop.size == 0 or cur_crop.size == 0:
        return 1.0
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, cur_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresholded = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresholded) / max(1, thresholded.size)


def detection_near_box(detections: list[Detection], box: list[int], margin: int = 20) -> bool:
    expanded = [
        box[0] - margin,
        box[1] - margin,
        box[2] + margin,
        box[3] + margin,
    ]
    return any(point_inside_box(det.center, expanded) or iou(det.box, expanded) > 0.05 for det in detections)


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 5), (4, 10, 20), -1)
    cv2.putText(frame, text, (x + 4, y - 3), font, scale, color, thickness, cv2.LINE_AA)


def draw_polygon(frame: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    if len(polygon) < 3:
        return
    pts = polygon.astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    first = tuple(int(v) for v in pts[0])
    draw_label(frame, label, (first[0] + 6, first[1] + 18), color)


def parse_detections(result: Any, model: YOLO) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    ids = getattr(boxes, "id", None)
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        coords = [int(v) for v in box.xyxy[0].tolist()]
        track_id = None
        if ids is not None and idx < len(ids):
            try:
                track_id = str(int(ids[idx].item()))
            except (TypeError, ValueError):
                track_id = None
        detections.append(
            Detection(
                box=coords,
                cls_id=cls_id,
                class_name=str(model.names.get(cls_id, cls_id)),
                confidence=float(box.conf[0].item()),
                track_id=track_id,
            )
        )
    return detections


def parse_poses(result: Any) -> list[Pose]:
    poses: list[Pose] = []
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return poses
    xy = getattr(keypoints, "xy", None)
    conf = getattr(keypoints, "conf", None)
    if xy is None or conf is None:
        return poses

    for idx, box in enumerate(boxes):
        coords = [int(v) for v in box.xyxy[0].tolist()]
        pose_points = []
        for point_idx, point in enumerate(xy[idx].tolist()):
            point_conf = float(conf[idx][point_idx].item()) if point_idx < len(conf[idx]) else 0.0
            pose_points.append((float(point[0]), float(point[1]), point_conf))
        poses.append(Pose(box=coords, keypoints=pose_points))
    return poses


def attach_pose(person: Detection, poses: list[Pose]) -> Pose | None:
    best_pose = None
    best_score = 0.0
    for pose in poses:
        score = iou(person.box, pose.box)
        if score > best_score:
            best_pose = pose
            best_score = score
    return best_pose if best_score >= 0.10 else None


def run_ffmpeg_clip(video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
    ffmpeg = get_ffmpeg_path()
    if ffmpeg is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_time - start_time)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path.exists()
    except subprocess.SubprocessError:
        return False


def get_ffmpeg_path() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_browser_mp4(input_path: Path, output_path: Path) -> Path:
    ffmpeg = get_ffmpeg_path()
    if ffmpeg is None:
        return input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
    except subprocess.SubprocessError:
        pass
    return input_path


def save_debug_change_image(
    output_path: Path,
    frame: np.ndarray,
    polygon: np.ndarray,
    changed_box: list[int] | None,
    zone_label: str,
) -> None:
    debug = frame.copy()
    draw_polygon(debug, polygon, (0, 220, 255), zone_label)
    if changed_box:
        x1, y1, x2, y2 = changed_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
        draw_label(debug, "changed area", (x1, max(20, y1 - 4)), (0, 0, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)


def copy_relative(path: Path, project_dir: Path) -> str:
    try:
        return str(path.relative_to(project_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def analyze_project(
    project_id: str,
    settings: DetectorSettings | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run detection, tracking and behaviour rules over a project's video.

    ``progress`` is called with a fraction in [0, 1] and a short status message.
    """
    settings = settings or DetectorSettings()
    project = load_project(project_id)
    paths = paths_for_project(project_id)
    zones_doc = read_json(paths.zones_path)
    video_path = Path(project["video"]["path"])
    results_dir = paths.results_dir
    evidence_dir = results_dir / "evidence"
    frames_dir = evidence_dir / "frames"
    people_dir = evidence_dir / "people"
    items_dir = evidence_dir / "items"
    clips_dir = evidence_dir / "clips"
    debug_dir = evidence_dir / "debug"
    for folder in (results_dir, frames_dir, people_dir, items_dir, clips_dir, debug_dir):
        folder.mkdir(parents=True, exist_ok=True)

    update_project(project_id, {"stage": "detector_running", "detector_started_at": utc_now_iso()})
    start_wall = time.perf_counter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration = total_frames / source_fps if source_fps else 0.0

    frame_doc = zones_doc.get("frame", {})
    analysis_w = int(frame_doc.get("width") or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    analysis_h = int(frame_doc.get("height") or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    sample_stride = max(1, int(round(source_fps / settings.max_analysis_fps)))
    analysis_fps = source_fps / sample_stride

    setup_seconds = float(frame_doc.get("seconds") or project.get("setup_frame_seconds") or 0.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(setup_seconds * source_fps)))
    ok, reference_frame = cap.read()
    if not ok or reference_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, reference_frame = cap.read()
    if not ok or reference_frame is None:
        raise RuntimeError("Could not read a reference frame from the video.")
    reference_frame = cv2.resize(reference_frame, (analysis_w, analysis_h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    product_zones: list[dict[str, Any]] = []
    customer_zone: dict[str, Any] | None = None
    for zone in zones_doc.get("zones", []):
        polygon = points_to_np(zone["points"])
        prepared = {
            "id": zone["id"],
            "label": zone.get("label", zone["id"]),
            "type": zone["type"],
            "polygon": polygon,
            "mask": create_mask(reference_frame.shape, polygon),
        }
        if zone["type"] == "product":
            product_zones.append(prepared)
        elif zone["type"] == "customer":
            customer_zone = prepared
    if not product_zones or customer_zone is None:
        raise RuntimeError("Detector requires at least one product zone and exactly one customer zone.")

    if progress:
        progress(0.0, "Loading detection and pose models")
    object_model = YOLO(settings.object_model)
    pose_model = YOLO(settings.pose_model)

    raw_annotated_path = results_dir / "annotated_review_raw.mp4"
    annotated_video_path = results_dir / "annotated_review.mp4"
    writer = cv2.VideoWriter(
        str(raw_annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        analysis_fps,
        (analysis_w, analysis_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {raw_annotated_path}")

    states: dict[str, dict[str, Any]] = {}
    alerts: list[dict[str, Any]] = []
    processed_frames = 0
    status_colors = {
        "NORMAL": (220, 220, 220),
        "INTERACTING_WITH_PRODUCT": (0, 220, 255),
        "HOLDING_VISIBLE_ITEM": (255, 210, 90),
        "CHECKING_ITEM_VISIBILITY": (150, 210, 255),
        "POSSIBLE_CONCEALMENT": (0, 150, 255),
        "HIDDEN_ITEM_SUSPECTED": (0, 100, 255),
        "ITEM_RETURNED_OK": (80, 220, 120),
        "HUMAN_REVIEW_REQUIRED": (60, 60, 255),
    }

    confirm_missing_frames = max(1, int(settings.missing_confirm_seconds * analysis_fps))
    confirm_concealment_frames = max(1, int(settings.concealment_confirm_seconds * analysis_fps))
    confirm_leave_frames = max(1, int(settings.leave_confirm_seconds * analysis_fps))
    confirm_return_frames = max(1, int(settings.return_confirm_seconds * analysis_fps))
    visible_item_grace_frames = max(1, int(settings.visible_item_grace_seconds * analysis_fps))
    confirm_hidden_frames = max(
        confirm_concealment_frames,
        int(settings.hidden_confirm_seconds * analysis_fps),
    )
    confirm_origin_return_frames = max(1, int(settings.origin_return_confirm_seconds * analysis_fps))

    for frame_index in range(total_frames if total_frames else 10**9):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_index % sample_stride != 0:
            continue

        current_time = frame_index / source_fps
        frame = cv2.resize(frame, (analysis_w, analysis_h))
        display = frame.copy()

        zone_scores = {
            zone["id"]: zone_change_score(reference_frame, frame, zone["mask"]) for zone in product_zones
        }

        object_results = object_model.track(
            source=frame,
            persist=True,
            tracker=settings.tracker,
            conf=settings.object_confidence,
            verbose=False,
        )
        detections = parse_detections(object_results[0], object_model) if object_results else []
        people = [det for det in detections if det.cls_id in settings.person_class_ids]
        bags = [det for det in detections if is_bag_or_container(det, settings)]
        products = [det for det in detections if is_visible_product(det, settings)]

        pose_results = pose_model.predict(source=frame, conf=settings.pose_confidence, verbose=False)
        poses = parse_poses(pose_results[0]) if pose_results else []

        draw_polygon(display, customer_zone["polygon"], (255, 130, 30), str(customer_zone["label"]))
        for zone in product_zones:
            draw_polygon(display, zone["polygon"], (0, 210, 180), str(zone["label"]))
            score_text = f"{zone['label']} change {zone_scores[zone['id']]:.3f}"
            first = tuple(int(v) for v in zone["polygon"][0])
            draw_label(display, score_text, (first[0] + 6, first[1] + 38), (0, 210, 180))

        for bag in bags:
            x1, y1, x2, y2 = bag.box
            cv2.rectangle(display, (x1, y1), (x2, y2), (180, 90, 255), 2)
            draw_label(display, f"{bag.class_name} {bag.confidence:.2f}", (x1, max(22, y1 - 3)), (180, 90, 255))

        active_track_ids = set()
        for person_index, person in enumerate(people, start=1):
            track_id = person.track_id or f"untracked_{person_index}"
            active_track_ids.add(track_id)
            state = states.setdefault(
                track_id,
                {
                    "status": "NORMAL",
                    "interaction_frames": 0,
                    "missing_frames": 0,
                    "concealment_frames": 0,
                    "outside_frames": 0,
                    "returned_frames": 0,
                    "return_candidate_frames": 0,
                    "visible_item_frames": 0,
                    "visible_item_lost_frames": 0,
                    "hidden_evidence_frames": 0,
                    "zone_occluded_frames": 0,
                    "origin_returned_frames": 0,
                    "origin_box": None,
                    "origin_patch_score": None,
                    "active_zone_id": None,
                    "suspicion_score": 0,
                    "last_container_type": "",
                    "alerted": False,
                },
            )

            pose = attach_pose(person, poses)
            hands = hand_points(pose.keypoints) if pose else []
            body_box = torso_box(person.box, pose.keypoints if pose else None)
            person_center = person.center
            bottom_center = (person_center[0], person.box[3])
            in_customer_zone = point_in_polygon(bottom_center, customer_zone["polygon"]) or point_in_polygon(
                person_center, customer_zone["polygon"]
            )
            if in_customer_zone:
                state["outside_frames"] = 0
            else:
                state["outside_frames"] += 1

            best_interaction_zone = None
            for zone in product_zones:
                hand_near_zone = any(
                    point_near_polygon(hand, zone["polygon"], settings.interaction_distance_px) for hand in hands
                )
                person_near_zone = point_near_polygon(person_center, zone["polygon"], settings.interaction_distance_px)
                changed = zone_scores[zone["id"]] >= settings.product_change_threshold
                if changed and (hand_near_zone or person_near_zone):
                    best_interaction_zone = zone
                    break

            if best_interaction_zone:
                state["active_zone_id"] = best_interaction_zone["id"]
                state["interaction_frames"] += 1
                state["status"] = "INTERACTING_WITH_PRODUCT"
                state["suspicion_score"] = max(int(state["suspicion_score"]), 25)

            active_zone = next((zone for zone in product_zones if zone["id"] == state.get("active_zone_id")), None)
            active_score = zone_scores.get(active_zone["id"], 0.0) if active_zone else 0.0

            object_near_hand = False
            for product in products:
                product_with_person = point_inside_box(product.center, person.box, 18)
                for hand in hands:
                    if point_distance(product.center, hand) <= settings.object_to_hand_distance_px:
                        object_near_hand = True
                        break
                if not object_near_hand and active_zone and product_with_person and state["missing_frames"] > 0:
                    object_near_hand = True
                if object_near_hand:
                    break
            if object_near_hand and state.get("active_zone_id"):
                state["visible_item_frames"] += 1
                state["visible_item_lost_frames"] = 0
                state["return_candidate_frames"] = 0
                state["status"] = "HOLDING_VISIBLE_ITEM"
                state["suspicion_score"] = 20
                state["concealment_frames"] = 0
                state["hidden_evidence_frames"] = 0
            elif state.get("visible_item_frames", 0):
                state["visible_item_lost_frames"] += 1
            item_recently_visible = (
                state.get("visible_item_frames", 0) > 0
                and state.get("visible_item_lost_frames", 0) <= visible_item_grace_frames
            )

            hand_in_body = any(point_inside_box(hand, body_box, settings.hand_to_body_margin_px) for hand in hands)
            hand_near_bag = False
            for bag in bags:
                if iou(person.box, bag.box) > 0.02 or point_distance(person_center, bag.center) < 140:
                    if any(
                        point_distance(hand, bag.center) <= settings.hand_to_bag_distance_px
                        or point_inside_box(hand, bag.box, 35)
                        for hand in hands
                    ):
                        hand_near_bag = True
                        state["last_container_type"] = bag.class_name
                        break
            if hand_in_body and not hand_near_bag:
                state["last_container_type"] = "clothing/body area"

            hand_near_active_zone = bool(active_zone) and any(
                point_near_polygon(hand, active_zone["polygon"], settings.return_zone_distance_px) for hand in hands
            )
            active_zone_occluded = (
                bool(active_zone)
                and not hand_near_active_zone
                and bbox_mask_overlap_ratio(person.box, active_zone["mask"]) >= settings.zone_occlusion_threshold
            )
            origin_box = state.get("origin_box")
            if active_zone and origin_box is None and active_score >= settings.product_change_threshold and not active_zone_occluded:
                changed_box, _ = find_changed_box(reference_frame, frame, active_zone["mask"])
                zone_pixels = cv2.countNonZero(active_zone["mask"])
                if changed_box and 25 <= bbox_area(changed_box) <= max(25, zone_pixels * 0.50):
                    state["origin_box"] = changed_box
                    origin_box = changed_box

            origin_patch_returned = False
            if active_zone and origin_box:
                origin_patch_score = patch_change_score(
                    reference_frame,
                    frame,
                    origin_box,
                    settings.origin_box_margin_px,
                )
                state["origin_patch_score"] = round(origin_patch_score, 4)
                origin_patch_returned = (
                    origin_patch_score <= settings.origin_return_patch_threshold
                    and not active_zone_occluded
                )
                if origin_patch_returned:
                    state["origin_returned_frames"] += 1
                else:
                    state["origin_returned_frames"] = 0

            if active_zone:
                if active_zone_occluded:
                    state["zone_occluded_frames"] += 1
                    state["status"] = "CHECKING_ITEM_VISIBILITY"
                    state["suspicion_score"] = min(max(int(state["suspicion_score"]), 25), 35)
                elif state.get("origin_returned_frames", 0) >= confirm_origin_return_frames:
                    reset_behavior_state(state, "ITEM_RETURNED_OK")
                    active_zone = None
                elif active_score >= settings.product_change_threshold:
                    state["missing_frames"] += 1
                    state["returned_frames"] = 0
                elif active_score <= settings.product_return_threshold:
                    state["returned_frames"] += 1
                    if state["returned_frames"] >= confirm_return_frames:
                        reset_behavior_state(state, "ITEM_RETURNED_OK")
                        active_zone = None

                returned_by_hand_action = (
                    bool(active_zone)
                    and state.get("visible_item_frames", 0) > 0
                    and hand_near_active_zone
                    and not object_near_hand
                    and not hand_in_body
                    and not hand_near_bag
                )
                if returned_by_hand_action:
                    state["return_candidate_frames"] += 1
                    if state["return_candidate_frames"] >= confirm_return_frames:
                        reset_behavior_state(state, "ITEM_RETURNED_OK")
                        active_zone = None
                elif bool(active_zone) and not object_near_hand:
                    state["return_candidate_frames"] = 0

            if active_zone and item_recently_visible:
                state["status"] = "HOLDING_VISIBLE_ITEM"
                state["suspicion_score"] = 20
                state["concealment_frames"] = 0
                state["hidden_evidence_frames"] = 0

            shelf_missing_confirmed = bool(active_zone) and state["missing_frames"] >= confirm_missing_frames
            hand_near_return_origin = hand_near_active_zone
            if origin_box:
                hand_near_return_origin = hand_near_return_origin or any(
                    point_inside_box(hand, origin_box, settings.return_zone_distance_px) for hand in hands
                )
            hidden_motion_cue = (hand_in_body or hand_near_bag) and not hand_near_return_origin
            hidden_candidate = (
                shelf_missing_confirmed
                and hidden_motion_cue
                and not object_near_hand
                and not item_recently_visible
                and not active_zone_occluded
                and state.get("origin_returned_frames", 0) < confirm_origin_return_frames
            )
            if hidden_candidate:
                state["hidden_evidence_frames"] += 1
                state["concealment_frames"] = state["hidden_evidence_frames"]
                state["suspicion_score"] = max(int(state["suspicion_score"]), 55)
                if state["hidden_evidence_frames"] >= confirm_hidden_frames:
                    state["status"] = "POSSIBLE_CONCEALMENT"
                else:
                    state["status"] = "CHECKING_ITEM_VISIBILITY"
            elif object_near_hand or item_recently_visible or hand_near_active_zone:
                state["hidden_evidence_frames"] = 0
                state["concealment_frames"] = 0
            else:
                state["hidden_evidence_frames"] = max(0, int(state["hidden_evidence_frames"]) - 1)
                state["concealment_frames"] = max(0, int(state["concealment_frames"]) - 1)
                if active_zone and shelf_missing_confirmed and int(state["hidden_evidence_frames"]) == 0:
                    state["status"] = "CHECKING_ITEM_VISIBILITY"
                    state["suspicion_score"] = min(max(int(state["suspicion_score"]), 25), 40)

            hidden_item_suspected = (
                bool(active_zone)
                and shelf_missing_confirmed
                and state["hidden_evidence_frames"] >= confirm_hidden_frames
                and not object_near_hand
                and not item_recently_visible
            )
            if hidden_item_suspected:
                state["status"] = "HIDDEN_ITEM_SUSPECTED"
                state["suspicion_score"] = max(int(state["suspicion_score"]), 75)

            final_alert = (
                hidden_item_suspected
                and not in_customer_zone
                and state["outside_frames"] >= confirm_leave_frames
                and not state["alerted"]
            )
            if final_alert:
                state["status"] = "HUMAN_REVIEW_REQUIRED"
                state["suspicion_score"] = 95
                state["alerted"] = True
                alert_number = len(alerts) + 1
                alert_id = f"ALERT_{alert_number:03d}_ID_{safe_name(track_id)}_{current_time:.2f}s"
                full_frame_path = frames_dir / f"{alert_id}_full_frame.jpg"
                person_crop_path = people_dir / f"{alert_id}_person_crop.jpg"
                item_crop_path = items_dir / f"{alert_id}_suspected_item_crop.jpg"
                debug_path = debug_dir / f"{alert_id}_changed_zone_debug.jpg"
                clip_path = clips_dir / f"{alert_id}_clip.mp4"

                cv2.imwrite(str(full_frame_path), frame)
                person_crop = crop_box(frame, expand_box(person.box, frame.shape, 20))
                if person_crop is not None:
                    cv2.imwrite(str(person_crop_path), person_crop)

                changed_box, changed_area = (None, 0.0)
                if active_zone:
                    changed_box, changed_area = find_changed_box(reference_frame, frame, active_zone["mask"])
                    if changed_box:
                        item_crop = crop_box(reference_frame, expand_box(changed_box, reference_frame.shape, 12))
                        if item_crop is not None:
                            cv2.imwrite(str(item_crop_path), item_crop)
                    save_debug_change_image(debug_path, frame, active_zone["polygon"], changed_box, str(active_zone["label"]))

                clip_start = max(0.0, current_time - settings.evidence_seconds_before)
                clip_end = min(source_duration, current_time + settings.evidence_seconds_after) if source_duration else current_time
                clip_created = run_ffmpeg_clip(video_path, clip_start, clip_end, clip_path)

                alerts.append(
                    {
                        "alert_id": alert_id,
                        "status": "HUMAN_REVIEW_REQUIRED",
                        "review_priority": "HIGH",
                        "person_id": track_id,
                        "time_seconds": round(current_time, 2),
                        "suspicion_score": state["suspicion_score"],
                        "product_zone": active_zone["id"] if active_zone else "",
                        "product_zone_label": active_zone["label"] if active_zone else "",
                        "product_change_score": round(active_score, 4),
                        "changed_area_pixels": round(changed_area, 2),
                        "container_interaction": state.get("last_container_type", ""),
                        "visible_item_frames": int(state.get("visible_item_frames", 0)),
                        "item_hidden_frames": int(state.get("visible_item_lost_frames", 0)),
                        "origin_patch_score": state.get("origin_patch_score", ""),
                        "origin_returned_frames": int(state.get("origin_returned_frames", 0)),
                        "full_frame_image": copy_relative(full_frame_path, paths.base_dir),
                        "person_crop_image": copy_relative(person_crop_path, paths.base_dir) if person_crop_path.exists() else "",
                        "suspected_item_crop_image": copy_relative(item_crop_path, paths.base_dir) if item_crop_path.exists() else "",
                        "debug_image": copy_relative(debug_path, paths.base_dir) if debug_path.exists() else "",
                        "evidence_clip": copy_relative(clip_path, paths.base_dir) if clip_created else "",
                        "notes": "Product-zone change followed by hidden item, body/bag interaction, and leaving customer zone. Human review required.",
                    }
                )

            x1, y1, x2, y2 = person.box
            color = status_colors.get(state["status"], (255, 255, 255))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(display, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (255, 180, 60), 1)
            for hand in hands:
                cv2.circle(display, (int(hand[0]), int(hand[1])), 5, (0, 255, 255), -1)
            label = f"ID {track_id} | {state['status']} | score {state['suspicion_score']}"
            draw_label(display, label, (x1, max(22, y1 - 4)), color)

        header = f"Retail Theft Review Assistant | t={current_time:.1f}s | alerts={len(alerts)}"
        cv2.rectangle(display, (0, 0), (analysis_w, 34), (4, 10, 20), -1)
        cv2.putText(display, header, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(display)
        processed_frames += 1
        if progress and total_frames:
            progress(min(0.95, frame_index / total_frames * 0.95), f"Analysing {current_time:.1f}s / {source_duration:.1f}s")

    cap.release()
    writer.release()

    if progress:
        progress(0.96, "Encoding the annotated review video")
    final_video = convert_to_browser_mp4(raw_annotated_path, annotated_video_path)

    alerts_json_path = results_dir / "alerts.json"
    alerts_csv_path = results_dir / "alerts.csv"
    summary_csv_path = results_dir / "review_summary.csv"
    write_json(alerts_json_path, {"alerts": alerts})
    with alerts_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "alert_id",
            "status",
            "review_priority",
            "person_id",
            "time_seconds",
            "suspicion_score",
            "product_zone",
            "product_zone_label",
            "product_change_score",
            "changed_area_pixels",
            "container_interaction",
            "visible_item_frames",
            "item_hidden_frames",
            "origin_patch_score",
            "origin_returned_frames",
            "full_frame_image",
            "person_crop_image",
            "suspected_item_crop_image",
            "debug_image",
            "evidence_clip",
            "notes",
        ]
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        for alert in alerts:
            writer_csv.writerow(alert)

    runtime_seconds = time.perf_counter() - start_wall
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        summary_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "project_id",
                "total_alerts",
                "processed_frames",
                "analysis_fps",
                "runtime_seconds",
                "processing_fps",
                "result_note",
            ],
        )
        summary_writer.writeheader()
        summary_writer.writerow(
            {
                "project_id": project_id,
                "total_alerts": len(alerts),
                "processed_frames": processed_frames,
                "analysis_fps": round(analysis_fps, 3),
                "runtime_seconds": round(runtime_seconds, 2),
                "processing_fps": round(processed_frames / runtime_seconds, 3) if runtime_seconds > 0 else 0,
                "result_note": "No human-review alerts generated." if not alerts else "Human-review alerts generated.",
            }
        )

    summary = {
        "schema_version": 1,
        "project_id": project_id,
        "created_at": utc_now_iso(),
        "settings": asdict(settings),
        "video": {
            "source_fps": round(source_fps, 3),
            "source_frames": total_frames,
            "source_duration_seconds": round(source_duration, 3),
            "analysis_width": analysis_w,
            "analysis_height": analysis_h,
            "sample_stride": sample_stride,
            "analysis_fps": round(analysis_fps, 3),
            "processed_frames": processed_frames,
        },
        "results": {
            "total_alerts": len(alerts),
            "annotated_video": copy_relative(final_video, paths.base_dir),
            "alerts_json": copy_relative(alerts_json_path, paths.base_dir),
            "alerts_csv": copy_relative(alerts_csv_path, paths.base_dir),
            "summary_csv": copy_relative(summary_csv_path, paths.base_dir),
            "runtime_seconds": round(runtime_seconds, 2),
            "processing_fps": round(processed_frames / runtime_seconds, 3) if runtime_seconds > 0 else 0,
        },
        "responsible_ai_note": "Alerts are suspected events for human review and are not automatic accusations.",
    }
    summary_path = results_dir / "summary.json"
    write_json(summary_path, summary)
    update_project(
        project_id,
        {
            "stage": "detector_complete",
            "detector_completed_at": utc_now_iso(),
            "last_results": summary["results"],
        },
    )
    if progress:
        progress(1.0, "Done")
    return summary
