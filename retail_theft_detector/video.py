from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .config import ALLOWED_VIDEO_EXTENSIONS


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration_seconds: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fps"] = round(self.fps, 3)
        data["duration_seconds"] = round(self.duration_seconds, 3)
        return data


def is_allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def safe_filename(filename: str) -> str:
    keep = []
    for char in Path(filename).name:
        if char.isalnum() or char in {".", "-", "_"}:
            keep.append(char)
        elif char in {" ", "(", ")"}:
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "uploaded_video.mp4"


def probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0
    duration = frames / fps if fps else 0.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video has invalid dimensions: {video_path}")

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration_seconds=duration,
    )


def extract_frame(
    video_path: Path,
    output_path: Path,
    at_seconds: float = 0.0,
    max_width: int | None = 1280,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_index = max(0, int(at_seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame at {at_seconds:.2f}s")

    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        new_size = (max_width, int(frame.shape[0] * scale))
        frame = cv2.resize(frame, new_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write frame: {output_path}")
    return output_path


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    rest = seconds - (minutes * 60)
    if minutes:
        return f"{minutes}m {rest:.1f}s"
    return f"{rest:.1f}s"
