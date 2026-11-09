import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
TMP_DIR = DATA_DIR / "tmp"
DATASETS_DIR = ROOT_DIR / "datasets"
REPORTS_DIR = ROOT_DIR / "reports"
LOCAL_REVIEW_DIR = ROOT_DIR / "local_review_workspace"
EVALUATION_LOG_PATH = LOCAL_REVIEW_DIR / "evaluation_log.csv"
NORMAL_TEST_VIDEOS_DIR = LOCAL_REVIEW_DIR / "normal_videos"
SUSPICIOUS_TEST_VIDEOS_DIR = LOCAL_REVIEW_DIR / "suspicious_videos"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 7860
    max_upload_mb: int = 1500


SETTINGS = AppSettings()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def retention_hours() -> float:
    """Hosted demos: delete uploaded projects after this many hours (0 keeps them)."""
    return _env_float("RETENTION_HOURS", 0)


def max_video_seconds() -> float:
    """Hosted demos: reject videos longer than this many seconds (0 means no limit)."""
    return _env_float("MAX_VIDEO_SECONDS", 0)
