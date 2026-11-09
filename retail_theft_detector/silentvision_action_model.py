from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ("bag_concealment", "clothes_concealment", "normal")


class SilentVisionUnavailable(RuntimeError):
    """Raised when optional TensorFlow dependencies are not installed."""


@dataclass(frozen=True)
class ClipPrediction:
    clip_index: int
    start_sample: int
    end_sample: int
    bag_concealment: float
    clothes_concealment: float
    normal: float
    predicted_label: str
    suspicion_score: float
    suspicious: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VideoPrediction:
    video_path: str
    model_name: str
    weights_path: str
    predicted_label: str
    model_alerted: bool
    suspicion_score: float
    suspicious_windows: int
    total_windows: int
    processed_frames: int
    runtime_seconds: float
    processing_fps: float
    clip_predictions: list[ClipPrediction]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["clip_predictions"] = [prediction.to_dict() for prediction in self.clip_predictions]
        return data


def _require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Add, Conv3D, Dense, Flatten, Input, Lambda, MaxPooling3D
        from tensorflow.keras.models import Model
    except Exception as exc:  # pragma: no cover - depends on optional local install
        raise SilentVisionUnavailable(
            "TensorFlow is required for the SilentVision/GateFlowSlowFast evaluator. "
            "Install the optional dependency in Colab or in a separate environment before running it."
        ) from exc
    return tf, Add, Conv3D, Dense, Flatten, Input, Lambda, MaxPooling3D, Model


def _build_gate_flow_slow_fast_rgb_model():
    _, Add, Conv3D, Dense, Flatten, Input, Lambda, MaxPooling3D, Model = _require_tensorflow()

    def conv3d(filters: int, kernel_size: tuple[int, int, int], activation: str = "relu"):
        return Conv3D(
            filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            kernel_initializer="he_normal",
            activation=activation,
            padding="same",
        )

    def fast_path(inputs):
        rgb = conv3d(16, (1, 3, 3))(inputs)
        rgb = conv3d(16, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        rgb = conv3d(16, (1, 3, 3))(rgb)
        rgb = conv3d(16, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        con_1 = Lambda(lambda x: x[:, ::18, ...], name="con_1")(rgb)

        rgb = conv3d(32, (1, 3, 3))(rgb)
        rgb = conv3d(32, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        rgb = conv3d(32, (1, 3, 3))(rgb)
        rgb = conv3d(32, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        con_2 = Lambda(lambda x: x[:, ::18, ...], name="con_2")(rgb)
        return rgb, {"con-1": con_1, "con-2": con_2}

    def slow_path(inputs, connections):
        rgb = conv3d(16, (1, 3, 3))(inputs)
        rgb = conv3d(16, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        rgb = conv3d(16, (1, 3, 3))(rgb)
        rgb = conv3d(16, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        rgb = Add(name="connection_1_rgb")([rgb, connections["con-1"]])

        rgb = conv3d(32, (1, 3, 3))(rgb)
        rgb = conv3d(32, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        rgb = conv3d(32, (1, 3, 3))(rgb)
        rgb = conv3d(32, (3, 1, 1))(rgb)
        rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        rgb = Add(name="connection_2_rgb")([rgb, connections["con-2"]])

        x = MaxPooling3D(pool_size=(1, 2, 2))(rgb)
        x = conv3d(64, (1, 3, 3))(x)
        x = conv3d(64, (3, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = conv3d(64, (1, 3, 3))(x)
        x = conv3d(64, (3, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = conv3d(128, (1, 3, 3))(x)
        x = conv3d(128, (3, 1, 1))(x)
        return x

    def merging_block(x):
        x = conv3d(64, (1, 3, 3))(x)
        x = conv3d(64, (3, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = conv3d(64, (1, 3, 3))(x)
        x = conv3d(64, (3, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = conv3d(128, (1, 3, 3))(x)
        x = conv3d(128, (3, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 3, 3))(x)
        return x

    clip_input = Input(shape=(64, 224, 224, 3))
    slow_input = Lambda(lambda x: x[:, ::16, ...], name="slow_input")(clip_input)
    fast_rgb, connections = fast_path(clip_input)
    slow_rgb = slow_path(slow_input, connections)
    fast_rgb = merging_block(fast_rgb)

    x = Add(name="ADD_slow_rgb_ans_fast_rgb_opt")([fast_rgb, slow_rgb])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    pred = Dense(3, activation="softmax")(x)
    return Model(inputs=clip_input, outputs=pred)


def _normalize_clip(clip: np.ndarray) -> np.ndarray:
    clip = clip.astype(np.float32)
    std = float(np.std(clip))
    if std < 1e-6:
        std = 1.0
    return (clip - float(np.mean(clip))) / std


def _read_video_samples(video_path: Path, target_fps: float = 10.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    step = 1
    if target_fps > 0 and source_fps > target_fps:
        step = max(1, int(round(source_fps / target_fps)))

    samples: list[np.ndarray] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            samples.append(frame)
        frame_index += 1
    cap.release()
    return samples


def _iter_windows(samples: list[np.ndarray], window_size: int = 64, step: int = 32, max_windows: int | None = None):
    if not samples:
        return
    start = 0
    clip_index = 0
    while start < len(samples):
        window = samples[start : start + window_size]
        if len(window) < window_size:
            pad_frame = window[-1] if window else samples[-1]
            window = window + [pad_frame] * (window_size - len(window))
        yield clip_index, start, start + window_size, np.array(window, dtype=np.float32)
        clip_index += 1
        if max_windows is not None and clip_index >= max_windows:
            break
        if start + window_size >= len(samples):
            break
        start += step


class SilentVisionActionModel:
    def __init__(self, weights_path: str | Path):
        self.weights_path = Path(weights_path)
        self.model = None

    def load(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"SilentVision weights not found: {self.weights_path}")
        model = _build_gate_flow_slow_fast_rgb_model()
        model.load_weights(str(self.weights_path))
        self.model = model

    def predict_video(
        self,
        video_path: str | Path,
        *,
        threshold: float = 0.55,
        min_suspicious_windows: int = 2,
        target_fps: float = 10.0,
        max_windows: int | None = None,
    ) -> VideoPrediction:
        if self.model is None:
            self.load()

        assert self.model is not None
        video_path = Path(video_path)
        started = time.perf_counter()
        samples = _read_video_samples(video_path, target_fps=target_fps)

        predictions: list[ClipPrediction] = []
        for clip_index, start_sample, end_sample, window in _iter_windows(samples, max_windows=max_windows):
            model_input = _normalize_clip(window).reshape((1, 64, 224, 224, 3))
            raw = self.model.predict(model_input, verbose=0)[0]
            bag = float(raw[0])
            clothes = float(raw[1])
            normal = float(raw[2])
            label_index = int(np.argmax(raw))
            predicted_label = CLASS_NAMES[label_index]
            suspicion_score = max(bag, clothes)
            suspicious = predicted_label != "normal" and suspicion_score >= threshold
            predictions.append(
                ClipPrediction(
                    clip_index=clip_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    bag_concealment=bag,
                    clothes_concealment=clothes,
                    normal=normal,
                    predicted_label=predicted_label,
                    suspicion_score=suspicion_score,
                    suspicious=suspicious,
                )
            )

        runtime = time.perf_counter() - started
        suspicious_windows = sum(1 for prediction in predictions if prediction.suspicious)
        model_alerted = suspicious_windows >= min_suspicious_windows
        best_prediction = max(predictions, key=lambda item: item.suspicion_score, default=None)
        suspicion_score = best_prediction.suspicion_score if best_prediction else 0.0
        predicted_label = best_prediction.predicted_label if model_alerted and best_prediction else "normal"
        processing_fps = len(samples) / runtime if runtime > 0 else 0.0

        return VideoPrediction(
            video_path=str(video_path),
            model_name="SilentVision GateFlowSlowFast RGB",
            weights_path=str(self.weights_path),
            predicted_label=predicted_label,
            model_alerted=model_alerted,
            suspicion_score=suspicion_score,
            suspicious_windows=suspicious_windows,
            total_windows=len(predictions),
            processed_frames=len(samples),
            runtime_seconds=round(runtime, 4),
            processing_fps=round(processing_fps, 4),
            clip_predictions=predictions,
        )
