"""Load settings from config.yaml into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    index: int
    width: int
    height: int
    fps_target: int


@dataclass
class ScreenConfig:
    mode: str


@dataclass
class GazeConfig:
    gain_x: float
    gain_y: float
    smoothing_alpha: float
    max_jump_px: float
    feature_smoothing_alpha: float
    prediction_alpha: float
    prediction_outlier_px: float


@dataclass
class SafetyConfig:
    kill_key: str
    face_lost_frames: int
    on_face_lost: str


@dataclass
class CalibrationConfig:
    grid: str
    samples_per_point: int
    dwell_ms_to_start: int


@dataclass
class ModelConfig:
    type: str
    hidden: list[int]
    learning_rate: float
    epochs: int
    batch_size: int
    crop_size: int
    backbone: str


@dataclass
class AppConfig:
    profile: str
    camera: CameraConfig
    screen: ScreenConfig
    gaze: GazeConfig
    safety: SafetyConfig
    calibration: CalibrationConfig
    model: ModelConfig


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_config_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit

    root = project_root()
    preferred = root / "config.yaml"
    if preferred.is_file():
        return preferred
    return root / "config.example.yaml"


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")

    return data


def _parse_camera(raw: dict[str, Any]) -> CameraConfig:
    return CameraConfig(
        index=raw["index"],
        width=raw["width"],
        height=raw["height"],
        fps_target=raw["fps_target"],
    )


def _parse_screen(raw: dict[str, Any]) -> ScreenConfig:
    return ScreenConfig(mode=raw["mode"])


def _parse_gaze(raw: dict[str, Any]) -> GazeConfig:
    return GazeConfig(
        gain_x=raw["gain_x"],
        gain_y=raw["gain_y"],
        smoothing_alpha=raw["smoothing_alpha"],
        max_jump_px=raw["max_jump_px"],
        feature_smoothing_alpha=float(raw.get("feature_smoothing_alpha", 0.4)),
        prediction_alpha=float(raw.get("prediction_alpha", 0.28)),
        prediction_outlier_px=float(raw.get("prediction_outlier_px", 100.0)),
    )


def _parse_safety(raw: dict[str, Any]) -> SafetyConfig:
    return SafetyConfig(
        kill_key=raw["kill_key"],
        face_lost_frames=raw["face_lost_frames"],
        on_face_lost=raw["on_face_lost"],
    )


def _parse_calibration(raw: dict[str, Any]) -> CalibrationConfig:
    return CalibrationConfig(
        grid=raw["grid"],
        samples_per_point=raw["samples_per_point"],
        dwell_ms_to_start=raw["dwell_ms_to_start"],
    )


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        type=raw["type"],
        hidden=raw["hidden"],
        learning_rate=raw["learning_rate"],
        epochs=raw["epochs"],
        batch_size=raw["batch_size"],
        crop_size=int(raw.get("crop_size", 128)),
        backbone=str(raw.get("backbone", "resnet18")),
    )


def _parse_app_config(data: dict[str, Any]) -> AppConfig:
    return AppConfig(
        profile=data["profile"],
        camera=_parse_camera(data["camera"]),
        screen=_parse_screen(data["screen"]),
        gaze=_parse_gaze(data["gaze"]),
        safety=_parse_safety(data["safety"]),
        calibration=_parse_calibration(data["calibration"]),
        model=_parse_model(data["model"]),
    )


def load_config(path: Path | None = None) -> AppConfig:
    config_path = resolve_config_path(path)
    data = _load_yaml_dict(config_path)
    return _parse_app_config(data)
