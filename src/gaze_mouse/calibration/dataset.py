"""Merge and validate calibration .npz archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def val_point_ids_for_grid(rows: int, cols: int) -> tuple[int, ...]:
    """Hold out corner dots for validation (never train on these point_ids)."""
    n = rows * cols
    if n < 2:
        return (0,)
    if rows == 3 and cols == 3:
        return (2, 7)
    if rows >= 2 and cols >= 2:
        top_left = 0
        top_right = cols - 1
        bottom_left = (rows - 1) * cols
        bottom_right = n - 1
        return (top_left, top_right, bottom_left, bottom_right)
    return (n - 1,)


def estimate_calibration_minutes(
    rows: int,
    cols: int,
    samples_per_point: int,
    dwell_ms: int,
    *,
    seconds_per_sample: float = 0.12,
) -> float:
    """Rough duration for one full grid pass (face must be visible)."""
    dots = rows * cols
    dwell_s = dwell_ms / 1000.0
    return dots * (dwell_s + samples_per_point * seconds_per_sample) / 60.0


def append_calibration_arrays(
    path: Path,
    new_X: np.ndarray,
    new_images: np.ndarray,
    new_y: np.ndarray,
    new_point_ids: np.ndarray,
    *,
    rows: int,
    cols: int,
    feature_version: int,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Concatenate with existing .npz if compatible; return None if cannot append."""
    if not path.is_file():
        return new_X, new_images, new_y, new_point_ids

    with np.load(path) as prior:
        if "images" not in prior:
            print(f"cannot append: {path} has no eye crops — run full calibrate (no --append)")
            return None

        old_rows = int(prior["grid_rows"]) if "grid_rows" in prior else -1
        old_cols = int(prior["grid_cols"]) if "grid_cols" in prior else -1
        if old_rows != rows or old_cols != cols:
            print(
                f"cannot append: saved grid {old_cols}x{old_rows} != config {cols}x{rows} "
                "— run fresh calibrate"
            )
            return None

        old_fv = int(prior["feature_version"]) if "feature_version" in prior else -1
        if old_fv != feature_version:
            print(f"cannot append: feature_version {old_fv} != {feature_version} — recalibrate")
            return None

        old_crop = int(prior["crop_size"]) if "crop_size" in prior else -1
        if old_crop != crop_size:
            print(f"cannot append: crop_size {old_crop} != {crop_size} — recalibrate")
            return None

        old_n = int(prior["X"].shape[0])
        X = np.concatenate([prior["X"], new_X], axis=0)
        images = np.concatenate([prior["images"], new_images], axis=0)
        y = np.concatenate([prior["y"], new_y], axis=0)
        point_ids = np.concatenate([prior["point_ids"], new_point_ids], axis=0)

    print(f"appended to {old_n} existing samples → {X.shape[0]} total")
    return X, images, y, point_ids
