"""Regression metrics for screen-coordinate predictions."""

from __future__ import annotations

import numpy as np


def mean_pixel_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average Euclidean distance in pixels between true and predicted (x, y)."""
    return float(np.mean(np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))))
