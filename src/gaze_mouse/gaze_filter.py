"""Stabilize noisy per-frame gaze predictions before moving the cursor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GazeStabilizer:
    """Two-step smooth: model output EMA + outlier rejection."""

    prediction_alpha: float = 0.28
    outlier_px: float = 100.0
    outlier_blend: float = 0.06
    _pred_x: float | None = None
    _pred_y: float | None = None

    def reset(self) -> None:
        self._pred_x = None
        self._pred_y = None

    def filter_prediction(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Return stabilized screen coordinates from a raw model prediction."""
        if self._pred_x is None or self._pred_y is None:
            self._pred_x, self._pred_y = raw_x, raw_y
            return raw_x, raw_y

        dist = float(
            ((raw_x - self._pred_x) ** 2 + (raw_y - self._pred_y) ** 2) ** 0.5
        )
        alpha = self.prediction_alpha
        if dist > self.outlier_px:
            alpha = self.outlier_blend

        self._pred_x = alpha * raw_x + (1.0 - alpha) * self._pred_x
        self._pred_y = alpha * raw_y + (1.0 - alpha) * self._pred_y
        return self._pred_x, self._pred_y
