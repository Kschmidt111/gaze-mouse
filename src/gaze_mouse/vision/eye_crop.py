"""Extract a fixed-size eye-region crop from a webcam frame + FaceMesh landmarks."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from gaze_mouse.features import EYE_FEATURE_INDICES, NUM_LANDMARKS


def extract_eye_crop(
    frame_bgr: np.ndarray,
    multi_face_landmarks: Any,
    crop_size: int = 128,
    padding_ratio: float = 0.4,
) -> np.ndarray | None:
    """Crop the eye region, resize to (crop_size, crop_size), return BGR uint8."""
    if multi_face_landmarks is None:
        return None

    landmarks = multi_face_landmarks[0].landmark
    if len(landmarks) < NUM_LANDMARKS:
        return None

    height, width = frame_bgr.shape[:2]
    xs: list[float] = []
    ys: list[float] = []
    for index in EYE_FEATURE_INDICES:
        lm = landmarks[index]
        xs.append(lm.x * width)
        ys.append(lm.y * height)

    x_min = float(min(xs))
    x_max = float(max(xs))
    y_min = float(min(ys))
    y_max = float(max(ys))
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < 2 or box_h < 2:
        return None

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    x0 = int(max(0, np.floor(x_min - pad_x)))
    y0 = int(max(0, np.floor(y_min - pad_y)))
    x1 = int(min(width, np.ceil(x_max + pad_x)))
    y1 = int(min(height, np.ceil(y_max + pad_y)))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
