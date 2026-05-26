"""Face landmark → fixed-length feature vector (calibrate / train / eval / run).

See docs/MODEL.md. When FEATURE_VERSION changes, re-run calibrate — old .npz rows are incompatible.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# --- version (bump + recalibrate when vector definition changes) -----------------

FEATURE_VERSION = 3
"""1 = raw full mesh. 2 = head-normalized full mesh (956). 3 = head-normalized eyes+iris (84)."""

# Base face mesh count; refine_landmarks=True (used in __main__) adds 10 iris points.
NUM_LANDMARKS_BASE = 468
NUM_LANDMARKS = 478

# Smallest inter-eye distance we allow as scale (avoids divide-by-zero).
_MIN_INTER_EYE_SCALE = 1e-6

# MediaPipe Face Mesh indices (first face, 468 landmarks). Diagram:
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

LEFT_EYE_INDICES: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
)
RIGHT_EYE_INDICES: tuple[int, ...] = (
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
)
# refine_landmarks=True adds iris points 468–477 (five per eye).
IRIS_INDICES: tuple[int, ...] = tuple(range(468, 478))

# v3: contours + iris only (gaze-relevant, much smaller than 956-D full mesh).
EYE_FEATURE_INDICES: tuple[int, ...] = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + IRIS_INDICES


def feature_dim() -> int:
    """Length of vectors from extract_feature_vector() for the current FEATURE_VERSION."""
    if FEATURE_VERSION == 1:
        return NUM_LANDMARKS * 2
    if FEATURE_VERSION == 2:
        return NUM_LANDMARKS * 2
    if FEATURE_VERSION == 3:
        return len(EYE_FEATURE_INDICES) * 2
    raise ValueError(f"unknown FEATURE_VERSION={FEATURE_VERSION}")


def _mesh_landmarks(landmarks: Sequence[Any]) -> Sequence[Any]:
    """First NUM_LANDMARKS points (matches FaceMesh with refine_landmarks=True)."""
    if len(landmarks) < NUM_LANDMARKS_BASE:
        raise ValueError(f"expected at least {NUM_LANDMARKS_BASE} landmarks, got {len(landmarks)}")
    return landmarks[:NUM_LANDMARKS]


def extract_feature_vector(multi_face_landmarks: Any | None) -> np.ndarray | None:
    """Convert FaceMesh output to one float32 row, or None if no face."""
    if multi_face_landmarks is None:
        return None

    landmarks = multi_face_landmarks[0].landmark

    if FEATURE_VERSION == 1:
        return _features_v1_raw_xy(landmarks)

    if FEATURE_VERSION == 2:
        return _features_v2_head_normalized(landmarks)

    if FEATURE_VERSION == 3:
        return _features_v3_eyes_head_normalized(landmarks)

    raise ValueError(f"unknown FEATURE_VERSION={FEATURE_VERSION}")


def _landmark_xy(landmarks: Sequence[Any], index: int) -> tuple[float, float]:
    """Read normalized image x,y for one landmark index (0–1 in frame)."""
    lm = landmarks[index]
    return float(lm.x), float(lm.y)


def _head_frame(landmarks: Sequence[Any]) -> tuple[float, float, float]:
    """Center and scale: eye midpoint and inter-eye distance."""
    left_x, left_y = _landmark_xy(landmarks, LEFT_EYE_OUTER)
    right_x, right_y = _landmark_xy(landmarks, RIGHT_EYE_OUTER)
    cx = (left_x + right_x) * 0.5
    cy = (left_y + right_y) * 0.5
    scale = float(np.hypot(right_x - left_x, right_y - left_y))
    if scale < _MIN_INTER_EYE_SCALE:
        scale = _MIN_INTER_EYE_SCALE
    return cx, cy, scale


def _features_v1_raw_xy(landmarks: Sequence[Any]) -> np.ndarray:
    """v1: raw x then y for each mesh landmark (image-normalized 0–1)."""
    mesh = _mesh_landmarks(landmarks)
    out = np.empty(NUM_LANDMARKS * 2, dtype=np.float32)
    j = 0
    for lm in mesh:
        out[j] = lm.x
        out[j + 1] = lm.y
        j += 2
    return out


def _features_v2_head_normalized(landmarks: Sequence[Any]) -> np.ndarray:
    """v2: head-normalized (x,y) for all mesh landmarks."""
    mesh = _mesh_landmarks(landmarks)
    cx, cy, scale = _head_frame(landmarks)
    out = np.empty(NUM_LANDMARKS * 2, dtype=np.float32)
    j = 0
    for lm in mesh:
        out[j] = (lm.x - cx) / scale
        out[j + 1] = (lm.y - cy) / scale
        j += 2
    return out


def _features_v3_eyes_head_normalized(landmarks: Sequence[Any]) -> np.ndarray:
    """v3: head-normalized (x,y) for eye contours + iris (requires refine_landmarks)."""
    if len(landmarks) < NUM_LANDMARKS:
        raise ValueError(
            f"v3 features need {NUM_LANDMARKS} landmarks (enable refine_landmarks=True), "
            f"got {len(landmarks)}"
        )
    cx, cy, scale = _head_frame(landmarks)
    out = np.empty(len(EYE_FEATURE_INDICES) * 2, dtype=np.float32)
    j = 0
    for index in EYE_FEATURE_INDICES:
        lm = landmarks[index]
        out[j] = (lm.x - cx) / scale
        out[j + 1] = (lm.y - cy) / scale
        j += 2
    return out


# =============================================================================
# Test:
#   python -c "from gaze_mouse.features import FEATURE_VERSION, feature_dim; print(FEATURE_VERSION, feature_dim())"
# Expected: 3 84
#
# After changing FEATURE_VERSION, recalibrate then train:
#   python -m gaze_mouse calibrate --profile test1
#   python -m gaze_mouse train --profile test1 --model ridge
#   python -m gaze_mouse eval --profile test1
# =============================================================================
