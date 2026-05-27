"""CLI entry point for gaze-mouse."""

import argparse
import ctypes
import math
import time
from typing import Any, cast

import cv2
import joblib
import numpy as np
import pygame
from mediapipe.python.solutions.face_mesh import FaceMesh
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from pynput import keyboard, mouse

from gaze_mouse.config import AppConfig, GazeConfig, load_config, project_root
from gaze_mouse.features import FEATURE_VERSION, extract_feature_vector, feature_dim
from gaze_mouse.training.metrics import mean_pixel_error
from gaze_mouse.training.cnn import (
    CnnTrainResult,
    load_cnn_predictor,
    save_cnn_checkpoint,
    train_gaze_cnn,
)
from gaze_mouse.training.mlp import (
    TrainResult,
    load_mlp_predictor,
    save_mlp_checkpoint,
    train_gaze_mlp,
)
from gaze_mouse.calibration.dataset import (
    append_calibration_arrays,
    estimate_calibration_minutes,
    val_point_ids_for_grid,
)
from gaze_mouse.gaze_filter import GazeStabilizer
from gaze_mouse.vision.eye_crop import extract_eye_crop


def default_val_point_ids(grid: str) -> tuple[int, ...]:
    """Return calibration point ids held out for validation (corner dots)."""
    rows, cols = parse_calibration_grid(grid)
    return val_point_ids_for_grid(rows, cols)


def calibration_data_path(profile: str) -> Path:
    return project_root() / "data" / "calibrations" / f"{profile}.npz"


def model_artifact_path(profile: str) -> Path:
    return project_root() / "models" / profile / "gaze_model.joblib"


def mlp_checkpoint_path(profile: str) -> Path:
    return project_root() / "models" / profile / "mlp.pt"


def cnn_checkpoint_path(profile: str) -> Path:
    return project_root() / "models" / profile / "cnn.pt"


def load_calibration_arrays(profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, point_ids from calibration npz; exit path should check is_file first."""
    data = np.load(calibration_data_path(profile))
    return data["X"], data["y"], data["point_ids"]


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    point_ids: np.ndarray,
    val_point_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by entire calibration dots — never mix frames from the same dot."""
    val_mask = np.isin(point_ids, val_point_ids)
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]


def save_gaze_artifact(
    profile: str,
    model: Any,
    scaler: StandardScaler,
    model_type: str,
    val_point_ids: tuple[int, ...],
) -> Path:
    """Save model + scaler + metadata for eval and run."""
    path = model_artifact_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "scaler": scaler,
        "model_type": model_type,
        "val_point_ids": np.asarray(val_point_ids, dtype=np.int32),
    }
    if model_type == "mlp":
        bundle["model"] = None
        bundle["y_scaler"] = model.y_scaler
    elif model_type == "cnn":
        bundle["model"] = None
        bundle["y_scaler"] = model.y_scaler
        bundle["crop_size"] = int(model.crop_size)
    else:
        bundle["model"] = model
    joblib.dump(bundle, path)
    return path


def load_gaze_artifact(profile: str) -> dict[str, Any]:
    """Load bundled model artifact; raise clear errors if missing or legacy format."""
    path = model_artifact_path(profile)
    legacy = project_root() / "models" / f"{profile}.pkl"
    if path.is_file():
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "scaler" not in bundle:
            raise RuntimeError(f"invalid gaze model artifact: {path}")
        model_type = bundle.get("model_type", "ridge")
        if model_type == "mlp":
            ckpt = mlp_checkpoint_path(profile)
            bundle["model"] = load_mlp_predictor(ckpt, y_scaler=bundle.get("y_scaler"))
        elif model_type == "cnn":
            ckpt = cnn_checkpoint_path(profile)
            feat_scaler = bundle.get("scaler")
            if isinstance(feat_scaler, StandardScaler):
                bundle["model"] = load_cnn_predictor(ckpt, x_scaler=feat_scaler)
            else:
                bundle["model"] = load_cnn_predictor(ckpt)
        elif bundle.get("model") is None:
            raise RuntimeError(f"checkpoint missing for profile {profile}; retrain")
        return bundle
    if legacy.is_file():
        raise RuntimeError(
            f"found old model at {legacy} (scaler missing). "
            f"retrain: python -m gaze_mouse train --profile {profile} --model ridge"
        )
    raise FileNotFoundError(
        f"no trained model at {path}. train first: python -m gaze_mouse train --profile {profile}"
    )


def predict_screen_position(
    model: Any,
    scaler: StandardScaler,
    model_type: str,
    *,
    features: np.ndarray | None = None,
    crop_bgr: np.ndarray | None = None,
) -> tuple[float, float]:
    """Predict screen (x, y) from features (ridge/mlp) or eye crop (cnn)."""
    if model_type == "cnn":
        if crop_bgr is None:
            raise ValueError("cnn model requires crop_bgr")
        if getattr(model, "fusion", False) and features is None:
            raise ValueError("fusion cnn requires features")
        pred = model.predict(crop_bgr, features=features)
        return float(pred[0]), float(pred[1])
    if features is None:
        raise ValueError(f"{model_type} model requires features")
    row = np.asarray(features, dtype=np.float32).reshape(1, -1)
    scaled = scaler.transform(row)
    pred = model.predict(scaled)[0]
    return float(pred[0]), float(pred[1])


def apply_gaze_mapping(
    raw_x: float,
    raw_y: float,
    screen_width: int,
    screen_height: int,
    gaze: GazeConfig,
    prev_x: float | None,
    prev_y: float | None,
) -> tuple[float, float]:
    """One smoothing step: gain from center, EMA toward target, optional jump cap."""
    center_x = screen_width / 2.0
    center_y = screen_height / 2.0
    target_x = center_x + (raw_x - center_x) * gaze.gain_x
    target_y = center_y + (raw_y - center_y) * gaze.gain_y
    if prev_x is None or prev_y is None:
        smooth_x, smooth_y = target_x, target_y
    else:
        alpha = gaze.smoothing_alpha
        smooth_x = alpha * target_x + (1.0 - alpha) * prev_x
        smooth_y = alpha * target_y + (1.0 - alpha) * prev_y
        dx = smooth_x - prev_x
        dy = smooth_y - prev_y
        dist = float(np.hypot(dx, dy))
        if gaze.max_jump_px > 0 and dist > gaze.max_jump_px:
            scale = gaze.max_jump_px / dist
            smooth_x = prev_x + dx * scale
            smooth_y = prev_y + dy * scale
    return (
        float(np.clip(smooth_x, 0, screen_width - 1)),
        float(np.clip(smooth_y, 0, screen_height - 1)),
    )


def parse_calibration_grid(grid: str) -> tuple[int, int]:
    """Parse a grid string like '3x3' into (rows, cols)."""
    rows_s, cols_s = grid.lower().split("x")
    return int(rows_s), int(cols_s)


def calibration_point_hint(point_id: int, rows: int, cols: int) -> str:
    """Human-readable position for on-screen calibration help."""
    if point_id < 0 or point_id >= rows * cols:
        return f"dot {point_id}"
    row = point_id // cols
    col = point_id % cols
    vert = "top" if row == 0 else ("bottom" if row == rows - 1 else "middle")
    horiz = "left" if col == 0 else ("right" if col == cols - 1 else "center")
    if vert == "middle" and horiz == "center":
        region = "center"
    elif vert == "middle":
        region = horiz
    elif horiz == "center":
        region = vert
    else:
        region = f"{vert}-{horiz}"
    return f"dot {point_id} ({region}) — look at the white circle"


def draw_calibration_target(
    screen: pygame.Surface,
    target_x: int,
    target_y: int,
    point_id: int,
    rows: int,
    cols: int,
    collected: int,
    quota: int,
) -> None:
    """Draw dot, id label, and sample progress."""
    screen.fill((40, 40, 40))
    pygame.draw.circle(screen, (255, 255, 255), (target_x, target_y), 12)
    font = pygame.font.SysFont(None, 32)
    hint = calibration_point_hint(point_id, rows, cols)
    label = font.render(hint, True, (220, 220, 220))
    screen.blit(label, (24, 24))
    progress = font.render(f"samples {collected}/{quota}", True, (180, 180, 180))
    screen.blit(progress, (24, 56))


def calibration_dot_positions(
    rows: int,
    cols: int,
    screen_width: int,
    screen_height: int,
    inset: float = 0.1,
) -> list[tuple[int, int]]:
    """Return screen pixel (x, y) at the center of each grid cell, inset from edges."""
    margin_x = screen_width * inset
    margin_y = screen_height * inset
    usable_w = screen_width - 2 * margin_x
    usable_h = screen_height - 2 * margin_y
    positions: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            screen_x = int(margin_x + usable_w * (col + 0.5) / cols)
            screen_y = int(margin_y + usable_h * (row + 0.5) / rows)
            positions.append((screen_x, screen_y))
    return positions


def detect_face_landmarks(frame: Any, face_mesh: FaceMesh) -> Any | None:
    """Run FaceMesh on a BGR frame; return landmarks or None if no face."""
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    return cast(Any, results).multi_face_landmarks


def pygame_escape_pressed() -> bool:
    """Return True if the user pressed Esc or closed the calibration window."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def draw_face_landmarks(frame: Any, multi_face_landmarks: Any | None) -> None:
    """Draw landmark dots on frame, or 'no face' text when landmarks are missing."""
    if multi_face_landmarks is None:
        cv2.putText(frame, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return
    for face_landmarks in multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)


def cmd_preview(config: AppConfig) -> int:
    """
    Parameters
    ----------
    config : AppConfig
        Uses config.camera (index, width, height, fps_target).

    Does
    ----
    Open the webcam, run face/eye landmarks on each frame, show a debug window.

    Returns
    -------
    int
        0 on clean exit, non-zero on error.
    """

    # Open the webcam using index/width/height from config.yaml.
    cap = open_camera(config.camera.index, config.camera.width, config.camera.height)
    # Create MediaPipe face detector once (expensive — don't put this inside the loop).
    # max_num_faces=1: only track one face. refine_landmarks=True: extra detail around eyes/iris.
    face_mesh = FaceMesh(max_num_faces=1, refine_landmarks=True)
    # try/finally guarantees cleanup even if something crashes mid-loop.
    try:
        # Loop forever — one pass = one frame (~30 times per second).
        while True:
            # Grab one frame from the webcam. ok=False means read failed → exit loop.
            ok, frame = cap.read()
            if not ok:
                break
            multi_face_landmarks = detect_face_landmarks(frame, face_mesh)
            draw_face_landmarks(frame, multi_face_landmarks)
            # Show the annotated frame in a window titled "preview".
            cv2.imshow("preview", frame)
            # waitKey(1) waits 1 ms and checks keyboard; if user pressed q, quit the loop.
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Always run these on exit — gives camera back to Windows and closes the window.
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
    # Tell the shell we exited successfully.
    return 0



def open_camera(device_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("could not open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_screen_size(mode: str) -> tuple[int, int]:
    """Return (width, height) in pixels for the configured screen mode."""
    if mode != "primary":
        raise ValueError(f"unsupported screen mode: {mode!r} (v1 supports primary only)")
    user32 = ctypes.windll.user32
    width = user32.GetSystemMetrics(0)   # SM_CXSCREEN — primary monitor width
    height = user32.GetSystemMetrics(1)  # SM_CYSCREEN — primary monitor height
    return width, height


def show_system_cursor() -> None:
    """Ensure the OS cursor is visible (pygame/calibration can hide it on Windows)."""
    ctypes.windll.user32.ShowCursor(True)


def restore_cursor_to_center(screen_width: int, screen_height: int) -> None:
    """Move cursor to screen center and force visible — recovery after run/calibrate."""
    center_x = screen_width // 2
    center_y = screen_height // 2
    ctypes.windll.user32.SetCursorPos(center_x, center_y)
    show_system_cursor()


def cmd_calibrate(config: AppConfig, profile: str, *, append: bool = False) -> int:
    """
    Parameters
    ----------
    config : AppConfig
        Uses config.calibration (grid, samples_per_point, dwell_ms_to_start).
    profile : str
        Saves to data/calibrations/{profile}.npz.

    Does
    ----
    Show calibration dots on screen, record (features, screen_x, screen_y) samples, save dataset.

    Returns
    -------
    int
        0 if saved, non-zero if cancelled or failed.
    """
    # Build output path under project data/calibrations/{profile}.npz.
    output_path = project_root() / "data" / "calibrations" / f"{profile}.npz"
    # Create parent folders if they do not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Parse grid string (e.g. "3x3") into row and column counts.
    rows, cols = parse_calibration_grid(config.calibration.grid)
    # Read primary monitor pixel size for placing calibration dots.
    screen_width, screen_height = get_screen_size(config.screen.mode)
    # Compute screen pixel center for each grid dot (10% inset from edges).
    dot_positions = calibration_dot_positions(rows, cols, screen_width, screen_height)
    val_ids = val_point_ids_for_grid(rows, cols)
    est_min = estimate_calibration_minutes(
        rows,
        cols,
        config.calibration.samples_per_point,
        config.calibration.dwell_ms_to_start,
        fps=float(config.camera.fps_target),
    )
    print(
        f"calibration: {config.calibration.grid}  {config.calibration.samples_per_point} samples/dot  "
        f"~{est_min:.1f} min per pass  val hold-out dots {val_ids}"
    )
    if append:
        print("append mode: new samples will be added to existing .npz (same grid & features)")
    else:
        print("tip: run again with --append to double data without replacing the file")

    # Lists that will become numpy arrays when we save the dataset.
    feature_rows: list[np.ndarray] = []
    crop_rows: list[np.ndarray] = []
    label_rows: list[list[int]] = []
    point_id_rows: list[int] = []
    crop_size = config.model.crop_size
    # Start pygame for fullscreen calibration targets.
    pygame.init()
    # Fullscreen surface covering the primary monitor.
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    # Create MediaPipe face detector once for the whole calibration run.
    face_mesh = FaceMesh(max_num_faces=1, refine_landmarks=True)
    # Open webcam with the same settings used in preview.
    cap = open_camera(config.camera.index, config.camera.width, config.camera.height)
    try:
        # Visit each calibration dot in order (row-major grid).
        for point_id, (target_x, target_y) in enumerate(dot_positions):
            collected = 0
            quota = config.calibration.samples_per_point
            draw_calibration_target(
                screen, target_x, target_y, point_id, rows, cols, collected, quota
            )
            pygame.display.flip()
            dwell_deadline = time.perf_counter() + config.calibration.dwell_ms_to_start / 1000.0
            while time.perf_counter() < dwell_deadline:
                if pygame_escape_pressed():
                    return 1
                time.sleep(0.01)
            while collected < quota:
                if pygame_escape_pressed():
                    return 1
                draw_calibration_target(
                    screen, target_x, target_y, point_id, rows, cols, collected, quota
                )
                pygame.display.flip()
                # Grab one frame from the webcam.
                ok, frame = cap.read()
                # Camera failure is treated as a hard error for calibration.
                if not ok:
                    return 1
                # Run face mesh on this frame.
                multi_face_landmarks = detect_face_landmarks(frame, face_mesh)
                # Skip frames with no detected face (do not count toward quota).
                if multi_face_landmarks is None:
                    continue
                vec = extract_feature_vector(multi_face_landmarks)
                if vec is None:
                    continue
                crop = extract_eye_crop(frame, multi_face_landmarks, crop_size=crop_size)
                if crop is None:
                    continue
                feature_rows.append(vec)
                crop_rows.append(crop)
                label_rows.append([target_x, target_y])
                point_id_rows.append(point_id)
                collected += 1
    finally:
        # Always release camera and MediaPipe resources.
        cap.release()
        face_mesh.close()
        # Shut down pygame display and subsystems.
        pygame.quit()
        screen_width, screen_height = get_screen_size(config.screen.mode)
        restore_cursor_to_center(screen_width, screen_height)
    # Convert collected rows to numpy arrays for training.
    x_array = np.asarray(feature_rows, dtype=np.float32)
    images_array = np.asarray(crop_rows, dtype=np.uint8)
    y_array = np.asarray(label_rows, dtype=np.float32)
    point_ids_array = np.asarray(point_id_rows, dtype=np.int32)

    if append:
        merged = append_calibration_arrays(
            output_path,
            x_array,
            images_array,
            y_array,
            point_ids_array,
            rows=rows,
            cols=cols,
            feature_version=FEATURE_VERSION,
            crop_size=crop_size,
        )
        if merged is None:
            return 1
        x_array, images_array, y_array, point_ids_array = merged

    np.savez(
        output_path,
        X=x_array,
        images=images_array,
        y=y_array,
        point_ids=point_ids_array,
        feature_version=np.int32(FEATURE_VERSION),
        crop_size=np.int32(crop_size),
        grid_rows=np.int32(rows),
        grid_cols=np.int32(cols),
    )
    print(
        f"saved {x_array.shape[0]} samples to {output_path} "
        f"(features {x_array.shape}, eye crops {images_array.shape})"
    )
    # Successful calibration run.
    return 0


def cmd_train(config: AppConfig, profile: str, model: str, *, train_final: bool = False) -> int:
    """
    Parameters
    ----------
    config : AppConfig
        Uses config.model for MLP hyperparameters.
    profile : str
        Loads data/calibrations/{profile}.npz, saves to models/{profile}/.
    model : str
        "ridge" or "mlp".

    Does
    ----
    Train regressor on calibration data, print validation pixel error, save weights.

    Returns
    -------
    int
        0 if trained and saved, non-zero on missing data or error.
    """
    data_path = calibration_data_path(profile)
    if not data_path.is_file():
        print(f"calibration not found: {data_path}")
        return 1
    archive = np.load(data_path)
    X = archive["X"]
    y = archive["y"]
    point_ids = archive["point_ids"]
    saved_version = int(archive["feature_version"]) if "feature_version" in archive else None
    val_point_ids = default_val_point_ids(config.calibration.grid)
    train_X, train_y, val_X, val_y = split_train_val(X, y, point_ids, val_point_ids)
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X_scaled = np.asarray(scaler.transform(train_X), dtype=np.float64)
    val_X_scaled = np.asarray(scaler.transform(val_X), dtype=np.float64)
    expected_dim = feature_dim()
    if train_X.shape[1] != expected_dim:
        print(
            f"warning: calibration features have dim {train_X.shape[1]}, "
            f"expected {expected_dim} for FEATURE_VERSION {FEATURE_VERSION} — recalibrate"
        )
    elif saved_version is not None and saved_version != FEATURE_VERSION:
        print(
            f"warning: .npz feature_version={saved_version} but code uses {FEATURE_VERSION} — recalibrate"
        )
    print(f"features: version {FEATURE_VERSION}, dim {expected_dim}")

    if model == "ridge":
        regressor: Any = RidgeCV(alphas=np.logspace(-1, 4, 40))
        regressor.fit(train_X_scaled, train_y)
        train_pred = regressor.predict(train_X_scaled)
        val_pred = regressor.predict(val_X_scaled)
        train_error_px = mean_pixel_error(train_y, train_pred)
        val_error_px = mean_pixel_error(val_y, val_pred)
        artifact_path = save_gaze_artifact(profile, regressor, scaler, model, val_point_ids)
        checkpoint_note = ""
    elif model == "mlp":
        result, y_scaler = train_gaze_mlp(
            train_X_scaled, train_y, val_X_scaled, val_y, config.model
        )
        train_error_px = result.train_error_px
        val_error_px = result.val_error_px
        ckpt_path = mlp_checkpoint_path(profile)
        save_mlp_checkpoint(
            ckpt_path,
            result.predictor,
            input_dim=int(train_X.shape[1]),
            hidden=config.model.hidden,
            best_epoch=result.best_epoch,
            device=result.device,
        )
        artifact_path = save_gaze_artifact(
            profile, result.predictor, scaler, model, val_point_ids
        )
        checkpoint_note = f"\nsaved checkpoint to {ckpt_path} (best epoch {result.best_epoch})"
    elif model == "cnn":
        if "images" not in archive:
            print(
                "calibration has no eye crops. re-run: python -m gaze_mouse calibrate --profile "
                f"{profile}"
            )
            return 1
        images = archive["images"]
        val_mask = np.isin(point_ids, val_point_ids)
        train_images = images[~val_mask]
        val_images = images[val_mask]
        train_X_cnn = X[~val_mask]
        val_X_cnn = X[val_mask]
        backbone = config.model.backbone
        print(
            f"CNN+fusion ({backbone}): {train_images.shape[0]} train, {val_images.shape[0]} val, "
            f"crops {config.model.crop_size}x{config.model.crop_size}, features dim {train_X_cnn.shape[1]}"
            + ("  --final: deploy retrain on all dots after holdout tuning" if train_final else "")
        )
        cnn_result: CnnTrainResult = train_gaze_cnn(
            train_images,
            train_X_cnn,
            train_y,
            val_images,
            val_X_cnn,
            val_y,
            config.model,
            backbone=backbone,
            train_final=train_final,
        )
        train_error_px = cnn_result.train_error_px
        val_error_px = cnn_result.val_error_px
        ckpt_path = cnn_checkpoint_path(profile)
        save_cnn_checkpoint(
            ckpt_path,
            cnn_result.predictor,
            crop_size=config.model.crop_size,
            best_epoch=cnn_result.best_epoch,
            device=cnn_result.device,
        )
        artifact_path = save_gaze_artifact(
            profile, cnn_result.predictor, scaler, model, val_point_ids
        )
        checkpoint_note = f"\nsaved checkpoint to {ckpt_path} (best epoch {cnn_result.best_epoch})"
    else:
        raise ValueError(f"unknown model: {model}")

    print(f"train mean pixel error: {train_error_px:.1f} px")
    print(f"val mean pixel error: {val_error_px:.1f} px (held-out points {val_point_ids})")
    if val_error_px > 100:
        print("warning: val error > 100 px — improve model/calibration before relying on run")
    elif val_error_px <= 50:
        print("val error within stretch target (<= 50 px)")
    print(f"saved model to {artifact_path}{checkpoint_note}")
    return 0


def cmd_eval(config: AppConfig, profile: str) -> int:
    """
    Parameters
    ----------
    config : AppConfig
        Kept for consistent CLI signature.
    profile : str
        Which dataset and model under models/{profile}/ to evaluate.

    Does
    ----
    Print held-out calibration point pixel error for the trained model.

    Returns
    -------
    int
        0 after printing metrics, non-zero if files missing.
    """
    data_path = calibration_data_path(profile)
    if not data_path.is_file():
        print(f"calibration not found: {data_path}")
        return 1
    try:
        bundle = load_gaze_artifact(profile)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return 1
    regressor = bundle["model"]
    scaler = bundle["scaler"]
    model_type = str(bundle.get("model_type", "unknown"))
    saved_val_ids = tuple(int(x) for x in bundle.get("val_point_ids", []))
    val_point_ids = saved_val_ids or default_val_point_ids(config.calibration.grid)
    archive = np.load(data_path)
    y = archive["y"]
    point_ids = archive["point_ids"]
    val_mask = np.isin(point_ids, val_point_ids)
    train_y = y[~val_mask]
    val_y = y[val_mask]

    if model_type == "cnn":
        if "images" not in archive:
            print("calibration missing eye crops — recalibrate first")
            return 1
        images = archive["images"]
        X_all = archive["X"]
        train_pred = regressor.predict_batch(images[~val_mask], X_all[~val_mask])
        val_pred = regressor.predict_batch(images[val_mask], X_all[val_mask])
        eval_images = images
        eval_features = X_all
    else:
        eval_features = None
        X = archive["X"]
        train_X, _, val_X, _ = split_train_val(X, y, point_ids, val_point_ids)
        train_X_scaled = np.asarray(scaler.transform(train_X), dtype=np.float64)
        val_X_scaled = np.asarray(scaler.transform(val_X), dtype=np.float64)
        train_pred = regressor.predict(train_X_scaled)
        val_pred = regressor.predict(val_X_scaled)
        eval_images = None

    train_error_px = mean_pixel_error(train_y, train_pred)
    val_error_px = mean_pixel_error(val_y, val_pred)
    print(f"model: {model_type}  profile: {profile}")
    print(f"train mean pixel error: {train_error_px:.1f} px")
    print(f"val mean pixel error: {val_error_px:.1f} px (held-out points {val_point_ids})")
    for point_id in sorted(set(int(p) for p in val_point_ids)):
        mask = point_ids == point_id
        if not np.any(mask):
            continue
        if model_type == "cnn":
            assert eval_images is not None
            assert eval_features is not None
            point_pred = regressor.predict_batch(
                eval_images[mask], eval_features[mask]
            )
        else:
            point_X = np.asarray(scaler.transform(archive["X"][mask]), dtype=np.float64)
            point_pred = regressor.predict(point_X)
        point_error = mean_pixel_error(y[mask], point_pred)
        print(f"  point {point_id}: {point_error:.1f} px ({int(np.sum(mask))} samples)")
    if val_error_px > 100:
        print("warning: val error > 100 px — tune calibration/model before daily use")
    elif val_error_px <= 50:
        print("val error within stretch target (<= 50 px)")
    return 0


def cmd_run(config: AppConfig, profile: str) -> int:
    """
    Parameters
    ----------
    config : AppConfig
        Uses config.gaze, config.safety, config.camera.
    profile : str
        Loads trained model from models/{profile}/.

    Does
    ----
    Live loop: webcam → landmarks → predict → smooth → move system mouse. Esc stops control.

    Returns
    -------
    int
        0 on clean exit, non-zero on startup error.
    """
    try:
        bundle = load_gaze_artifact(profile)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return 1
    regressor = bundle["model"]
    scaler = bundle["scaler"]
    model_type = str(bundle.get("model_type", "ridge"))
    crop_size = int(bundle.get("crop_size", config.model.crop_size))
    screen_width, screen_height = get_screen_size(config.screen.mode)
    show_system_cursor()
    mouse_controller = mouse.Controller()
    start_pos = mouse_controller.position
    smooth_x, smooth_y = float(start_pos[0]), float(start_pos[1])
    killed = False
    warmup_frames = 15
    stable_face_frames = 0
    control_enabled = False

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        nonlocal killed
        if key == keyboard.Key.esc:
            killed = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print(
        f"gaze-mouse run ({model_type}) on {screen_width}x{screen_height}. "
        "Keep face in frame. Esc or q to stop. Cursor starts where it is now."
    )
    face_mesh = FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = open_camera(config.camera.index, config.camera.width, config.camera.height)
    face_lost_streak = 0
    smoothed_features: np.ndarray | None = None
    stabilizer = GazeStabilizer(
        prediction_alpha=config.gaze.prediction_alpha,
        outlier_px=config.gaze.prediction_outlier_px,
    )
    frame_interval_s = 1.0 / max(config.camera.fps_target, 1)
    feat_alpha = config.gaze.feature_smoothing_alpha
    try:
        while not killed:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                return 1
            multi_face_landmarks = detect_face_landmarks(frame, face_mesh)
            if multi_face_landmarks is None:
                face_lost_streak += 1
                smoothed_features = None
                stabilizer.reset()
                if control_enabled and face_lost_streak >= config.safety.face_lost_frames:
                    continue
            else:
                face_lost_streak = 0
                if not control_enabled:
                    stable_face_frames += 1
                    if stable_face_frames < warmup_frames:
                        if stable_face_frames in (5, 10, 15) or stable_face_frames == warmup_frames - 1:
                            print(f"warmup: {stable_face_frames}/{warmup_frames} (keep face in frame)")
                        continue
                    control_enabled = True
                    print("face stable — cursor control enabled.")
                if model_type == "cnn":
                    crop = extract_eye_crop(
                        frame, multi_face_landmarks, crop_size=crop_size
                    )
                    features = extract_feature_vector(multi_face_landmarks)
                    if crop is None or features is None:
                        continue
                    if feat_alpha > 0:
                        if smoothed_features is None:
                            smoothed_features = features.copy()
                        else:
                            smoothed_features = (
                                feat_alpha * features
                                + (1.0 - feat_alpha) * smoothed_features
                            ).astype(np.float32)
                        feat_for_pred = smoothed_features
                    else:
                        feat_for_pred = features
                    raw_x, raw_y = predict_screen_position(
                        regressor,
                        scaler,
                        model_type,
                        features=feat_for_pred,
                        crop_bgr=crop,
                    )
                else:
                    features = extract_feature_vector(multi_face_landmarks)
                    if features is None:
                        continue
                    if feat_alpha > 0:
                        if smoothed_features is None:
                            smoothed_features = features.copy()
                        else:
                            smoothed_features = (
                                feat_alpha * features
                                + (1.0 - feat_alpha) * smoothed_features
                            ).astype(np.float32)
                        feat_for_pred = smoothed_features
                    else:
                        feat_for_pred = features
                    raw_x, raw_y = predict_screen_position(
                        regressor, scaler, model_type, features=feat_for_pred
                    )
                if not (math.isfinite(raw_x) and math.isfinite(raw_y)):
                    continue
                if not (
                    -screen_width <= raw_x <= 2 * screen_width
                    and -screen_height <= raw_y <= 2 * screen_height
                ):
                    continue
                stable_x, stable_y = stabilizer.filter_prediction(raw_x, raw_y)
                smooth_x, smooth_y = apply_gaze_mapping(
                    stable_x,
                    stable_y,
                    screen_width,
                    screen_height,
                    config.gaze,
                    smooth_x,
                    smooth_y,
                )
                mouse_controller.position = (int(smooth_x), int(smooth_y))
            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval_s:
                time.sleep(frame_interval_s - elapsed)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        listener.stop()
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        restore_cursor_to_center(screen_width, screen_height)
    print("stopped — cursor moved to screen center.")
    return 0

def main(argv: list[str] | None = None) -> int:
    """
    Parameters
    ----------
    argv : list[str] | None
        CLI args; None uses sys.argv.

    Does
    ----
    Parse subcommands, load config.yaml, dispatch to cmd_* handler.

    Returns
    -------
    int
        Exit code for the shell (0 = success).
    """
    # Create the top-level CLI parser and program name shown in help output.
    parser = argparse.ArgumentParser(prog="gaze-mouse")
    # Add optional --config so user can point to a non-default YAML file.
    parser.add_argument("--config", type=str, default=None, help="custom config file path")
    # Create subcommand parser; required=True forces user to choose a command.
    sub = parser.add_subparsers(dest="command", required=True)
    # Register preview subcommand (no extra flags needed).
    sub.add_parser("preview")
    # Register calibrate subcommand parser object for adding calibrate-specific args.
    calibrate_parser = sub.add_parser("calibrate")
    # Register train subcommand parser object for adding train-specific args.
    train_parser = sub.add_parser("train")
    # Register eval subcommand parser object for adding eval-specific args.
    eval_parser = sub.add_parser("eval")
    # Register run subcommand parser object for adding run-specific args.
    run_parser = sub.add_parser("run")
    # Add optional --profile to calibrate so user can choose save/load profile name.
    calibrate_parser.add_argument("--profile", type=str, default=None)
    calibrate_parser.add_argument(
        "--append",
        action="store_true",
        help="add samples to existing .npz (same grid; run twice for ~2x data)",
    )
    # Add optional --profile to train so user can pick which calibration data to train on.
    train_parser.add_argument("--profile", type=str, default=None)
    # Add optional --profile to eval so user can pick which trained model profile to evaluate.
    eval_parser.add_argument("--profile", type=str, default=None)
    # Add optional --profile to run so user can pick which trained model profile to run live.
    run_parser.add_argument("--profile", type=str, default=None)
    # Add optional --model to train to choose algorithm; default comes from config if omitted.
    train_parser.add_argument(
        "--model", type=str, choices=["ridge", "mlp", "cnn"], default=None
    )
    train_parser.add_argument(
        "--final",
        action="store_true",
        help="CNN only: after holdout tuning, retrain on all grid points for run (corners included)",
    )
    # Parse CLI args (or passed argv for testing).
    args = parser.parse_args(argv)
    # If user passed --config, load that exact file path.
    if args.config:
        config = load_config(Path(args.config))
    # Otherwise load default config resolution (config.yaml or example fallback).
    else:
        config = load_config()
    # Use CLI --profile when present; otherwise fall back to profile in config.
    profile = args.profile if getattr(args, "profile", None) else config.profile

    # Dispatch preview command to webcam/landmark preview flow.
    if args.command == "preview":
        return cmd_preview(config)
    # Dispatch calibrate command to data collection flow.
    elif args.command == "calibrate":
        return cmd_calibrate(config, profile, append=args.append)
    # Dispatch train command to model training flow.
    elif args.command == "train":
        # Use CLI --model when provided; otherwise use model type from config.
        model = args.model if args.model else config.model.type
        return cmd_train(config, profile, model, train_final=bool(args.final))
    # Dispatch eval command to offline evaluation flow.
    elif args.command == "eval":
        return cmd_eval(config, profile)
    # Dispatch run command to live cursor control flow.
    elif args.command == "run":
        return cmd_run(config, profile)
    # Guard for impossible/unknown command values.
    else:
        raise ValueError(f"unknown command: {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())
