"""CLI entry point for gaze-mouse."""

import argparse
import ctypes
import time
from typing import Any, cast

import cv2
import numpy as np
import pygame
from mediapipe.python.solutions.face_mesh import FaceMesh
from pathlib import Path

from gaze_mouse.config import AppConfig, load_config, project_root


def parse_calibration_grid(grid: str) -> tuple[int, int]:
    """Parse a grid string like '3x3' into (rows, cols)."""
    rows_s, cols_s = grid.lower().split("x")
    return int(rows_s), int(cols_s)


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


def landmarks_to_feature_vector(multi_face_landmarks: Any) -> list[float]:
    """Flatten the first face's landmark x/y into one numeric row for training."""
    landmarks = multi_face_landmarks[0].landmark
    features: list[float] = []
    for landmark in landmarks:
        features.append(landmark.x)
        features.append(landmark.y)
    return features


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


def cmd_calibrate(config: AppConfig, profile: str) -> int:
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
    # Lists that will become numpy arrays when we save the dataset.
    feature_rows: list[list[float]] = []
    label_rows: list[list[int]] = []
    point_id_rows: list[int] = []
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
            # Fill background dark gray for this target.
            screen.fill((40, 40, 40))
            # Draw the white dot where the user should look.
            pygame.draw.circle(screen, (255, 255, 255), (target_x, target_y), 12)
            # Push the frame to the display.
            pygame.display.flip()
            # Wait until dwell time passes so the user's gaze settles on the dot.
            dwell_deadline = time.perf_counter() + config.calibration.dwell_ms_to_start / 1000.0
            while time.perf_counter() < dwell_deadline:
                # Esc during dwell cancels calibration without saving.
                if pygame_escape_pressed():
                    return 1
                # Small sleep so we do not busy-spin the CPU.
                time.sleep(0.01)
            # Count how many valid face samples we have collected for this dot.
            collected = 0
            # Keep sampling until we reach samples_per_point for this dot.
            while collected < config.calibration.samples_per_point:
                # Esc during sampling cancels calibration without saving.
                if pygame_escape_pressed():
                    return 1
                # Keep the target dot visible while sampling.
                screen.fill((40, 40, 40))
                pygame.draw.circle(screen, (255, 255, 255), (target_x, target_y), 12)
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
                # Convert landmarks to a flat numeric feature vector.
                feature_rows.append(landmarks_to_feature_vector(multi_face_landmarks))
                # Label is the screen pixel the user was told to look at.
                label_rows.append([target_x, target_y])
                # Record which grid dot this sample belongs to.
                point_id_rows.append(point_id)
                # One more valid sample for this dot.
                collected += 1
    finally:
        # Always release camera and MediaPipe resources.
        cap.release()
        face_mesh.close()
        # Shut down pygame display and subsystems.
        pygame.quit()
    # Convert collected rows to numpy arrays for training.
    x_array = np.asarray(feature_rows, dtype=np.float32)
    y_array = np.asarray(label_rows, dtype=np.float32)
    point_ids_array = np.asarray(point_id_rows, dtype=np.int32)
    # Save dataset to disk (.npz = compressed numpy archive).
    np.savez(output_path, X=x_array, y=y_array, point_ids=point_ids_array)
    # Tell the user where data was written and how many rows were saved.
    print(f"saved {len(feature_rows)} samples to {output_path}")
    # Successful calibration run.
    return 0


def cmd_train(config: AppConfig, profile: str, model: str) -> int:
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
    # --- Phase 3–4: train (not started) ---
    #
    # Goal: learn a mapping from face feature vectors → screen (x, y).
    #   "Ridge" = simple sklearn linear model (baseline, fast).
    #   "MLP" = small neural network in PyTorch (what you'll use for real control).
    #
    # TODO 1 — Load data/calibrations/{profile}.npz with numpy.load.
    #   If file missing, print a message and return 1.
    #
    # TODO 2 — Split into train and validation sets BY CALIBRATION POINT:
    #   Hold out ~2 entire dots (e.g. point ids 2 and 7), not random individual frames.
    #   Why: consecutive frames at the same dot are nearly identical; random split would cheat.
    #
    # TODO 3 — If model == "ridge":
    #   Scale features (StandardScaler fit on train only), fit Ridge, predict on val,
    #   print average pixel distance between predicted and actual screen coords, save model file.
    #
    # TODO 4 — If model == "mlp":
    #   Same scaler/split. Train a small network (input = feature count, output = 2 for x,y).
    #   Save weights to models/{profile}/.
    #
    # Test: python -m gaze_mouse train --profile test1 --model ridge

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
    # --- Phase 4: eval (not started) ---
    #
    # Goal: report how accurate the trained model is on dots it did NOT train on.
    #
    # TODO 1 — Load the same .npz and saved model files that train produced.
    #
    # TODO 2 — Use the same validation split as train. Run predict on val features.
    #
    # TODO 3 — Print mean pixel error (average distance between predicted and true screen coords).
    #
    # Test: python -m gaze_mouse eval --profile test1

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
    # --- Phase 5: live mouse (not started) ---
    #
    # Goal: same loop as preview, but instead of showing a window, move the real mouse cursor.
    #
    # TODO 1 — Load the trained model + scaler for profile (same files train saved).
    #
    # TODO 2 — open_camera + FaceMesh loop. Each frame: features → model.predict → (x, y) pixels.
    #
    # TODO 3 — Apply config.gaze gain (scale movement), smooth between frames (reduce jitter),
    #   then set mouse position with pynput (library that moves the OS cursor).
    #
    # TODO 4 — Esc key sets a "killed" flag; stop moving mouse when flag is set.
    #   If face not detected for config.safety.face_lost_frames in a row, stop moving too.
    #
    # Test: python -m gaze_mouse run --profile test1

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
    # Add optional --profile to train so user can pick which calibration data to train on.
    train_parser.add_argument("--profile", type=str, default=None)
    # Add optional --profile to eval so user can pick which trained model profile to evaluate.
    eval_parser.add_argument("--profile", type=str, default=None)
    # Add optional --profile to run so user can pick which trained model profile to run live.
    run_parser.add_argument("--profile", type=str, default=None)
    # Add optional --model to train to choose algorithm; default comes from config if omitted.
    train_parser.add_argument("--model", type=str, choices=["ridge", "mlp"], default=None)
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
        return cmd_calibrate(config, profile)
    # Dispatch train command to model training flow.
    elif args.command == "train":
        # Use CLI --model when provided; otherwise use model type from config.
        model = args.model if args.model else config.model.type
        return cmd_train(config, profile, model)
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
