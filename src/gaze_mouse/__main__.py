"""CLI entry point for gaze-mouse."""

from typing import Any, cast
import cv2
from mediapipe.python.solutions.face_mesh import FaceMesh
from pathlib import Path

from gaze_mouse.config import AppConfig, load_config


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
            # OpenCV stores colors as BGR; MediaPipe expects RGB — convert before processing.
            rbg_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run face detection on this frame; results holds landmark data if a face was found.
            results = face_mesh.process(rbg_image)
            # cast() is only for the type checker — at runtime results.multi_face_landmarks works fine.
            multi_face_landmarks = cast(Any, results).multi_face_landmarks
            if multi_face_landmarks is None:
                # No face in frame — draw red "no face" text in top-left corner.
                cv2.putText(frame, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Loop over each detected face (we only allow 1, but MediaPipe returns a list).
                for face_landmarks in multi_face_landmarks:
                    # Each face has ~468 landmark points; loop over all of them.
                    for landmark in face_landmarks.landmark:
                        # landmark.x/y are 0.0–1.0 (fraction of image). Multiply by size to get pixels.
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        # Draw a small red dot at each landmark on the frame we're about to show.
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
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
    # --- Phase 2: calibration (not started) ---
    #
    # Goal: while you stare at dots on your monitor, record pairs of
    #   (face features from webcam, screen pixel where you were looking).
    #   Save them to a file for training later.
    #
    # TODO 1 — Build the output file path:
    #   data/calibrations/{profile}.npz  (create folders if missing).
    #   .npz is numpy's way of saving arrays to disk.
    #
    # TODO 2 — Parse config.calibration.grid (e.g. "3x3"):
    #   Split on "x", convert to ints → 3 rows, 3 cols → 9 dot positions total.
    #
    # TODO 3 — Get primary monitor size in pixels (Windows API via ctypes).
    #   Place dots in a grid inset ~10% from each edge so they are not in corners.
    #   Each target is an (screen_x, screen_y) int — where the dot appears on your monitor.
    #
    # TODO 4 — Reuse open_camera and FaceMesh from preview (same webcam pipeline).
    #
    # TODO 5 — For each target position:
    #   a. Show a fullscreen window with a white dot at (screen_x, screen_y) — use pygame.
    #   b. Wait config.calibration.dwell_ms_to_start ms so your eyes settle on the dot.
    #   c. Collect config.calibration.samples_per_point frames (e.g. 45):
    #        read frame → face_mesh → build feature vector (numbers describing your face/eyes).
    #        Skip frames where face not detected.
    #        Append feature row to list X, append [screen_x, screen_y] to list y.
    #   d. If user presses Esc → stop, return 1 (cancelled).
    #
    # TODO 6 — Save with numpy.savez: arrays X, y, point_ids (which dot each row came from).
    #
    # Test: python -m gaze_mouse calibrate --profile test1

    raise NotImplementedError("Phase 2")


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
    # --- Phase 0: wire up the CLI (you are here) ---
    #
    # Goal: when someone runs "python -m gaze_mouse preview" in terminal, call cmd_preview.
    #   argparse is Python's built-in library for parsing command-line flags and subcommands.
    #
    # TODO 1 — parser = argparse.ArgumentParser(prog="gaze-mouse")
    #   This creates the object that understands --help and subcommands.
    #
    # TODO 2 — parser.add_argument("--config", type=str, default=None, help="...")
    #   Lets user pass a custom config file path. Optional — default None means use load_config().
    #
    # TODO 3 — sub = parser.add_subparsers(dest="command", required=True)
    #   subcommands are the words after the program name: preview, calibrate, train, etc.
    #   required=True means user MUST pick one (no bare "python -m gaze_mouse").
    #   Add each: sub.add_parser("preview"), sub.add_parser("calibrate"), etc.
    #
    # TODO 4 — On calibrate/train/eval/run parsers, add:
    #   --profile (str, default None) — which saved dataset/model name to use.
    #   On train only, also add --model with choices ["ridge", "mlp"].
    #
    # TODO 5 — args = parser.parse_args(argv)
    #   If args.config is set, pass Path(args.config) to load_config. Else load_config().
    #   profile = args.profile if set, else config.profile (from yaml).
    #
    # TODO 6 — Dispatch with if/elif:
    #   if args.command == "preview": return cmd_preview(config)
    #   ... same for calibrate, train, eval, run.
    #   Each cmd_* returns an int exit code; return that from main().
    #
    # Test: python -m gaze_mouse --help
    #   Should list all five subcommands. No NotImplementedError when you run --help.

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
