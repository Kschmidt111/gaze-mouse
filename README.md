# gaze-mouse

Webcam-based eye gaze control for Windows. Uses face/eye landmarks from MediaPipe, maps them to screen coordinates with a model trained on your own calibration data, and moves the system mouse cursor.

## Requirements

- Windows 10/11
- Python 3.11+
- Webcam (built-in or USB)
- Optional: NVIDIA GPU for faster MLP training

## Setup

```powershell
cd gaze-mouse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the example config and edit camera index if needed:

```powershell
copy config.example.yaml config.yaml
```

## Usage

**1. Calibrate** (collect samples at screen targets, saves under `data/calibrations/`):

```powershell
python -m gaze_mouse calibrate --profile default
```

**2. Train** (Ridge baseline, then MLP on GPU):

```powershell
python -m gaze_mouse train --profile default
```

**3. Run** (live cursor control; Esc kills control):

```powershell
python -m gaze_mouse run --profile default
```

## Project layout

```
src/gaze_mouse/     application code
data/calibrations/  your calibration datasets (gitignored)
models/             trained weights per profile (gitignored)
config.yaml         local settings (gitignored)
```

## Safety

- Press **Esc** anytime to stop cursor updates immediately.
- If your face is not detected, the cursor stops moving (no runaway drift).

## License

MIT (or your choice — add `LICENSE` before publishing.)
