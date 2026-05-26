"""PyTorch MLP for gaze → screen regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from gaze_mouse.config import ModelConfig
from gaze_mouse.training.metrics import mean_pixel_error

# Stop if validation error does not improve for this many epochs.
EARLY_STOPPING_PATIENCE = 30
DROPOUT = 0.35


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GazeMLP(nn.Module):
    """Fully connected network: features → (screen_x, screen_y)."""

    def __init__(self, input_dim: int, hidden: Sequence[int], dropout: float = DROPOUT) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GazeMLPPredictor:
    """Sklearn-like wrapper for eval, run, and saving."""

    def __init__(
        self,
        module: GazeMLP,
        device: torch.device,
        y_scaler: StandardScaler,
    ) -> None:
        self.module = module
        self.device = device
        self.y_scaler = y_scaler
        self.module.to(device)
        self.module.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict screen coords for scaled feature rows, shape (n, 2)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            scaled = self.module(tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(scaled)


@dataclass
class TrainResult:
    predictor: GazeMLPPredictor
    train_error_px: float
    val_error_px: float
    best_epoch: int
    device: str


def train_gaze_mlp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    model_config: ModelConfig,
) -> tuple[TrainResult, StandardScaler]:
    """Train MLP with MSE loss; early stopping on validation mean pixel error."""
    device = pick_device()
    input_dim = int(train_X.shape[1])
    hidden = tuple(int(h) for h in model_config.hidden)

    y_scaler = StandardScaler()
    train_y_scaled = y_scaler.fit_transform(train_y)
    val_y_scaled = y_scaler.transform(val_y)

    module = GazeMLP(input_dim, hidden).to(device)
    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=model_config.learning_rate,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_X.astype(np.float32)),
        torch.from_numpy(train_y_scaled.astype(np.float32)),
    )
    loader = DataLoader(
        train_ds,
        batch_size=min(model_config.batch_size, len(train_ds)),
        shuffle=True,
    )

    val_X_t = torch.from_numpy(val_X.astype(np.float32)).to(device)
    val_y_np = val_y.astype(np.float32)
    best_val_px = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    print(f"training on {device}  input_dim={input_dim}  hidden={hidden}")

    for epoch in range(1, model_config.epochs + 1):
        module.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = module(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

        module.eval()
        with torch.no_grad():
            val_pred_scaled = module(val_X_t).cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_pred_scaled)
        val_px = mean_pixel_error(val_y_np, val_pred)

        if val_px < best_val_px:
            best_val_px = val_px
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 20 == 0 or stale_epochs == 0:
            train_scaled = _predict_numpy(module, device, train_X)
            train_px = mean_pixel_error(train_y, y_scaler.inverse_transform(train_scaled))
            print(f"  epoch {epoch:4d}  train {train_px:.1f} px  val {val_px:.1f} px")

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"early stopping at epoch {epoch} (no val improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    if best_state is not None:
        module.load_state_dict(best_state)

    predictor = GazeMLPPredictor(module, device, y_scaler)
    train_pred = predictor.predict(train_X)
    val_pred = predictor.predict(val_X)
    result = TrainResult(
        predictor=predictor,
        train_error_px=mean_pixel_error(train_y, train_pred),
        val_error_px=mean_pixel_error(val_y, val_pred),
        best_epoch=best_epoch,
        device=str(device),
    )
    return result, y_scaler


def _predict_numpy(module: GazeMLP, device: torch.device, X: np.ndarray) -> np.ndarray:
    module.eval()
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32)).to(device)
        return module(t).cpu().numpy()


def _y_scaler_to_lists(scaler: StandardScaler) -> tuple[list[float], list[float]]:
    """Serialise a fitted y StandardScaler for torch.save (plain lists, no numpy globals)."""
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("y_scaler must be fit before saving checkpoint")
    mean_arr: np.ndarray = np.asarray(scaler.mean_, dtype=np.float64)
    scale_arr: np.ndarray = np.asarray(scaler.scale_, dtype=np.float64)
    return mean_arr.tolist(), scale_arr.tolist()


def save_mlp_checkpoint(
    path: Path,
    predictor: GazeMLPPredictor,
    *,
    input_dim: int,
    hidden: Sequence[int],
    best_epoch: int,
    device: str,
) -> None:
    """Save weights and metadata for load_mlp_predictor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y_mean, y_scale = _y_scaler_to_lists(predictor.y_scaler)
    torch.save(
        {
            "state_dict": predictor.module.state_dict(),
            "input_dim": input_dim,
            "hidden": list(hidden),
            "best_epoch": best_epoch,
            "device_trained": device,
            "y_scaler_mean": y_mean,
            "y_scaler_scale": y_scale,
        },
        path,
    )


def _y_scaler_from_checkpoint(payload: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(payload["y_scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(payload["y_scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = 2
    return scaler


def load_mlp_predictor(
    path: Path,
    device: torch.device | None = None,
    y_scaler: StandardScaler | None = None,
) -> GazeMLPPredictor:
    """Load MLP from mlp.pt; uses CPU weights then moves to current device."""
    if not path.is_file():
        raise FileNotFoundError(f"mlp checkpoint not found: {path}")
    dev = device or pick_device()
    payload = torch.load(path, map_location=dev, weights_only=True)
    input_dim = int(payload["input_dim"])
    hidden = tuple(int(h) for h in payload["hidden"])
    module = GazeMLP(input_dim, hidden)
    module.load_state_dict(payload["state_dict"])
    if y_scaler is None:
        if "y_scaler_mean" not in payload:
            raise RuntimeError(f"mlp checkpoint missing y_scaler (retrain): {path}")
        y_scaler = _y_scaler_from_checkpoint(payload)
    return GazeMLPPredictor(module, dev, y_scaler)
