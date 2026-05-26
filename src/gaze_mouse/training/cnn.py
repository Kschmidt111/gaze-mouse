"""PyTorch CNN on eye crops → screen position (GPU when available)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from gaze_mouse.config import ModelConfig
from gaze_mouse.training.metrics import mean_pixel_error
from gaze_mouse.training.device import autocast_cuda, cuda_amp_enabled, grad_scaler, pick_device
from gaze_mouse.training.mlp import _y_scaler_from_checkpoint, _y_scaler_to_lists

EARLY_STOPPING_PATIENCE = 35


class EyeCropDataset(Dataset):
    """BGR uint8 crops (H,W,3) and screen labels."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool = False,
    ) -> None:
        self.images = images
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[index].astype(np.float32) / 255.0
        if self.augment:
            img = _augment_crop(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.from_numpy(self.labels[index])
        return tensor, label


def _augment_crop(img: np.ndarray) -> np.ndarray:
    """Mild augmentation for small calibration sets."""
    out = img.copy()
    if np.random.rand() < 0.5:
        brightness = float(np.random.uniform(0.85, 1.15))
        out = np.clip(out * brightness, 0.0, 1.0)
    if np.random.rand() < 0.3:
        shift = np.random.uniform(-0.03, 0.03, size=2)
        out = np.roll(out, int(shift[0] * out.shape[0]), axis=0)
        out = np.roll(out, int(shift[1] * out.shape[1]), axis=1)
    return out


class GazeCNN(nn.Module):
    """Small CNN for eye crop → (screen_x, screen_y) in scaled label space."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class GazeCNNPredictor:
    """Load CNN + y scaler; predict from BGR crop."""

    def __init__(
        self,
        module: GazeCNN,
        device: torch.device,
        y_scaler: StandardScaler,
        crop_size: int,
    ) -> None:
        self.module = module
        self.device = device
        self.y_scaler = y_scaler
        self.crop_size = crop_size
        self.module.to(device)
        self.module.eval()

    def predict(self, crop_bgr: np.ndarray) -> np.ndarray:
        """One crop (H,W,3) uint8 BGR → (2,) screen pixels."""
        if crop_bgr.shape[:2] != (self.crop_size, self.crop_size):
            raise ValueError(f"expected crop {self.crop_size}x{self.crop_size}, got {crop_bgr.shape[:2]}")
        img = crop_bgr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scaled = self.module(tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(scaled)[0]

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """(N,H,W,3) uint8 → (N,2) screen coords."""
        batch = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            scaled = self.module(batch).cpu().numpy()
        return self.y_scaler.inverse_transform(scaled)


@dataclass
class CnnTrainResult:
    predictor: GazeCNNPredictor
    train_error_px: float
    val_error_px: float
    best_epoch: int
    device: str


def train_gaze_cnn(
    train_images: np.ndarray,
    train_y: np.ndarray,
    val_images: np.ndarray,
    val_y: np.ndarray,
    model_config: ModelConfig,
) -> CnnTrainResult:
    """Train CNN with MSE on scaled labels; early stop on val pixel error."""
    device = pick_device(warn_if_cpu=True)
    use_amp = cuda_amp_enabled(device)

    y_scaler = StandardScaler()
    train_y_scaled = y_scaler.fit_transform(train_y)
    val_y_scaled = y_scaler.transform(val_y)

    module = GazeCNN().to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=model_config.learning_rate,
        weight_decay=5e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    loss_fn = nn.SmoothL1Loss()

    train_loader = DataLoader(
        EyeCropDataset(train_images, train_y_scaled, augment=True),
        batch_size=min(model_config.batch_size, len(train_images)),
        shuffle=True,
        drop_last=len(train_images) > model_config.batch_size,
    )
    val_tensor = (
        torch.from_numpy(val_images.astype(np.float32) / 255.0)
        .permute(0, 3, 1, 2)
        .to(device)
    )
    val_y_np = val_y.astype(np.float32)

    best_val_px = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    print(f"CNN training on {device}  amp={use_amp}  train={len(train_images)} val={len(val_images)}")

    scaler_amp = grad_scaler(use_amp)

    for epoch in range(1, model_config.epochs + 1):
        module.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(use_amp):
                pred = module(batch_x)
                loss = loss_fn(pred, batch_y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

        module.eval()
        with torch.no_grad():
            with autocast_cuda(use_amp):
                val_pred_scaled = module(val_tensor).cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_pred_scaled)
        val_px = mean_pixel_error(val_y_np, val_pred)
        scheduler.step(val_px)

        if val_px < best_val_px:
            best_val_px = val_px
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or stale_epochs == 0:
            predictor_tmp = GazeCNNPredictor(module, device, y_scaler, model_config.crop_size)
            train_pred = predictor_tmp.predict_batch(train_images)
            train_px = mean_pixel_error(train_y, train_pred)
            print(f"  epoch {epoch:4d}  train {train_px:.1f} px  val {val_px:.1f} px")

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"early stopping at epoch {epoch} (patience {EARLY_STOPPING_PATIENCE})")
            break

    if best_state is not None:
        module.load_state_dict(best_state)

    predictor = GazeCNNPredictor(module, device, y_scaler, model_config.crop_size)
    train_pred = predictor.predict_batch(train_images)
    val_pred = predictor.predict_batch(val_images)
    return CnnTrainResult(
        predictor=predictor,
        train_error_px=mean_pixel_error(train_y, train_pred),
        val_error_px=mean_pixel_error(val_y, val_pred),
        best_epoch=best_epoch,
        device=str(device),
    )


def save_cnn_checkpoint(
    path: Path,
    predictor: GazeCNNPredictor,
    *,
    crop_size: int,
    best_epoch: int,
    device: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_mean, y_scale = _y_scaler_to_lists(predictor.y_scaler)
    torch.save(
        {
            "state_dict": predictor.module.state_dict(),
            "crop_size": crop_size,
            "best_epoch": best_epoch,
            "device_trained": device,
            "y_scaler_mean": y_mean,
            "y_scaler_scale": y_scale,
        },
        path,
    )


def load_cnn_predictor(path: Path, device: torch.device | None = None) -> GazeCNNPredictor:
    if not path.is_file():
        raise FileNotFoundError(f"cnn checkpoint not found: {path}")
    dev = device or pick_device()
    payload = torch.load(path, map_location=dev, weights_only=True)
    crop_size = int(payload["crop_size"])
    module = GazeCNN()
    module.load_state_dict(payload["state_dict"])
    y_scaler = _y_scaler_from_checkpoint(payload)
    return GazeCNNPredictor(module, dev, y_scaler, crop_size)
