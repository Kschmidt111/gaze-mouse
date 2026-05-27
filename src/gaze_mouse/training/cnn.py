"""PyTorch CNN / ResNet (+ landmark fusion) on eye crops → screen position."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from gaze_mouse.config import ModelConfig
from gaze_mouse.training.metrics import mean_pixel_error
from gaze_mouse.training.device import autocast_cuda, cuda_amp_enabled, grad_scaler, pick_device
from gaze_mouse.training.mlp import _y_scaler_from_checkpoint, _y_scaler_to_lists

EARLY_STOPPING_PATIENCE = 35
CNN_EMBED_DIM = 256
BACKBONE_CUSTOM = "custom"
BACKBONE_RESNET18 = "resnet18"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _x_scaler_from_checkpoint(payload: dict) -> StandardScaler:
    fake = {
        "y_scaler_mean": payload["x_scaler_mean"],
        "y_scaler_scale": payload["x_scaler_scale"],
    }
    return _y_scaler_from_checkpoint(fake)


def _normalize_backbone(name: str) -> str:
    if name in (BACKBONE_CUSTOM, BACKBONE_RESNET18):
        return name
    raise ValueError(f"unknown backbone {name!r} (use custom or resnet18)")


def _preprocess_image_tensor(tensor: torch.Tensor, backbone: str) -> torch.Tensor:
    """tensor: (3,H,W) or (N,3,H,W) float in [0,1]."""
    if backbone != BACKBONE_RESNET18:
        return tensor
    device = tensor.device
    if tensor.dim() == 3:
        mean = _IMAGENET_MEAN.squeeze(0).to(device)
        std = _IMAGENET_STD.squeeze(0).to(device)
    else:
        mean = _IMAGENET_MEAN.to(device)
        std = _IMAGENET_STD.to(device)
    return (tensor - mean) / std


class FusionEyeCropDataset(Dataset):
    """Eye crop + scaled landmark features + screen labels."""

    def __init__(
        self,
        images: np.ndarray,
        features_scaled: np.ndarray,
        labels: np.ndarray,
        *,
        backbone: str = BACKBONE_RESNET18,
        augment: bool = False,
    ) -> None:
        self.images = images
        self.features_scaled = features_scaled.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.backbone = backbone
        self.augment = augment

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.images[index].astype(np.float32) / 255.0
        if self.augment:
            img = _augment_crop(img)
        image_tensor = _preprocess_image_tensor(
            torch.from_numpy(img).permute(2, 0, 1), self.backbone
        )
        feat_tensor = torch.from_numpy(self.features_scaled[index])
        label = torch.from_numpy(self.labels[index])
        return image_tensor, feat_tensor, label


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


def _make_cnn_backbone() -> nn.Sequential:
    return nn.Sequential(
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


class GazeCNN(nn.Module):
    """Small CNN for eye crop → (screen_x, screen_y) in scaled label space."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _make_cnn_backbone()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_EMBED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class GazeFusionCNN(nn.Module):
    """Vision backbone + scaled landmark features → screen coords."""

    def __init__(self, feature_dim: int, *, backbone: str = BACKBONE_RESNET18) -> None:
        super().__init__()
        self.backbone_name = _normalize_backbone(backbone)
        if self.backbone_name == BACKBONE_RESNET18:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.embed_dim = 512
        else:
            self.backbone = _make_cnn_backbone()
            self.embed_dim = CNN_EMBED_DIM
        self.fusion_head = nn.Sequential(
            nn.Linear(self.embed_dim + feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x).flatten(1)
        return self.fusion_head(torch.cat([emb, feats], dim=1))


def _images_to_tensor_batch(
    images: np.ndarray, device: torch.device, backbone: str
) -> torch.Tensor:
    batch = (
        torch.from_numpy(images.astype(np.float32) / 255.0)
        .permute(0, 3, 1, 2)
        .to(device, non_blocking=True)
    )
    return _preprocess_image_tensor(batch, backbone)


def _predict_scaled_chunks(
    module: GazeCNN,
    images: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    use_amp: bool,
) -> np.ndarray:
    """Run CNN-only module on images in chunks."""
    module.eval()
    chunks: list[np.ndarray] = []
    step = max(1, batch_size)
    for start in range(0, len(images), step):
        batch = _images_to_tensor_batch(images[start : start + step], device, BACKBONE_CUSTOM)
        with torch.no_grad():
            with autocast_cuda(use_amp):
                chunks.append(module(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _predict_fusion_scaled_chunks(
    module: GazeFusionCNN,
    images: np.ndarray,
    features_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    use_amp: bool,
) -> np.ndarray:
    """Run fusion module on image + feature batches."""
    module.eval()
    backbone = module.backbone_name
    chunks: list[np.ndarray] = []
    step = max(1, batch_size)
    for start in range(0, len(images), step):
        batch_img = _images_to_tensor_batch(images[start : start + step], device, backbone)
        batch_feat = (
            torch.from_numpy(features_scaled[start : start + step].astype(np.float32))
            .to(device, non_blocking=True)
        )
        with torch.no_grad():
            with autocast_cuda(use_amp):
                chunks.append(module(batch_img, batch_feat).cpu().numpy())
    return np.concatenate(chunks, axis=0)


class GazeCNNPredictor:
    """CNN or CNN+landmark fusion; predict from BGR crop (+ features when fusion)."""

    def __init__(
        self,
        module: GazeCNN | GazeFusionCNN,
        device: torch.device,
        y_scaler: StandardScaler,
        crop_size: int,
        *,
        fusion: bool = False,
        x_scaler: StandardScaler | None = None,
        backbone: str = BACKBONE_RESNET18,
    ) -> None:
        self.module = module
        self.device = device
        self.y_scaler = y_scaler
        self.crop_size = crop_size
        self.fusion = fusion
        self.x_scaler = x_scaler
        if fusion:
            if x_scaler is None:
                raise ValueError("fusion predictor requires x_scaler")
            if isinstance(module, GazeFusionCNN):
                self.backbone = module.backbone_name
            else:
                self.backbone = _normalize_backbone(backbone)
        else:
            self.backbone = BACKBONE_CUSTOM
        self.module.to(device)
        self.module.eval()

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        if self.x_scaler is None:
            raise RuntimeError("x_scaler missing on fusion predictor")
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return np.asarray(self.x_scaler.transform(row), dtype=np.float32)

    def predict(
        self,
        crop_bgr: np.ndarray,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """One crop (H,W,3) uint8 BGR → (2,) screen pixels."""
        if crop_bgr.shape[:2] != (self.crop_size, self.crop_size):
            raise ValueError(
                f"expected crop {self.crop_size}x{self.crop_size}, got {crop_bgr.shape[:2]}"
            )
        img = crop_bgr.astype(np.float32) / 255.0
        tensor = _preprocess_image_tensor(
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0), self.backbone
        ).to(self.device)
        with torch.no_grad():
            if self.fusion:
                if features is None:
                    raise ValueError("fusion model requires features")
                feat = torch.from_numpy(self._scale_features(features)).to(self.device)
                scaled = self.module(tensor, feat).cpu().numpy()  # type: ignore[operator]
            else:
                if not isinstance(self.module, GazeCNN):
                    raise TypeError("expected GazeCNN module")
                scaled = self.module(tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(scaled)[0]

    def predict_batch(
        self,
        images: np.ndarray,
        features: np.ndarray | None = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """(N,H,W,3) uint8 → (N,2) screen coords."""
        use_amp = cuda_amp_enabled(self.device)
        if self.fusion:
            if features is None:
                raise ValueError("fusion model requires features")
            scaled = _predict_fusion_scaled_chunks(
                self.module,  # type: ignore[arg-type]
                images,
                np.asarray(self.x_scaler.transform(features), dtype=np.float32),  # type: ignore[union-attr]
                self.device,
                batch_size,
                use_amp=use_amp,
            )
        else:
            if not isinstance(self.module, GazeCNN):
                raise TypeError("expected GazeCNN module")
            scaled = _predict_scaled_chunks(
                self.module, images, self.device, batch_size, use_amp=use_amp
            )
        return self.y_scaler.inverse_transform(scaled)


@dataclass
class CnnTrainResult:
    predictor: GazeCNNPredictor
    train_error_px: float
    val_error_px: float
    best_epoch: int
    device: str


def _fusion_optimizer(module: GazeFusionCNN, learning_rate: float) -> torch.optim.AdamW:
    if module.backbone_name == BACKBONE_RESNET18:
        return torch.optim.AdamW(
            [
                {"params": module.backbone.parameters(), "lr": learning_rate * 0.1},
                {"params": module.fusion_head.parameters(), "lr": learning_rate},
            ],
            weight_decay=5e-3,
        )
    return torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=5e-3)


def _fit_fusion_model(
    train_images: np.ndarray,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_images: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    model_config: ModelConfig,
    *,
    backbone: str,
    max_epochs: int | None = None,
    early_stop: bool = True,
    label: str = "train",
) -> tuple[GazeFusionCNN, StandardScaler, StandardScaler, int]:
    """Train fusion model; return module, y_scaler, x_scaler, best_epoch."""
    device = pick_device(warn_if_cpu=True)
    use_amp = cuda_amp_enabled(device)
    backbone = _normalize_backbone(backbone)
    feature_dim = int(train_X.shape[1])

    y_scaler = StandardScaler()
    train_y_scaled = y_scaler.fit_transform(train_y)
    y_scaler.transform(val_y)

    x_scaler = StandardScaler()
    train_X_scaled = np.asarray(x_scaler.fit_transform(train_X), dtype=np.float32)
    val_X_scaled = np.asarray(x_scaler.transform(val_X), dtype=np.float32)

    module = GazeFusionCNN(feature_dim, backbone=backbone).to(device)
    optimizer = _fusion_optimizer(module, model_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    loss_fn = nn.SmoothL1Loss()

    epochs = max_epochs if max_epochs is not None else model_config.epochs
    train_loader = DataLoader(
        FusionEyeCropDataset(
            train_images, train_X_scaled, train_y_scaled, backbone=backbone, augment=True
        ),
        batch_size=min(model_config.batch_size, len(train_images)),
        shuffle=True,
        drop_last=len(train_images) > model_config.batch_size,
    )
    infer_bs = min(model_config.batch_size, 64)
    val_y_np = val_y.astype(np.float32)

    best_val_px = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    print(
        f"{label}: {backbone}+fusion on {device}  amp={use_amp}  "
        f"train={len(train_images)} val={len(val_images)}  feature_dim={feature_dim}"
    )

    scaler_amp = grad_scaler(use_amp)

    for epoch in range(1, epochs + 1):
        module.train()
        for batch_x, batch_feat, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_feat = batch_feat.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(use_amp):
                pred = module(batch_x, batch_feat)
                loss = loss_fn(pred, batch_y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

        val_pred_scaled = _predict_fusion_scaled_chunks(
            module, val_images, val_X_scaled, device, infer_bs, use_amp=use_amp
        )
        val_pred = y_scaler.inverse_transform(val_pred_scaled)
        val_px = mean_pixel_error(val_y_np, val_pred)
        if early_stop:
            scheduler.step(val_px)

        improved = val_px < best_val_px
        if improved:
            best_val_px = val_px
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            train_scaled = _predict_fusion_scaled_chunks(
                module, train_images, train_X_scaled, device, infer_bs, use_amp=use_amp
            )
            train_px = mean_pixel_error(train_y, y_scaler.inverse_transform(train_scaled))
            print(f"  epoch {epoch:4d}  train {train_px:.1f} px  val {val_px:.1f} px")

        if early_stop and stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"early stopping at epoch {epoch} (patience {EARLY_STOPPING_PATIENCE})")
            break

    if best_state is not None:
        module.load_state_dict(best_state)

    return module, y_scaler, x_scaler, best_epoch


def train_gaze_cnn(
    train_images: np.ndarray,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_images: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    model_config: ModelConfig,
    *,
    backbone: str = BACKBONE_RESNET18,
    train_final: bool = False,
) -> CnnTrainResult:
    """Train fusion model; optional second phase on all calibration points for deploy."""
    backbone = _normalize_backbone(backbone)
    device = pick_device(warn_if_cpu=True)

    module, y_scaler, x_scaler, best_epoch = _fit_fusion_model(
        train_images,
        train_X,
        train_y,
        val_images,
        val_X,
        val_y,
        model_config,
        backbone=backbone,
        label="phase 1 (holdout val)",
    )

    infer_bs = min(model_config.batch_size, 64)
    val_mask_note = ""

    if train_final:
        all_images = np.concatenate([train_images, val_images], axis=0)
        all_X = np.concatenate([train_X, val_X], axis=0)
        all_y = np.concatenate([train_y, val_y], axis=0)
        print(
            f"\nphase 2 (--final): retrain on all {len(all_images)} samples "
            f"for {best_epoch} epochs (corners included for run)"
        )
        module, y_scaler, x_scaler, _ = _fit_fusion_model(
            all_images,
            all_X,
            all_y,
            val_images,
            val_X,
            val_y,
            model_config,
            backbone=backbone,
            max_epochs=best_epoch,
            early_stop=False,
            label="phase 2 (deploy)",
        )
        val_mask_note = "  [deploy model; val below is honest held-out eval]"

    predictor = GazeCNNPredictor(
        module,
        device,
        y_scaler,
        model_config.crop_size,
        fusion=True,
        x_scaler=x_scaler,
        backbone=backbone,
    )
    train_pred = predictor.predict_batch(train_images, train_X, batch_size=infer_bs)
    val_pred = predictor.predict_batch(val_images, val_X, batch_size=infer_bs)
    result = CnnTrainResult(
        predictor=predictor,
        train_error_px=mean_pixel_error(train_y, train_pred),
        val_error_px=mean_pixel_error(val_y, val_pred),
        best_epoch=best_epoch,
        device=str(device),
    )
    if val_mask_note:
        print(val_mask_note.strip())
    return result


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
    payload: dict = {
        "state_dict": predictor.module.state_dict(),
        "crop_size": crop_size,
        "best_epoch": best_epoch,
        "device_trained": device,
        "y_scaler_mean": y_mean,
        "y_scaler_scale": y_scale,
        "fusion": predictor.fusion,
        "backbone": predictor.backbone,
    }
    if predictor.fusion and predictor.x_scaler is not None:
        x_mean, x_scale = _y_scaler_to_lists(predictor.x_scaler)
        payload["feature_dim"] = int(predictor.x_scaler.n_features_in_ or len(x_mean))
        payload["x_scaler_mean"] = x_mean
        payload["x_scaler_scale"] = x_scale
    torch.save(payload, path)


def load_cnn_predictor(
    path: Path,
    device: torch.device | None = None,
    *,
    x_scaler: StandardScaler | None = None,
) -> GazeCNNPredictor:
    if not path.is_file():
        raise FileNotFoundError(f"cnn checkpoint not found: {path}")
    dev = device or pick_device()
    payload = torch.load(path, map_location=dev, weights_only=True)
    crop_size = int(payload["crop_size"])
    y_scaler = _y_scaler_from_checkpoint(payload)
    fusion = bool(payload.get("fusion", False))
    backbone = str(payload.get("backbone", BACKBONE_CUSTOM))

    if fusion:
        feature_dim = int(payload["feature_dim"])
        module: GazeCNN | GazeFusionCNN = GazeFusionCNN(feature_dim, backbone=backbone)
        module.load_state_dict(payload["state_dict"])
        feat_scaler = x_scaler if x_scaler is not None else _x_scaler_from_checkpoint(payload)
        return GazeCNNPredictor(
            module,
            dev,
            y_scaler,
            crop_size,
            fusion=True,
            x_scaler=feat_scaler,
            backbone=backbone,
        )

    module = GazeCNN()
    module.load_state_dict(payload["state_dict"])
    return GazeCNNPredictor(module, dev, y_scaler, crop_size, fusion=False)
