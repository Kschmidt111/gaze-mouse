"""PyTorch device selection (CUDA when installed)."""

from __future__ import annotations

from typing import Any

import torch


def pick_device(*, warn_if_cpu: bool = False) -> torch.device:
    """Return cuda device if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if warn_if_cpu:
        version = torch.__version__
        hint = "cpu" in version.lower() or "+cpu" in version
        if hint:
            print(
                "GPU not used: PyTorch is CPU-only ({0}).\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu124\n"
                "Then re-run train. (cu124 fits RTX 40-series; use cu121 if that fails.)".format(
                    version
                )
            )
        else:
            print(
                "GPU not used: torch.cuda.is_available() is False "
                f"(torch {version}). Check NVIDIA drivers."
            )

    return torch.device("cpu")


def cuda_amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def grad_scaler(enabled: bool) -> Any:
    """AMP grad scaler (torch.amp stubs omit GradScaler; getattr avoids type-checker noise)."""
    factory = getattr(torch.amp, "GradScaler")
    return factory("cuda", enabled=enabled)


def autocast_cuda(enabled: bool) -> Any:
    factory = getattr(torch.amp, "autocast")
    return factory("cuda", enabled=enabled)
