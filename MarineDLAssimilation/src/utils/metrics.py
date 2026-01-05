from __future__ import annotations

from typing import Dict, Optional

import torch


def _masked_mean(x: torch.Tensor, w: torch.Tensor, dim) -> torch.Tensor:
    """masked mean: sum(w*x)/sum(w), with safe denom."""
    num = (x * w).sum(dim=dim)
    den = w.sum(dim=dim).clamp(min=1.0)
    return num / den


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-channel RMSE over (B,H,W)."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(0, 2, 3)))


@torch.no_grad()
def rmse_masked(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-channel masked RMSE; weight must broadcast to (B,C,H,W)."""
    diff2 = (pred - target) ** 2
    mse_c = _masked_mean(diff2, weight, dim=(0, 2, 3))
    return torch.sqrt(torch.clamp(mse_c, min=0.0))


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target), dim=(0, 2, 3))


@torch.no_grad()
def mae_masked(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean(torch.abs(pred - target), weight, dim=(0, 2, 3))


@torch.no_grad()
def corrcoef(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation per channel, flattening (B,H,W)."""
    B, C, H, W = pred.shape
    p = pred.reshape(B, C, H * W)
    t = target.reshape(B, C, H * W)
    p = p.reshape(C, B * H * W)
    t = t.reshape(C, B * H * W)

    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).mean(dim=1)
    den = torch.sqrt((p * p).mean(dim=1) * (t * t).mean(dim=1) + 1e-12)
    return num / den


def metrics_dict(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Dict[str, list]:
    if weight is None:
        return {
            "rmse": rmse(pred, target).detach().cpu().tolist(),
            "mae": mae(pred, target).detach().cpu().tolist(),
            "corr": corrcoef(pred, target).detach().cpu().tolist(),
        }
    return {
        "rmse": rmse_masked(pred, target, weight).detach().cpu().tolist(),
        "mae": mae_masked(pred, target, weight).detach().cpu().tolist(),
        # corrcoef 未做mask版（本科毕设够用）；你也可以按weight做加权相关
        "corr": corrcoef(pred, target).detach().cpu().tolist(),
    }
