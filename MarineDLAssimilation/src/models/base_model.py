from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    统一接口：forward(x) -> y_hat

    这里 x 默认是拼接后的输入张量：
        x = concat([xb, xobs, mask], dim=1)  # (B, Cin, H, W)
    输出 y_hat 形状与 y 一致：
        y_hat: (B, Cout, H, W)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def make_input_tensor(
    xb: torch.Tensor,
    xobs: torch.Tensor,
    mask: torch.Tensor,
    wobs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    xb: (B,C,H,W)
    xobs: (B,C,H,W)
    mask: (B,1,H,W) or (B,C,H,W)
    wobs: optional (B,1,H,W) or (B,C,H,W) 观测误差权重通道（例如 1/σ^2）。
    -> x: (B,2C+M(+W),H,W)
    """
    if mask.dim() != 4:
        raise ValueError("mask must be 4D tensor")
    if wobs is None:
        return torch.cat([xb, xobs, mask], dim=1)
    if wobs.dim() != 4:
        raise ValueError("wobs must be 4D tensor")
    return torch.cat([xb, xobs, mask, wobs], dim=1)
