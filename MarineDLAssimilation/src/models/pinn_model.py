from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel
from .unet_model import UNetModel
from .cnn_model import CNNModel


class PINNModel(BaseModel):
    """
    PINNs 这里指：模型结构仍可用 U-Net/CNN，但训练时在 loss 中加入物理残差项。
    物理项实现在 utils/physics.py，训练脚本 train.py 会在 model==pinn 时启用。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        backbone: str = "unet",
        base_width: int = 32,
    ):
        super().__init__()
        backbone = backbone.lower()
        if backbone == "unet":
            self.net = UNetModel(in_channels, out_channels, base_width=base_width)
        elif backbone == "cnn":
            self.net = CNNModel(in_channels, out_channels, width=64, depth=6)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
