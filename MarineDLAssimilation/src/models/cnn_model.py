from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, padding=p),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, kernel_size=k, padding=p),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(cin, cout, kernel_size=1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class CNNModel(BaseModel):
    """
    纯CNN基线：多层卷积残差块，保持分辨率不变。
    """
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, depth: int = 6):
        super().__init__()
        layers = [nn.Conv2d(in_channels, width, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(depth):
            layers.append(ConvBlock(width, width))
        layers.append(nn.Conv2d(width, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
