from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            _norm(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            _norm(cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(cin, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if needed (odd size)
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy != 0 or dx != 0:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetModel(BaseModel):
    """
    U-Net：多尺度重建，适合“稀疏观测 → 全场重建”。
    """
    def __init__(self, in_channels: int, out_channels: int, base_width: int = 32):
        super().__init__()
        w1, w2, w3, w4 = base_width, base_width * 2, base_width * 4, base_width * 8

        self.inc = DoubleConv(in_channels, w1)
        self.down1 = Down(w1, w2)
        self.down2 = Down(w2, w3)
        self.down3 = Down(w3, w4)

        self.up1 = Up(w4, w3)
        self.up2 = Up(w3, w2)
        self.up3 = Up(w2, w1)

        self.outc = nn.Conv2d(w1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
