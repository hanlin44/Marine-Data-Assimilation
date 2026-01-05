from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel


class CNNLSTMPINNModel(BaseModel):
    """CNN 编码 -> (latent) LSTM -> CNN 解码

    说明：
    - 这是你毕设中“CNN-LSTM-PINNs”方法的网络结构骨架。
    - “PINNs”部分由训练脚本里的 physics_loss 控制（只要 model 名包含 'pinn' 就启用）。
    - 当前工程的数据是单日 patch，所以默认 T=1；未来如果你把数据集改成连续多日序列，
      输入改为 (B,T,C,H,W) 就能直接用。
    """
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

        self.lstm: nn.LSTM | None = None
        self.head: nn.Linear | None = None
        self._latent_hw: tuple[int,int] | None = None

    def _ensure(self, H: int, W: int, device, dtype):
        if self.lstm is not None and self._latent_hw == (H, W):
            return
        F = self.width * H * W
        self.lstm = nn.LSTM(F, self.hidden_size, self.num_layers, batch_first=True).to(device=device, dtype=dtype)
        self.head = nn.Linear(self.hidden_size, F).to(device=device, dtype=dtype)
        self._latent_hw = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,C,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected x as 4D/5D, got {x.shape}")

        B, T, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {C}")

        # encode each step
        z = []
        for t in range(T):
            zt = self.enc(x[:, t])  # (B,width,H,W)
            z.append(zt)
        z = torch.stack(z, dim=1)  # (B,T,width,H,W)

        self._ensure(H, W, x.device, x.dtype)
        zflat = z.reshape(B, T, -1)  # (B,T,F)
        out, _ = self.lstm(zflat)
        h_last = out[:, -1, :]
        z_last = self.head(h_last).reshape(B, self.width, H, W)
        return self.dec(z_last)
