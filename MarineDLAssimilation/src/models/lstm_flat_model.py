from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel


class LSTMFlatModel(BaseModel):
    """方案A：把格点 patch 展平，直接喂给 LSTM。

    - 兼容当前工程的数据流：train/evaluate 中输入是 (B,C,H,W)。
    - 同时支持未来扩展到时间序列：如果输入是 (B,T,C,H,W)，就按时间维度跑 LSTM。

    输入:
        x: (B,C,H,W) 或 (B,T,C,H,W)
    输出:
        y_hat: (B,out_channels,H,W)
    注意:
        flatten 后的特征维度非常大，只适合作为“方法对比”的基线骨架；
        真正要做长序列/大区域，请考虑降维后再进序列模型（例如 CNN/UNet 编码后再 LSTM）。
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Lazy init because H,W not known until first forward
        self.lstm: nn.LSTM | None = None
        self.head: nn.Linear | None = None
        self._hw: tuple[int,int] | None = None

    def _ensure(self, H: int, W: int, device, dtype):
        if self.lstm is not None and self._hw == (H, W):
            return
        F_in = self.in_channels * H * W
        F_out = self.out_channels * H * W
        self.lstm = nn.LSTM(F_in, self.hidden_size, self.num_layers, batch_first=True).to(device=device, dtype=dtype)
        self.head = nn.Linear(self.hidden_size, F_out).to(device=device, dtype=dtype)
        self._hw = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,C,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected x as 4D/5D, got {x.shape}")

        B, T, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {C}")

        self._ensure(H, W, x.device, x.dtype)

        x_flat = x.reshape(B, T, -1)  # (B,T,F_in)
        out, _ = self.lstm(x_flat)
        h_last = out[:, -1, :]  # (B,Hid)
        y_flat = self.head(h_last)  # (B,F_out)
        return y_flat.reshape(B, self.out_channels, H, W)
