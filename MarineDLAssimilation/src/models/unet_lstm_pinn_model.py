from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel
from .unet_model import DoubleConv, Down, Up


class UNetLSTMPINNModel(BaseModel):
    """U-Net 编码器瓶颈处加入 LSTM（flatten bottleneck），再解码。

    - 适用于“稀疏观测 + 背景场 → 全场重建”，同时通过 physics_loss 施加物理一致性（PINNs）。
    - 当前工程默认 T=1（单日 patch），未来改成多日序列输入 (B,T,C,H,W) 也能直接用。
    """
    def __init__(self, in_channels: int, out_channels: int, base_width: int = 32, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_width = base_width
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        w1, w2, w3, w4 = base_width, base_width * 2, base_width * 4, base_width * 8

        self.inc = DoubleConv(in_channels, w1)
        self.down1 = Down(w1, w2)
        self.down2 = Down(w2, w3)
        self.down3 = Down(w3, w4)

        self.up1 = Up(w4, w3)
        self.up2 = Up(w3, w2)
        self.up3 = Up(w2, w1)
        self.outc = nn.Conv2d(w1, out_channels, kernel_size=1)

        self.lstm: nn.LSTM | None = None
        self.head: nn.Linear | None = None
        self._b_hw: tuple[int,int] | None = None
        self._b_ch: int = w4

    def _ensure(self, Hb: int, Wb: int, device, dtype):
        if self.lstm is not None and self._b_hw == (Hb, Wb):
            return
        F = self._b_ch * Hb * Wb
        self.lstm = nn.LSTM(F, self.hidden_size, self.num_layers, batch_first=True).to(device=device, dtype=dtype)
        self.head = nn.Linear(self.hidden_size, F).to(device=device, dtype=dtype)
        self._b_hw = (Hb, Wb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,C,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected x as 4D/5D, got {x.shape}")

        B, T, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {C}")

        bseq = []
        x1_last = x2_last = x3_last = None

        for t in range(T):
            xt = x[:, t]
            x1 = self.inc(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            b  = self.down3(x3)   # (B,w4,H/8,W/8)
            bseq.append(b)
            if t == T - 1:
                x1_last, x2_last, x3_last = x1, x2, x3

        bseq = torch.stack(bseq, dim=1)  # (B,T,w4,Hb,Wb)
        Hb, Wb = bseq.shape[-2], bseq.shape[-1]
        self._ensure(Hb, Wb, x.device, x.dtype)

        bflat = bseq.reshape(B, T, -1)
        out, _ = self.lstm(bflat)
        h_last = out[:, -1, :]
        b_last = self.head(h_last).reshape(B, self._b_ch, Hb, Wb)

        x = self.up1(b_last, x3_last)
        x = self.up2(x, x2_last)
        x = self.up3(x, x1_last)
        return self.outc(x)
