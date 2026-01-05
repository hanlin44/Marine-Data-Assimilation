from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseModel


class ConvLSTMCell(nn.Module):
    def __init__(self, cin: int, ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(cin + ch, 4 * ch, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # x: (B,C,H,W), h/c: (B,ch,H,W)
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class CNNLSTMModel(BaseModel):
    """
    CNN-LSTM（ConvLSTM）：
    - 典型用法是输入多日序列 patch (T>1) 做时序融合
    - 本工程为了先跑通，也支持 T=1（退化为一个卷积门控块）
    """
    def __init__(self, in_channels: int, out_channels: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.in_proj = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.cells = nn.ModuleList([ConvLSTMCell(hidden, hidden) for _ in range(num_layers)])
        self.out = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)  -> treat as T=1
        B, _, H, W = x.shape
        x = self.in_proj(x)

        h = [torch.zeros((B, self.hidden, H, W), device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        c = [torch.zeros((B, self.hidden, H, W), device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]

        # single step
        inp = x
        for li, cell in enumerate(self.cells):
            h[li], c[li] = cell(inp, h[li], c[li])
            inp = h[li]

        return self.out(inp)
