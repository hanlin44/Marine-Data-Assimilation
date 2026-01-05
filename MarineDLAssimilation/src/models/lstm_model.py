from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .base_model import BaseModel


class DepthLSTMModel(BaseModel):
    """
    LSTM 基线：把 depth 方向当作序列（Sequence length = D）。

    对每个网格点 (lat, lon)，取若干 3D 变量的垂向剖面：
      [var1(z), var2(z), ...]_{z=1..D}
    用 LSTM 做序列到序列重建。

    注意：
    - 它不显式建模水平空间相关性（H,W），因此通常不如 CNN/U-Net，但适合作为对照实验。
    - 本实现仍按工程统一输入：x = concat([xb, xobs, mask])。
    """

    def __init__(
        self,
        include_channels: int,
        out_channels: int,
        *,
        var_slices: Dict[str, Dict[str, int]],
        depth_max: int,
        vars_3d: Optional[List[str]] = None,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.C = include_channels
        self.out_channels = out_channels
        self.var_slices = var_slices
        self.depth_max = depth_max
        self.vars_3d = vars_3d or ["thetao", "so", "uo", "vo"]

        self.use_vars = [v for v in self.vars_3d if v in var_slices and var_slices[v]["is3d"] == 1]
        if not self.use_vars:
            raise ValueError(f"No 3D vars found for LSTM among {self.vars_3d}. Available: {list(var_slices.keys())}")

        # 假设这些3D变量深度层数一致（GLORYS常见）
        D0 = var_slices[self.use_vars[0]]["ndepth"]
        self.D0 = D0
        self.nv = len(self.use_vars)

        # 每个深度点的输入特征：对每个变量都输入 (xb, xobs)，再加mask
        feat_size = self.nv * 2 + 1
        self.lstm = nn.LSTM(input_size=feat_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.nv),
        )

        # 回退输出：若 target_vars 包含 LSTM 不覆盖的通道，用 1x1 卷积给一个基线
        self.fallback = nn.Conv2d(self.C * 2 + 1, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2C + 1, H, W)  (默认mask=1通道；若你启用per_channel_mask，可在train.py里禁用LSTM或自行扩展)
        """
        B, Cin, H, W = x.shape
        if Cin < self.C * 2 + 1:
            raise ValueError(f"Expected input channels >= {self.C*2+1}, got {Cin}")

        xb = x[:, :self.C, :, :]
        xobs = x[:, self.C:self.C * 2, :, :]
        mask = x[:, self.C * 2:self.C * 2 + 1, :, :]  # (B,1,H,W)

        # 组装序列特征：列表元素形状 (B,D,H,W)
        feats = []
        for v in self.use_vars:
            sl = self.var_slices[v]
            start, end = sl["start"], sl["end"]
            xb_v = xb[:, start:end, :, :]
            xobs_v = xobs[:, start:end, :, :]
            feats.append(xb_v)
            feats.append(xobs_v)

        mask_seq = mask.expand(B, self.D0, H, W)
        feats.append(mask_seq)

        # stack -> (B,D,F,H,W) -> reshape到 (B*H*W, D, F)
        feat = torch.stack(feats, dim=2)  # (B,D,F,H,W)
        feat = feat.permute(0, 3, 4, 1, 2).contiguous()  # (B,H,W,D,F)
        feat = feat.view(B * H * W, self.D0, feat.shape[-1])  # (B*H*W,D,F)

        out, _ = self.lstm(feat)       # (B*H*W,D,hidden)
        pred = self.head(out)          # (B*H*W,D,nv)

        # reshape 回 (B,nv,D,H,W) -> flatten: (B,nv*D,H,W)
        pred = pred.view(B, H, W, self.D0, self.nv).permute(0, 4, 3, 1, 2).contiguous()
        pred_flat = pred.view(B, self.nv * self.D0, H, W)

        # 基线输出
        y_hat = self.fallback(x)

        # 尝试把预测写回对应通道位置（要求 target_vars 的通道空间与 var_slices 对齐）
        offset = 0
        for v in self.use_vars:
            sl = self.var_slices[v]
            D = sl["ndepth"]
            src = pred_flat[:, offset:offset + D, :, :]
            offset += D
            if sl["end"] <= y_hat.shape[1]:
                y_hat[:, sl["start"]:sl["end"], :, :] = src

        return y_hat
