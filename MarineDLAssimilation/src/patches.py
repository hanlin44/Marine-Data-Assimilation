from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch


@dataclass(frozen=True)
class PatchSpec:
    """描述一个patch在全局网格上的位置与尺寸（按索引切片）"""
    y0: int
    x0: int
    h: int
    w: int


def random_patch_spec(
    H: int,
    W: int,
    patch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> PatchSpec:
    """在 [0,H)×[0,W) 上随机采样一个 patch_size×patch_size 的patch。"""
    if patch_size > H or patch_size > W:
        raise ValueError(f"patch_size={patch_size} exceeds H={H}, W={W}")
    if generator is None:
        y0 = int(torch.randint(0, H - patch_size + 1, (1,)).item())
        x0 = int(torch.randint(0, W - patch_size + 1, (1,)).item())
    else:
        y0 = int(torch.randint(0, H - patch_size + 1, (1,), generator=generator).item())
        x0 = int(torch.randint(0, W - patch_size + 1, (1,), generator=generator).item())
    return PatchSpec(y0=y0, x0=x0, h=patch_size, w=patch_size)


def extract_patch(x: torch.Tensor, spec: PatchSpec) -> torch.Tensor:
    """
    从张量中切 patch。

    允许输入：
    - (C,H,W) 或 (B,C,H,W)
    """
    if x.dim() == 3:
        return x[:, spec.y0:spec.y0 + spec.h, spec.x0:spec.x0 + spec.w]
    if x.dim() == 4:
        return x[:, :, spec.y0:spec.y0 + spec.h, spec.x0:spec.x0 + spec.w]
    raise ValueError(f"Unsupported tensor dim: {x.dim()}")


def sliding_window_specs(H: int, W: int, patch_size: int, stride: int) -> Iterator[PatchSpec]:
    """用于评估/推理时，将全局切成不重叠或重叠patch。"""
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            yield PatchSpec(y0=y0, x0=x0, h=patch_size, w=patch_size)


def assemble_patches_mean(
    patches: torch.Tensor,
    specs: list[PatchSpec],
    out_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    将预测patch拼回全局（重叠区域取平均）。

    patches: (N,C,h,w)
    out_shape: (C,H,W)
    """
    C, H, W = out_shape
    device = patches.device
    out = torch.zeros((C, H, W), device=device, dtype=patches.dtype)
    cnt = torch.zeros((1, H, W), device=device, dtype=patches.dtype)

    for i, spec in enumerate(specs):
        out[:, spec.y0:spec.y0 + spec.h, spec.x0:spec.x0 + spec.w] += patches[i]
        cnt[:, spec.y0:spec.y0 + spec.h, spec.x0:spec.x0 + spec.w] += 1.0

    out = out / torch.clamp(cnt, min=1.0)
    return out
