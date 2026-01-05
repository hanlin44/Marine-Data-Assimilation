from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


OMEGA = 7.2921159e-5  # rad/s
G = 9.81              # m/s^2
R_EARTH = 6371000.0   # m


def coriolis_f(lat_deg: torch.Tensor) -> torch.Tensor:
    """f = 2Ω sin(lat)"""
    lat_rad = lat_deg * torch.pi / 180.0
    return 2.0 * OMEGA * torch.sin(lat_rad)


def meters_per_degree_lat() -> float:
    return 2.0 * torch.pi * R_EARTH / 360.0


def meters_per_degree_lon(lat_deg: torch.Tensor) -> torch.Tensor:
    return meters_per_degree_lat() * torch.cos(lat_deg * torch.pi / 180.0)


def grid_spacing_m(lat_deg_1d: torch.Tensor, lon_deg_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    估计 dx, dy（每个格点附近的水平网格间距，单位m）。
    简化：dy常数（按纬度间隔），dx随纬度变化。
    返回：
      dx: (H,1) 每一行一个dx
      dy: (1,1) 常数dy
    """
    # dy: 用lat相邻差的均值
    if lat_deg_1d.numel() < 2 or lon_deg_1d.numel() < 2:
        raise ValueError("lat/lon length too small for spacing")
    dlat = torch.mean(torch.abs(lat_deg_1d[1:] - lat_deg_1d[:-1]))
    dlon = torch.mean(torch.abs(lon_deg_1d[1:] - lon_deg_1d[:-1]))

    dy = dlat * meters_per_degree_lat()
    # dx: 随纬度 cos(lat)
    mpdl = meters_per_degree_lon(lat_deg_1d)  # (H,)
    dx = dlon * mpdl  # (H,)
    return dx[:, None], torch.tensor([[dy]], device=lat_deg_1d.device, dtype=lat_deg_1d.dtype)


def ddx(field: torch.Tensor, dx_row: torch.Tensor) -> torch.Tensor:
    """
    field: (B,1,H,W)
    dx_row: (H,1)  每行一个dx
    """
    # central diff in x
    B, _, H, W = field.shape
    f = field
    # pad replicate
    fp = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")
    d = (fp[..., 2:] - fp[..., :-2]) / 2.0
    # divide by dx (broadcast to B,1,H,W)
    return d / (dx_row[None, None, :, :].clamp(min=1.0))


def ddy(field: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """
    dy: scalar tensor (1,1)
    """
    fp = torch.nn.functional.pad(field, (0, 0, 1, 1), mode="replicate")
    d = (fp[..., 2:, :] - fp[..., :-2, :]) / 2.0
    return d / dy.clamp(min=1.0)


def laplacian(field: torch.Tensor, dx_row: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """
    简单五点差分拉普拉斯：
    ∇²f ≈ (f_{i+1}-2f_i+f_{i-1})/dx^2 + (f_{j+1}-2f_j+f_{j-1})/dy^2
    """
    f = field
    fxp = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")
    fyp = torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")

    fxx = (fxp[..., 2:] - 2.0 * fxp[..., 1:-1] + fxp[..., :-2]) / (dx_row[None, None, :, :].clamp(min=1.0) ** 2)
    fyy = (fyp[..., 2:, :] - 2.0 * fyp[..., 1:-1, :] + fyp[..., :-2, :]) / (dy.clamp(min=1.0) ** 2)
    return fxx + fyy


def _surface(field_flat: torch.Tensor, sl: Dict[str, int]) -> torch.Tensor:
    """
    从扁平通道中取表层（depth=0）或2D变量（唯一通道）
    field_flat: (B,C,H,W)
    sl: {"start":..,"end":..,"ndepth":..,"is3d":..} within that tensor's channel space
    """
    start = sl["start"]
    if sl["ndepth"] >= 1:
        return field_flat[:, start:start + 1, :, :]
    raise ValueError("slice ndepth < 1")


@dataclass
class PhysWeights:
    geo: float = 1.0
    advdiff: float = 0.1
    kappa: float = 10.0  # m^2/s, 有效扩散系数（仅用于量纲配平）


def geostrophic_loss(
    pred: torch.Tensor,
    pred_slices: Dict[str, Dict[str, int]],
    lat_patch: torch.Tensor,
    lon_patch: torch.Tensor,
    *,
    eta_name: str = "zos",
    u_name: str = "uo",
    v_name: str = "vo",
) -> torch.Tensor:
    """
    地转平衡损失（表层近似）：
      u_g = -(g/f) dη/dy,  v_g = (g/f) dη/dx
    pred: (B,C,H,W) 其中应包含 eta/u/v 的通道
    """
    device = pred.device
    lat_patch = lat_patch.to(device)
    lon_patch = lon_patch.to(device)

    if eta_name not in pred_slices or u_name not in pred_slices or v_name not in pred_slices:
        return torch.tensor(0.0, device=device)

    eta = _surface(pred, pred_slices[eta_name])  # (B,1,H,W)
    u = _surface(pred, pred_slices[u_name])
    v = _surface(pred, pred_slices[v_name])

    dx_row, dy = grid_spacing_m(lat_patch, lon_patch)
    dx_row = dx_row.to(device)
    dy = dy.to(device)

    deta_dx = ddx(eta, dx_row)
    deta_dy = ddy(eta, dy)

    f = coriolis_f(lat_patch).to(device)  # (H,)
    # 避免低纬f≈0导致爆炸：对|f|小的地方降权
    f_abs = torch.abs(f)
    w = torch.clamp(f_abs / (f_abs.max() + 1e-12), min=0.0, max=1.0)  # (H,)
    # 低于阈值的行几乎不算
    w = torch.where(f_abs < 1e-5, torch.zeros_like(w), w)
    w2d = w[None, None, :, None]  # (1,1,H,1)

    f_safe = torch.where(f_abs < 1e-5, torch.full_like(f, 1e-5), f)
    f_safe = f_safe[None, None, :, None]  # broadcast

    ug = -G / f_safe * deta_dy
    vg =  G / f_safe * deta_dx

    loss = ((u - ug) ** 2 + (v - vg) ** 2) * w2d
    return loss.mean()


def advdiff_loss(
    pred: torch.Tensor,
    pred_slices: Dict[str, Dict[str, int]],
    lat_patch: torch.Tensor,
    lon_patch: torch.Tensor,
    *,
    tracer_names: Tuple[str, ...] = ("thetao", "so"),
    u_name: str = "uo",
    v_name: str = "vo",
    kappa: float = 10.0,
) -> torch.Tensor:
    """
    表层平流-扩散残差：
      r = u dT/dx + v dT/dy - kappa ∇²T
    对 thetao 与 so 分别计算并求和。
    """
    device = pred.device
    lat_patch = lat_patch.to(device)
    lon_patch = lon_patch.to(device)

    if u_name not in pred_slices or v_name not in pred_slices:
        return torch.tensor(0.0, device=device)

    u = _surface(pred, pred_slices[u_name])
    v = _surface(pred, pred_slices[v_name])

    dx_row, dy = grid_spacing_m(lat_patch, lon_patch)
    dx_row = dx_row.to(device)
    dy = dy.to(device)

    total = torch.tensor(0.0, device=device)
    for tr in tracer_names:
        if tr not in pred_slices:
            continue
        T = _surface(pred, pred_slices[tr])

        dTdx = ddx(T, dx_row)
        dTdy = ddy(T, dy)
        lapT = laplacian(T, dx_row, dy)

        r = u * dTdx + v * dTdy - kappa * lapT
        total = total + (r * r).mean()
    return total


def physics_loss(
    pred: torch.Tensor,
    pred_slices: Dict[str, Dict[str, int]],
    lat_patch: torch.Tensor,
    lon_patch: torch.Tensor,
    weights: PhysWeights,
) -> torch.Tensor:
    return (
        weights.geo * geostrophic_loss(pred, pred_slices, lat_patch, lon_patch)
        + weights.advdiff * advdiff_loss(pred, pred_slices, lat_patch, lon_patch, kappa=weights.kappa)
    )
