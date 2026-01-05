from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from .patches import PatchSpec, random_patch_spec


_DATE_RE = re.compile(r"_mean_(\d{8})_R(\d{8})\.nc$", re.IGNORECASE)


def _sort_key_from_filename(p: Path) -> Tuple[int, str]:
    """按文件名中的日期排序；无法解析则按文件名。"""
    m = _DATE_RE.search(p.name)
    if not m:
        return (10**12, p.name)
    return (int(m.group(1)), p.name)


def list_nc_files(data_dir: str | Path) -> List[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.nc"), key=_sort_key_from_filename)
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {data_dir}")
    return files


def split_files(files: List[Path], train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(files)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = min(n_train, n - 2) if n >= 3 else n_train
    n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else n_val
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


@dataclass
class NormStats:
    mean: torch.Tensor  # (C,)
    std: torch.Tensor   # (C,)

    def to_dict(self):
        return {"mean": self.mean.cpu().numpy().tolist(), "std": self.std.cpu().numpy().tolist()}

    @staticmethod
    def from_path(path: str | Path) -> "NormStats":
        obj = torch.load(path, map_location="cpu")
        return NormStats(mean=obj["mean"], std=obj["std"])


class DatasetCache:
    """简单LRU：避免每次getitem都重新open大文件。"""

    def __init__(self, max_open: int = 2):
        self.max_open = max_open
        self._cache: Dict[str, xr.Dataset] = {}
        self._order: List[str] = []

    def get(self, path: Path) -> xr.Dataset:
        key = str(path)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]

        ds = xr.open_dataset(key, engine="netcdf4")  # 依赖 netCDF4
        self._cache[key] = ds
        self._order.append(key)
        while len(self._order) > self.max_open:
            old = self._order.pop(0)
            try:
                self._cache[old].close()
            except Exception:
                pass
            self._cache.pop(old, None)
        return ds

    def close(self):
        for ds in self._cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self._cache.clear()
        self._order.clear()


@dataclass
class ChannelMeta:
    include_vars: List[str]
    target_vars: List[str]
    depth_dim: str
    lat_dim: str
    lon_dim: str
    time_dim: str
    depth_max: int
    channels: List[Tuple[str, Optional[int]]]     # [(var, depth_index or None), ...]
    var_slices: Dict[str, Dict[str, int]]         # var -> {"start":.., "end":.., "ndepth":.., "is3d":0/1}

    def to_jsonable(self):
        return {
            "include_vars": self.include_vars,
            "target_vars": self.target_vars,
            "depth_dim": self.depth_dim,
            "lat_dim": self.lat_dim,
            "lon_dim": self.lon_dim,
            "time_dim": self.time_dim,
            "depth_max": self.depth_max,
            "channels": self.channels,
            "var_slices": self.var_slices,
        }


def _parse_vars_arg(arg: str) -> List[str]:
    arg = arg.strip()
    if arg.upper() == "ALL":
        return ["ALL"]
    return [v.strip() for v in arg.split(",") if v.strip()]


def build_channel_meta(
    sample_file: Path,
    include_vars_arg: str,
    target_vars_arg: str,
    depth_max: int,
) -> ChannelMeta:
    ds = xr.open_dataset(str(sample_file), engine="netcdf4")
    time_dim = "time" if "time" in ds.dims else list(ds.dims.keys())[0]
    depth_dim = "depth" if "depth" in ds.dims else "depth"
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    all_vars = list(ds.data_vars.keys())

    include_vars = _parse_vars_arg(include_vars_arg)
    if include_vars == ["ALL"]:
        include_vars = all_vars

    target_vars = _parse_vars_arg(target_vars_arg)
    if target_vars == ["ALL"]:
        target_vars = include_vars

    # 仅保留存在于文件中的变量
    include_vars = [v for v in include_vars if v in all_vars]
    target_vars = [v for v in target_vars if v in all_vars]
    if not include_vars:
        raise ValueError(f"No include_vars found in file. Available: {all_vars}")

    channels: List[Tuple[str, Optional[int]]] = []
    var_slices: Dict[str, Dict[str, int]] = {}
    c0 = 0

    for v in include_vars:
        da = ds[v]
        is3d = int(depth_dim in da.dims)
        if is3d:
            ndepth = min(int(ds.dims[depth_dim]), depth_max)
            start, end = c0, c0 + ndepth
            for k in range(ndepth):
                channels.append((v, k))
            c0 = end
        else:
            start, end = c0, c0 + 1
            channels.append((v, None))
            c0 = end

        var_slices[v] = {"start": start, "end": end, "ndepth": (end - start), "is3d": is3d}

    ds.close()
    return ChannelMeta(
        include_vars=include_vars,
        target_vars=target_vars,
        depth_dim=depth_dim,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        time_dim=time_dim,
        depth_max=depth_max,
        channels=channels,
        var_slices=var_slices,
    )


def _torch_norm(x: torch.Tensor, stats: NormStats) -> torch.Tensor:
    # x: (C,H,W)
    return (x - stats.mean[:, None, None]) / (stats.std[:, None, None] + 1e-6)


class OceanPatchDataset(Dataset):
    """从GLORYS日均NetCDF中按patch采样样本。

    注意：全球格点数据在陆地/缺测区域通常包含 NaN 或 _FillValue。
    如果不处理，归一化统计会直接变成 NaN，训练 loss/metric 也会变成 NaN。

    输出：
      xb:   (C,H,W) 背景
      xobs: (C,H,W) 稀疏观测填充
      mask: (1,H,W) 或 (C,H,W) 观测位置
      y:    (Cout,H,W) 目标真值
      valid:   (C,H,W) include_vars 的有效掩膜（1=有效，0=无效/缺测）
      valid_y: (Cout,H,W) y 对应的有效掩膜
    """

    def __init__(
        self,
        files: List[Path],
        meta: ChannelMeta,
        patch_size: int = 64,
        obs_ratio: float = 0.02,
        obs_mode: str = "random",
        sst_ratio: float | None = None,
        sla_ratio: float | None = None,
        argo_profiles: int = 1,
        argo_depth_stride: int = 1,
        use_obs_error_weight: bool = False,
        sigma_sst: float = 0.4,
        sigma_sla: float = 0.03,
        sigma_argo_temp: float = 0.1,
        sigma_argo_sal: float = 0.02,
        smooth_kernel: int = 7,
        noise_std: float = 0.05,
        per_channel_mask: bool = False,
        norm_stats: NormStats | None = None,
        seed: int = 0,
        cache_max_open: int = 2,
    ):
        self.files = files
        self.meta = meta
        self.patch_size = patch_size
        self.obs_ratio = obs_ratio
        self.obs_mode = obs_mode.lower()
        self.sst_ratio = float(sst_ratio) if sst_ratio is not None else None
        self.sla_ratio = float(sla_ratio) if sla_ratio is not None else None
        self.argo_profiles = int(argo_profiles)
        self.argo_depth_stride = max(1, int(argo_depth_stride))
        self.use_obs_error_weight = bool(use_obs_error_weight)
        self.sigma_sst = float(sigma_sst)
        self.sigma_sla = float(sigma_sla)
        self.sigma_argo_temp = float(sigma_argo_temp)
        self.sigma_argo_sal = float(sigma_argo_sal)
        self.smooth_kernel = smooth_kernel
        self.noise_std = noise_std
        # multisource / error-weighted obs 必须使用 per-channel mask
        self.per_channel_mask = bool(per_channel_mask) or (self.obs_mode == "multisource") or self.use_obs_error_weight
        self.norm_stats = norm_stats

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self._cache = DatasetCache(max_open=cache_max_open)

        ds0 = self._cache.get(self.files[0])
        self.H = int(ds0.dims[self.meta.lat_dim])
        self.W = int(ds0.dims[self.meta.lon_dim])
        self.lat = torch.tensor(ds0[self.meta.lat_dim].values.astype(np.float32))
        self.lon = torch.tensor(ds0[self.meta.lon_dim].values.astype(np.float32))

        # Pre-compute channel indices for multisource observations
        self._idx_thetao0: Optional[int] = None
        self._idx_zos: Optional[int] = None
        self._idx_thetao_all: List[int] = []
        self._idx_so_all: List[int] = []
        for i, (v, k) in enumerate(self.meta.channels):
            if v == "thetao" and k == 0:
                self._idx_thetao0 = i
            if v == "zos" and k is None:
                self._idx_zos = i
            if v == "thetao" and (k is not None):
                self._idx_thetao_all.append(i)
            if v == "so" and (k is not None):
                self._idx_so_all.append(i)

    def __len__(self):
        # 每个文件采样多个patch（够跑通&对比实验）
        return len(self.files) * 50

    def close(self):
        self._cache.close()

    @staticmethod
    def _clean_and_mask(arr: np.ndarray, *, extreme_abs: float = 1e10) -> Tuple[torch.Tensor, torch.Tensor]:
        """把缺测/无穷/异常极值转成0，并返回有效掩膜。"""
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        invalid = (~torch.isfinite(t)) | (torch.abs(t) > extreme_abs)
        valid = (~invalid).float()
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.where(invalid, torch.zeros_like(t), t)
        return t, valid

    def _read_true_patch(self, ds: xr.Dataset, spec: PatchSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取真实场 patch，输出 (y_true, valid) 形状均为 (C,h,w)。"""
        yslice = slice(spec.y0, spec.y0 + spec.h)
        xslice = slice(spec.x0, spec.x0 + spec.w)

        arrays: List[torch.Tensor] = []
        valids: List[torch.Tensor] = []
        depth_dim = self.meta.depth_dim

        for v in self.meta.include_vars:
            da = ds[v]
            if depth_dim in da.dims:  # 3D: (time,depth,lat,lon)
                ndepth = min(int(ds.dims[depth_dim]), self.meta.depth_max)
                arr = da.isel({
                    self.meta.time_dim: 0,
                    depth_dim: slice(0, ndepth),
                    self.meta.lat_dim: yslice,
                    self.meta.lon_dim: xslice,
                }).values
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim != 3:
                    raise ValueError(f"Unexpected ndim for {v}: {arr.shape}")
                t, m = self._clean_and_mask(arr)
                arrays.append(t)  # (D,h,w)
                valids.append(m)
            else:  # 2D: (time,lat,lon)
                arr = da.isel({
                    self.meta.time_dim: 0,
                    self.meta.lat_dim: yslice,
                    self.meta.lon_dim: xslice,
                }).values
                arr = np.asarray(arr, dtype=np.float32)
                t, m = self._clean_and_mask(arr)
                arrays.append(t[None, ...])
                valids.append(m[None, ...])

        y_true = torch.cat(arrays, dim=0)   # (C,h,w)
        valid = torch.cat(valids, dim=0)    # (C,h,w)
        return y_true, valid

    def _masked_smooth(self, x: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
        """对每个通道做 masked average pooling 平滑（避免NaN/陆地0污染海洋）。"""
        if k % 2 == 0:
            k += 1
        num = F.avg_pool2d((x * valid)[None, ...], kernel_size=k, stride=1, padding=k // 2)[0]
        den = F.avg_pool2d(valid[None, ...], kernel_size=k, stride=1, padding=k // 2)[0]
        return num / (den + 1e-6)

    def _make_background_and_obs(
        self,
        y_true: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """用真值生成背景 xb 与稀疏观测 xobs，并输出观测 mask。

        返回:
            xb:   (C,H,W)
            xobs: (C,H,W)
            mask: (1,H,W) or (C,H,W)
            wobs: (1,H,W) or (C,H,W) or None  # 观测误差权重通道(例如 1/sigma^2)
        """
        C, H, W = y_true.shape
        k = int(self.smooth_kernel)
        xb = self._masked_smooth(y_true, valid, k)
        xb = xb + self.noise_std * torch.randn(xb.shape, generator=self._rng, device=xb.device, dtype=xb.dtype)

        # 把无效点归零
        xb = xb * valid

        # ---- observation mask (random vs. multisource) ----
        wobs: torch.Tensor | None = None
        if self.obs_mode == "multisource":
            # force per-channel mask
            mask = torch.zeros((C, H, W), dtype=y_true.dtype)

            # ratios: if not set, derive from obs_ratio (roughly: SST denser than SLA denser than Argo)
            sst_ratio = self.sst_ratio if self.sst_ratio is not None else min(1.0, max(0.02, self.obs_ratio * 20.0))
            sla_ratio = self.sla_ratio if self.sla_ratio is not None else min(1.0, max(0.01, self.obs_ratio * 5.0))

            # 1) SST-like: surface thetao (depth0) 2D obs (with gaps)
            if self._idx_thetao0 is not None:
                raw = (torch.rand((1, H, W), generator=self._rng) < sst_ratio).float()
                mask[self._idx_thetao0:self._idx_thetao0 + 1] = raw

            # 2) SLA-like: zos 2D obs
            if self._idx_zos is not None:
                raw = (torch.rand((1, H, W), generator=self._rng) < sla_ratio).float()
                mask[self._idx_zos:self._idx_zos + 1] = torch.maximum(mask[self._idx_zos:self._idx_zos + 1], raw)

            # 3) Argo-like: sparse profiles of thetao/so (all depths at a few (y,x) points)
            nprof = max(0, int(self.argo_profiles))
            if nprof > 0 and (self._idx_thetao_all or self._idx_so_all):
                ys = torch.randint(low=0, high=H, size=(nprof,), generator=self._rng)
                xs = torch.randint(low=0, high=W, size=(nprof,), generator=self._rng)
                for yy, xx in zip(ys.tolist(), xs.tolist()):
                    if self._idx_thetao_all:
                        for ci, ch in enumerate(self._idx_thetao_all[:: self.argo_depth_stride]):
                            mask[ch, yy, xx] = 1.0
                    if self._idx_so_all:
                        for ch in self._idx_so_all[:: self.argo_depth_stride]:
                            mask[ch, yy, xx] = 1.0

            # Do not place obs on invalid points
            mask = mask * valid
        else:
            # random sparse obs (baseline)
            if self.per_channel_mask:
                raw = (torch.rand((C, H, W), generator=self._rng) < self.obs_ratio).float()
                mask = raw * valid
            else:
                raw = (torch.rand((1, H, W), generator=self._rng) < self.obs_ratio).float()
                valid_any = (valid.sum(dim=0, keepdim=True) > 0).float()
                mask = raw * valid_any

        mask_b = mask if mask.shape[0] == C else mask.expand(C, -1, -1)
        xobs = mask_b * y_true + (1.0 - mask_b) * xb
        xobs = xobs * valid

        # ---- obs error weight channel (optional) ----
        if self.use_obs_error_weight:
            # weights are per-channel; if mask is 1-channel, still provide 1-channel weight
            if self.norm_stats is not None:
                std = self.norm_stats.std.to(y_true.device, dtype=y_true.dtype)
            else:
                std = torch.ones((C,), device=y_true.device, dtype=y_true.dtype)

            sigma = torch.full((C,), float("inf"), device=y_true.device, dtype=y_true.dtype)
            # SST obs -> thetao depth0
            if self._idx_thetao0 is not None:
                sigma[self._idx_thetao0] = self.sigma_sst
            # SLA obs -> zos
            if self._idx_zos is not None:
                sigma[self._idx_zos] = self.sigma_sla
            # Argo profiles -> thetao/so (all depths)
            for ch in self._idx_thetao_all:
                sigma[ch] = self.sigma_argo_temp
            for ch in self._idx_so_all:
                sigma[ch] = self.sigma_argo_sal

            # normalized-space weight: (std/sigma)^2 ; if sigma=inf -> 0
            w_ch = torch.zeros((C,), device=y_true.device, dtype=y_true.dtype)
            finite = torch.isfinite(sigma) & (sigma > 0)
            w_ch[finite] = (std[finite] / sigma[finite]) ** 2

            w_full = w_ch[:, None, None] * mask_b
            wobs = w_full if mask.shape[0] == C else w_full.mean(dim=0, keepdim=True)
        return xb, xobs, mask, wobs

    def __getitem__(self, idx: int):
        file_idx = idx % len(self.files)
        f = self.files[file_idx]
        ds = self._cache.get(f)

        spec = random_patch_spec(self.H, self.W, self.patch_size, generator=self._rng)
        y_true, valid = self._read_true_patch(ds, spec)  # (C,h,w)
        xb, xobs, mask, wobs = self._make_background_and_obs(y_true, valid)

        # 只输出目标变量对应通道作为 y
        target_slices = []
        for v in self.meta.target_vars:
            sl = self.meta.var_slices[v]
            target_slices.append(slice(sl["start"], sl["end"]))
        if target_slices:
            y = torch.cat([y_true[s] for s in target_slices], dim=0)
            valid_y = torch.cat([valid[s] for s in target_slices], dim=0)
        else:
            y = y_true
            valid_y = valid


        # mask_y：把观测mask映射到 y 的通道空间（用于只在观测点计算数据项）
        C = y_true.shape[0]
        mask_b = mask if mask.shape[0] == C else mask.expand(C, -1, -1)
        if target_slices:
            mask_y = torch.cat([mask_b[s] for s in target_slices], dim=0)
        else:
            mask_y = mask_b

        # wobs_y：把观测误差权重映射到 y 的通道空间（用于加权观测项）
        wobs_y = None
        if wobs is not None:
            wobs_b = wobs if wobs.shape[0] == C else wobs.expand(C, -1, -1)
            if target_slices:
                wobs_y = torch.cat([wobs_b[s] for s in target_slices], dim=0)
            else:
                wobs_y = wobs_b

        # 归一化（并在无效点重新置0，避免把0变成(-mean/std)）
        if self.norm_stats is not None:
            xb = _torch_norm(xb, self.norm_stats) * valid
            xobs = _torch_norm(xobs, self.norm_stats) * valid

            if y.shape[0] == self.norm_stats.mean.shape[0]:
                y = _torch_norm(y, self.norm_stats) * valid_y
            else:
                idxs = []
                for s in target_slices:
                    idxs.extend(list(range(s.start, s.stop)))
                mean = self.norm_stats.mean[idxs]
                std = self.norm_stats.std[idxs]
                y = (y - mean[:, None, None]) / (std[:, None, None] + 1e-6)
                y = y * valid_y

        lat_patch = self.lat[spec.y0:spec.y0 + spec.h]
        lon_patch = self.lon[spec.x0:spec.x0 + spec.w]

        return {
            "xb": xb,
            "xobs": xobs,
            "mask": mask,
            "mask_y": mask_y,
            "wobs": wobs,
            "wobs_y": wobs_y,
            "y": y,
            "valid": valid,
            "valid_y": valid_y,
            "spec": (spec.y0, spec.x0, spec.h, spec.w),
            "lat": lat_patch,
            "lon": lon_patch,
        }


@torch.no_grad()
def estimate_norm_stats(
    dataset: OceanPatchDataset,
    num_batches: int = 10,
    batch_size: int = 4,
) -> NormStats:
    """随机抽若干 batch 估计每个通道的 mean/std（仅统计有效点）。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    total_sum = None
    total_sumsq = None
    total_cnt = None

    n = 0
    for batch in loader:
        xb = batch["xb"]       # (B,C,H,W)
        valid = batch["valid"]  # (B,C,H,W)

        s = (xb * valid).sum(dim=(0, 2, 3))
        ss = ((xb * xb) * valid).sum(dim=(0, 2, 3))
        c = valid.sum(dim=(0, 2, 3))

        if total_sum is None:
            total_sum = s
            total_sumsq = ss
            total_cnt = c
        else:
            total_sum = total_sum + s
            total_sumsq = total_sumsq + ss
            total_cnt = total_cnt + c

        n += 1
        if n >= num_batches:
            break

    cnt = total_cnt.clamp(min=1.0)
    mean = total_sum / cnt
    var = total_sumsq / cnt - mean * mean
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)
    return NormStats(mean=mean, std=std)


def make_dataloaders(
    data_dir: str | Path,
    *,
    model: str,
    include_vars: str,
    target_vars: str,
    depth_max: int,
    patch_size: int,
    obs_ratio: float,
    obs_mode: str = "random",
    sst_ratio: float | None = None,
    sla_ratio: float | None = None,
    argo_profiles: int = 1,
    argo_depth_stride: int = 1,
    use_obs_error_weight: bool = False,
    sigma_sst: float = 0.4,
    sigma_sla: float = 0.03,
    sigma_argo_temp: float = 0.1,
    sigma_argo_sal: float = 0.02,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    per_channel_mask: bool = False,
    cache_max_open: int = 2,
    compute_norm: bool = True,
):
    files = list_nc_files(data_dir)
    train_files, val_files, test_files = split_files(files)

    meta = build_channel_meta(train_files[0], include_vars, target_vars, depth_max)

    train_ds_tmp = OceanPatchDataset(
        train_files,
        meta,
        patch_size=patch_size,
        obs_ratio=obs_ratio,
        obs_mode=obs_mode,
        sst_ratio=sst_ratio,
        sla_ratio=sla_ratio,
        argo_profiles=argo_profiles,
        argo_depth_stride=argo_depth_stride,
        use_obs_error_weight=use_obs_error_weight,
        sigma_sst=sigma_sst,
        sigma_sla=sigma_sla,
        sigma_argo_temp=sigma_argo_temp,
        sigma_argo_sal=sigma_argo_sal,
        per_channel_mask=per_channel_mask,
        norm_stats=None,
        seed=seed,
        cache_max_open=cache_max_open,
    )
    stats = estimate_norm_stats(train_ds_tmp) if compute_norm else None
    train_ds_tmp.close()

    train_ds = OceanPatchDataset(
        train_files,
        meta,
        patch_size=patch_size,
        obs_ratio=obs_ratio,
        obs_mode=obs_mode,
        sst_ratio=sst_ratio,
        sla_ratio=sla_ratio,
        argo_profiles=argo_profiles,
        argo_depth_stride=argo_depth_stride,
        use_obs_error_weight=use_obs_error_weight,
        sigma_sst=sigma_sst,
        sigma_sla=sigma_sla,
        sigma_argo_temp=sigma_argo_temp,
        sigma_argo_sal=sigma_argo_sal,
        per_channel_mask=per_channel_mask,
        norm_stats=stats,
        seed=seed,
        cache_max_open=cache_max_open,
    )
    val_ds = OceanPatchDataset(
        val_files,
        meta,
        patch_size=patch_size,
        obs_ratio=obs_ratio,
        obs_mode=obs_mode,
        sst_ratio=sst_ratio,
        sla_ratio=sla_ratio,
        argo_profiles=argo_profiles,
        argo_depth_stride=argo_depth_stride,
        use_obs_error_weight=use_obs_error_weight,
        sigma_sst=sigma_sst,
        sigma_sla=sigma_sla,
        sigma_argo_temp=sigma_argo_temp,
        sigma_argo_sal=sigma_argo_sal,
        per_channel_mask=per_channel_mask,
        norm_stats=stats,
        seed=seed + 1,
        cache_max_open=cache_max_open,
    )
    test_ds = OceanPatchDataset(
        test_files,
        meta,
        patch_size=patch_size,
        obs_ratio=obs_ratio,
        obs_mode=obs_mode,
        sst_ratio=sst_ratio,
        sla_ratio=sla_ratio,
        argo_profiles=argo_profiles,
        argo_depth_stride=argo_depth_stride,
        use_obs_error_weight=use_obs_error_weight,
        sigma_sst=sigma_sst,
        sigma_sla=sigma_sla,
        sigma_argo_temp=sigma_argo_temp,
        sigma_argo_sal=sigma_argo_sal,
        per_channel_mask=per_channel_mask,
        norm_stats=stats,
        seed=seed + 2,
        cache_max_open=cache_max_open,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, meta, stats
