from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# ---- imports (support running as both module and script) ----
if __package__ is None or __package__ == "":
    import sys as _sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))

    from src.data_loader import build_channel_meta, NormStats, OceanPatchDataset
    from src.models.base_model import make_input_tensor
    from src.models.cnn_model import CNNModel
    from src.models.unet_model import UNetModel
    from src.models.lstm_flat_model import LSTMFlatModel
    from src.models.pinn_model import PINNModel
    from src.models.cnn_lstm_pinn_model import CNNLSTMPINNModel
    from src.models.unet_lstm_pinn_model import UNetLSTMPINNModel
else:
    from .data_loader import build_channel_meta, NormStats, OceanPatchDataset
    from .models.base_model import make_input_tensor
    from .models.cnn_model import CNNModel
    from .models.unet_model import UNetModel
    from .models.lstm_flat_model import LSTMFlatModel
    from .models.pinn_model import PINNModel
    from .models.cnn_lstm_pinn_model import CNNLSTMPINNModel
    from .models.unet_lstm_pinn_model import UNetLSTMPINNModel


def parse_args():
    p = argparse.ArgumentParser("Reconstruct a full day (full grid inference) using a trained run")
    p.add_argument("--run_dir", type=str, required=True, help="runs/<timestamp> directory containing checkpoint.pt")
    p.add_argument("--nc_path", type=str, required=True, help="path to a single daily .nc file")
    p.add_argument("--out_dir", type=str, default=None, help="output folder (default: <run_dir>/reconstructions)")
    p.add_argument("--stride", type=int, default=None, help="sliding window stride (default: patch_size)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_model(model_name: str, in_channels: int, out_channels: int, pinn_backbone: str = "unet"):
    m = model_name.lower()
    if m == "cnn":
        return CNNModel(in_channels, out_channels, width=64, depth=6)
    if m == "unet":
        return UNetModel(in_channels, out_channels, base_width=32)
    if m == "lstm":
        return LSTMFlatModel(in_channels, out_channels, hidden_size=256, num_layers=2)
    if m == "pinn":
        return PINNModel(in_channels, out_channels, backbone=pinn_backbone, base_width=32)
    if m == "cnn_lstm_pinn":
        return CNNLSTMPINNModel(in_channels, out_channels, width=64, hidden_size=256, num_layers=2)
    if m == "unet_lstm_pinn":
        return UNetLSTMPINNModel(in_channels, out_channels, base_width=32, hidden_size=256, num_layers=2)
    raise ValueError(m)


def _ensure_covering_starts(L: int, patch: int, stride: int) -> List[int]:
    """Start indices that cover [0, L) with patches of size `patch` and step `stride`."""
    if L <= patch:
        return [0]
    starts = list(range(0, L - patch + 1, stride))
    last = L - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_args = ckpt["args"]
    pred_slices: Dict[str, Dict[str, int]] = ckpt["pred_slices"]

    nc_path = Path(args.nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    # meta + norm
    meta = build_channel_meta(nc_path, ckpt_args["include_vars"], ckpt_args["target_vars"], ckpt_args["depth_max"])
    if ckpt.get("norm") is not None:
        mean = torch.tensor(ckpt["norm"]["mean"], dtype=torch.float32)
        std = torch.tensor(ckpt["norm"]["std"], dtype=torch.float32)
        stats = NormStats(mean=mean, std=std)
    else:
        # fallback (should not happen)
        stats = None

    # dataset wrapper for reading patches + generating background/obs
    ds = OceanPatchDataset(
        files=[nc_path],
        meta=meta,
        patch_size=int(ckpt_args["patch_size"]),
        obs_ratio=float(ckpt_args.get("obs_ratio", 0.02)),
        obs_mode=str(ckpt_args.get("obs_mode", "random")),
        sst_ratio=ckpt_args.get("sst_ratio", None),
        sla_ratio=ckpt_args.get("sla_ratio", None),
        argo_profiles=int(ckpt_args.get("argo_profiles", 1)),
        argo_depth_stride=int(ckpt_args.get("argo_depth_stride", 1)),
        use_obs_error_weight=bool(ckpt_args.get("use_obs_error_weight", 0)),
        sigma_sst=float(ckpt_args.get("sigma_sst", 0.4)),
        sigma_sla=float(ckpt_args.get("sigma_sla", 0.03)),
        sigma_argo_temp=float(ckpt_args.get("sigma_argo_temp", 0.1)),
        sigma_argo_sal=float(ckpt_args.get("sigma_argo_sal", 0.02)),
        per_channel_mask=bool(ckpt_args.get("per_channel_mask", 0)) or bool(ckpt_args.get("use_obs_error_weight", 0)) or (ckpt_args.get("obs_mode", "random") == "multisource"),
        norm_stats=stats,
        seed=int(args.seed),
        cache_max_open=1,
    )

    C = len(meta.channels)
    eff_per_channel = ds.per_channel_mask
    mask_ch = C if eff_per_channel else 1
    wobs_ch = mask_ch if bool(ckpt_args.get("use_obs_error_weight", 0)) else 0
    in_channels = 2 * C + mask_ch + wobs_ch
    out_channels = sum(meta.var_slices[v]["ndepth"] for v in meta.target_vars)

    model = build_model(ckpt_args["model"], in_channels, out_channels, pinn_backbone=ckpt_args.get("pinn_backbone", "unet"))
    device = torch.device(args.device)
    model.to(device)
    # warmup for lazy models (e.g. LSTMFlatModel)
    dummy = torch.zeros((1, in_channels, ds.patch_size, ds.patch_size), device=device)
    _ = model(dummy)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    stride = int(args.stride) if args.stride is not None else int(ds.patch_size)
    ys = _ensure_covering_starts(ds.H, ds.patch_size, stride)
    xs = _ensure_covering_starts(ds.W, ds.patch_size, stride)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_dir / "reconstructions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # disk-backed accumulators (sum/count) to avoid huge RAM usage
    sum_path = out_dir / f"{nc_path.stem}.pred_sum.float32.dat"
    cnt_path = out_dir / f"{nc_path.stem}.pred_cnt.uint16.dat"
    pred_sum = np.memmap(sum_path, mode="w+", dtype="float32", shape=(out_channels, ds.H, ds.W))
    pred_cnt = np.memmap(cnt_path, mode="w+", dtype="uint16", shape=(ds.H, ds.W))
    pred_sum[:] = 0.0
    pred_cnt[:] = 0

    # metrics accumulators (physical units)
    se_sum = np.zeros((out_channels,), dtype=np.float64)
    se_cnt = np.zeros((out_channels,), dtype=np.float64)

    # map target channel indices to include channel indices for de-normalization
    if stats is not None:
        idxs: List[int] = []
        for v in meta.target_vars:
            sl = meta.var_slices[v]
            idxs.extend(list(range(sl["start"], sl["end"])))
        mean_y = stats.mean[idxs].numpy().astype(np.float32)
        std_y = stats.std[idxs].numpy().astype(np.float32)
    else:
        mean_y = np.zeros((out_channels,), dtype=np.float32)
        std_y = np.ones((out_channels,), dtype=np.float32)

    from src.patches import PatchSpec

    pbar = tqdm(total=len(ys) * len(xs), desc="reconstruct")
    try:
        for y0 in ys:
            for x0 in xs:
                spec = PatchSpec(y0=y0, x0=x0, h=ds.patch_size, w=ds.patch_size)
                xr_ds = ds._cache.get(nc_path)
                y_true, valid = ds._read_true_patch(xr_ds, spec)
                xb, xobs, mask, wobs = ds._make_background_and_obs(y_true, valid)

                # build y/valid_y (target space)
                target_slices = [slice(meta.var_slices[v]["start"], meta.var_slices[v]["end"]) for v in meta.target_vars]
                y_t = torch.cat([y_true[s] for s in target_slices], dim=0)
                valid_t = torch.cat([valid[s] for s in target_slices], dim=0)

                # normalize inputs like training
                if stats is not None:
                    xb_n = (xb - stats.mean[:, None, None]) / (stats.std[:, None, None] + 1e-6)
                    xobs_n = (xobs - stats.mean[:, None, None]) / (stats.std[:, None, None] + 1e-6)
                    xb_n = xb_n * valid
                    xobs_n = xobs_n * valid
                else:
                    xb_n, xobs_n = xb, xobs

                xb_b = xb_n.unsqueeze(0).to(device)
                xobs_b = xobs_n.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)
                wobs_b = wobs.unsqueeze(0).to(device) if (wobs is not None and bool(ckpt_args.get("use_obs_error_weight", 0))) else None

                x_in = make_input_tensor(xb_b, xobs_b, mask_b, wobs_b)
                y_hat_n = model(x_in)[0].detach().cpu().numpy().astype(np.float32)  # (Cout,h,w)

                # de-normalize prediction to physical units
                y_hat = y_hat_n * std_y[:, None, None] + mean_y[:, None, None]

                # write into accumulators (with overlap average)
                pred_sum[:, y0:y0 + ds.patch_size, x0:x0 + ds.patch_size] += y_hat
                pred_cnt[y0:y0 + ds.patch_size, x0:x0 + ds.patch_size] += 1

                # patch metrics (only valid points)
                y_true_t = y_t.numpy().astype(np.float32)
                valid_np = valid_t.numpy().astype(np.float32)
                err = (y_hat - y_true_t) * valid_np
                se_sum += (err * err).sum(axis=(1, 2)).astype(np.float64)
                se_cnt += valid_np.sum(axis=(1, 2)).astype(np.float64)

                pbar.update(1)
    finally:
        pbar.close()
        try:
            ds.close()
        except Exception:
            pass

    # finalize prediction (average overlaps)
    cnt = np.maximum(pred_cnt.astype(np.float32), 1.0)
    pred_mean_path = out_dir / f"{nc_path.stem}.pred_mean.float32.dat"
    pred = np.memmap(pred_mean_path, mode="w+", dtype="float32", shape=(out_channels, ds.H, ds.W))
    for ci in range(out_channels):
        pred[ci, :, :] = pred_sum[ci, :, :] / cnt
    pred.flush()
    pred_sum.flush()
    pred_cnt.flush()

    rmse = np.sqrt(se_sum / np.maximum(se_cnt, 1.0))
    metrics = {
        "rmse_per_channel": rmse.tolist(),
        "rmse_mean": float(np.mean(rmse)),
        "out_channels": int(out_channels),
        "H": int(ds.H),
        "W": int(ds.W),
        "stride": int(stride),
        "patch_size": int(ds.patch_size),
    }
    (out_dir / f"{nc_path.stem}.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[OK] reconstruction saved under", out_dir)


if __name__ == "__main__":
    main()
