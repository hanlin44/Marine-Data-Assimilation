from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict
from pathlib import Path
import time


import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---- imports (support running as both module and script) ----
# If you run `python -m src.train`, __package__ == 'src' and relative imports work.
# If you click-run this file directly in PyCharm, __package__ is empty and relative imports break.
# This block makes both ways work.
if __package__ is None or __package__ == "":
    import sys as _sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))

    from src.data_loader import make_dataloaders, NormStats
    from src.models.base_model import make_input_tensor
    from src.models.cnn_model import CNNModel
    from src.models.unet_model import UNetModel
    from src.models.lstm_flat_model import LSTMFlatModel
    from src.models.pinn_model import PINNModel
    from src.models.cnn_lstm_model import CNNLSTMModel
    from src.models.cnn_lstm_pinn_model import CNNLSTMPINNModel
    from src.models.unet_lstm_pinn_model import UNetLSTMPINNModel
    from src.utils.metrics import rmse, rmse_masked
    from src.utils.physics import PhysWeights, physics_loss
    from src.utils.exp_logging import make_run_dir, setup_logger, log_environment, log_args, save_json, append_csv_row
else:
    from .data_loader import make_dataloaders, NormStats
    from .models.base_model import make_input_tensor
    from .models.cnn_model import CNNModel
    from .models.unet_model import UNetModel
    from .models.lstm_flat_model import LSTMFlatModel
    from .models.pinn_model import PINNModel
    from .models.cnn_lstm_model import CNNLSTMModel
    from .models.cnn_lstm_pinn_model import CNNLSTMPINNModel
    from .models.unet_lstm_pinn_model import UNetLSTMPINNModel
    from .utils.metrics import rmse, rmse_masked
    from .utils.physics import PhysWeights, physics_loss
    from .utils.exp_logging import make_run_dir, setup_logger, log_environment, log_args, save_json, append_csv_row


def build_pred_slices(meta) -> Dict[str, Dict[str, int]]:
    """
    预测张量(pred/y)的通道空间由 target_vars 按顺序拼接得到：
      pred = [target_var1(all depths), target_var2(...), ...]
    返回 var -> slice信息（相对于pred通道空间）。
    """
    pred_slices: Dict[str, Dict[str, int]] = {}
    c0 = 0
    for v in meta.target_vars:
        sl_inc = meta.var_slices[v]
        n = sl_inc["ndepth"]
        pred_slices[v] = {"start": c0, "end": c0 + n, "ndepth": n, "is3d": sl_inc["is3d"]}
        c0 += n
    return pred_slices


def parse_args():
    p = argparse.ArgumentParser("MarineDLAssimilation training")
    # 兼容工程内置数据目录：
    #   <project_root>\\data\\06\\*.nc
    # 让 PyCharm 直接右键运行时也能不传参数就找到数据。
    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "06"
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir),
        help="directory containing daily *.nc files (default: <project_root>/data/06)",
    )
    p.add_argument("--model", type=str, default="lstm", choices=["cnn", "unet", "lstm", "pinn", "cnn_lstm_pinn", "unet_lstm_pinn"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--depth_max", type=int, default=10, help="max depth levels used for 3D vars (<=50)")
    p.add_argument("--obs_ratio", type=float, default=0.02, help="sparse observation ratio in a patch")
    p.add_argument("--obs_mode", type=str, default="multisource", choices=["random", "multisource"], help="observation simulation mode")
    p.add_argument("--sst_ratio", type=float, default=None, help="(multisource) SST surface coverage ratio; default derives from obs_ratio")
    p.add_argument("--sla_ratio", type=float, default=None, help="(multisource) SLA coverage ratio; default derives from obs_ratio")
    p.add_argument("--argo_profiles", type=int, default=1, help="(multisource) number of Argo-like profiles per patch")
    p.add_argument("--argo_depth_stride", type=int, default=1, help="(multisource) observe every k depth levels for Argo profiles")
    p.add_argument("--use_obs_error_weight", type=int, default=1, help="add observation error weight channel (wobs) and use it in obs-only loss")
    p.add_argument("--sigma_sst", type=float, default=0.4)
    p.add_argument("--sigma_sla", type=float, default=0.03)
    p.add_argument("--sigma_argo_temp", type=float, default=0.1)
    p.add_argument("--sigma_argo_sal", type=float, default=0.02)
    p.add_argument("--include_vars", type=str, default="thetao,so,uo,vo,zos", help='comma list or "ALL"')
    p.add_argument("--target_vars", type=str, default="thetao,so,uo,vo,zos", help='comma list or "ALL"')
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0, help="Windows建议0")
    p.add_argument("--seed", type=int, default=0)

    # loss / ablation
    p.add_argument("--data_loss_on_obs_only", type=int, default=0)
    p.add_argument("--per_channel_mask", type=int, default=0)
    p.add_argument("--ignore_invalid", type=int, default=1, help="mask out invalid (land/缺测) points in data loss")

    # PINNs weights
    p.add_argument("--phys_geo", type=float, default=0.0, help="geostrophic loss weight")
    p.add_argument("--phys_advdiff", type=float, default=0.0, help="advection-diffusion loss weight")
    p.add_argument("--phys_kappa", type=float, default=10.0, help="effective diffusivity in advdiff")
    p.add_argument("--pinn_backbone", type=str, default="unet", choices=["unet", "cnn"])

    p.add_argument("--run_name", type=str, default=None, help="optional run folder name under <project_root>/runs")

    return p.parse_args()


def main():
    import torch
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- logging & run directory ----
    # Put runs/ under project root (works no matter where you click-run from in PyCharm)
    _proj_root = Path(__file__).resolve().parents[1]
    run_dir = make_run_dir(root=_proj_root / "runs", run_name=args.run_name)
    
    logger = setup_logger(run_dir)
    logger.info(f"run_dir = {run_dir}")
    env = log_environment(logger)
    hparams = log_args(logger, args)
    
    # Persist run metadata (nice for reproducibility)
    save_json(env, run_dir / "env.json")
    save_json(hparams, run_dir / "hparams.json")
    
    # CSV history (epoch-level)
    history_csv = run_dir / "history.csv"
    history_header = ["epoch", "train_loss", "val_mse", "val_rmse", "best_val_mse", "elapsed_sec"]
    t0 = time.time()
    eff_per_channel_mask = bool(args.per_channel_mask) or (args.obs_mode == "multisource") or bool(args.use_obs_error_weight)

    train_loader, val_loader, test_loader, meta, stats = make_dataloaders(
        args.data_dir,
        model=args.model,
        include_vars=args.include_vars,
        target_vars=args.target_vars,
        depth_max=args.depth_max,
        patch_size=args.patch_size,
        obs_ratio=args.obs_ratio,
        obs_mode=args.obs_mode,
        sst_ratio=args.sst_ratio,
        sla_ratio=args.sla_ratio,
        argo_profiles=args.argo_profiles,
        argo_depth_stride=args.argo_depth_stride,
        use_obs_error_weight=bool(args.use_obs_error_weight),
        sigma_sst=args.sigma_sst,
        sigma_sla=args.sigma_sla,
        sigma_argo_temp=args.sigma_argo_temp,
        sigma_argo_sal=args.sigma_argo_sal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        per_channel_mask=eff_per_channel_mask,
        compute_norm=True,
    )

    C = len(meta.channels)
    mask_ch = C if eff_per_channel_mask else 1
    wobs_ch = mask_ch if bool(args.use_obs_error_weight) else 0
    in_channels = 2 * C + mask_ch + wobs_ch
    out_channels = sum(meta.var_slices[v]["ndepth"] for v in meta.target_vars)

    pred_slices = build_pred_slices(meta)

    # save meta
    (run_dir / "meta.json").write_text(json.dumps({
        "args": vars(args),
        "channel_meta": meta.to_jsonable(),
        "pred_slices": pred_slices,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats is not None:
        torch.save({"mean": stats.mean, "std": stats.std}, run_dir / "norm_stats.pt")

    # build model
    m = args.model.lower()
    if m == "cnn":
        model = CNNModel(in_channels, out_channels, width=64, depth=6)
    elif m == "unet":
        model = UNetModel(in_channels, out_channels, base_width=32)
    elif m == "lstm":
        model = LSTMFlatModel(in_channels, out_channels, hidden_size=256, num_layers=2)
    elif m == "pinn":
        model = PINNModel(in_channels, out_channels, backbone=args.pinn_backbone, base_width=32)
    elif m == "cnn_lstm_pinn":
        model = CNNLSTMPINNModel(in_channels, out_channels, width=64, hidden_size=256, num_layers=2)
    elif m == "unet_lstm_pinn":
        model = UNetLSTMPINNModel(in_channels, out_channels, base_width=32, hidden_size=256, num_layers=2)
    else:
        raise ValueError(m)

    device = torch.device(args.device)
    model.to(device)

    # ---- auto init for models with lazy layers (e.g., LSTMFlatModel) ----
    # This avoids "optimizer got an empty parameter list" when the model
    # creates parameters on the first forward.
    try:
        with torch.no_grad():
            b0 = next(iter(train_loader))
            xb0 = b0["xb"].to(device)
            xobs0 = b0["xobs"].to(device)
            mask0 = b0["mask"].to(device)
            wobs0 = b0.get("wobs")
            wobs0 = wobs0.to(device) if (wobs0 is not None and bool(args.use_obs_error_weight)) else None
            x0 = make_input_tensor(xb0, xobs0, mask0, wobs0)
            _ = model(x0)
    except Exception:
        pass

    # if m == "lstm":
    #     opt = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
    # else:
    #     opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    phys_w = PhysWeights(geo=args.phys_geo, advdiff=args.phys_advdiff, kappa=args.phys_kappa)

    best_val = float("inf")
    best_path = run_dir / "checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            xb = batch["xb"].to(device)
            xobs = batch["xobs"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)

            wobs = batch.get("wobs")
            wobs = wobs.to(device) if (wobs is not None and bool(args.use_obs_error_weight)) else None
            x = make_input_tensor(xb, xobs, mask, wobs)

            y_hat = model(x)

            # ---- masked/weighted data loss (avoid NaNs from land/缺测) ----
            mask_y = batch.get("mask_y")
            valid_y = batch.get("valid_y")
            if valid_y is None:
                valid_y = torch.ones_like(y)
            else:
                valid_y = valid_y.to(device)

            if args.ignore_invalid:
                w = valid_y
            else:
                w = torch.ones_like(y)

            if args.data_loss_on_obs_only:
                if mask_y is not None:
                    w_obs = mask_y.to(device)
                else:
                    # fallback：用输入mask近似（当旧数据集不提供mask_y时）
                    if mask.shape[1] == 1:
                        w_obs = mask.expand_as(y)
                    else:
                        w_obs = mask[:, :y.shape[1], :, :]
                # 观测误差权重：wobs_y already includes mask_b * (std/sigma)^2
                if bool(args.use_obs_error_weight) and (batch.get("wobs_y") is not None):
                    w_obs = batch["wobs_y"].to(device)
                w = w * w_obs

            num = (w * (y_hat - y) ** 2).sum()
            den = w.sum().clamp(min=1.0)
            data_loss = num / den

            loss = data_loss

            if ("pinn" in m) and (phys_w.geo > 0 or phys_w.advdiff > 0):
                # 物理损失在pred空间按target_vars通道定义，因此需要 target_vars 至少包含 zos/uo/vo/thetao/so
                pl = 0.0
                # 逐样本计算物理损失（lat/lon随patch变化）
                for bi in range(y_hat.shape[0]):
                    pl = pl + physics_loss(y_hat[bi:bi+1], pred_slices, lat[bi], lon[bi], phys_w)
                pl = pl / y_hat.shape[0]
                loss = loss + pl

            opt.zero_grad()
            # opt.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{tr_losses[-1]:.4e}")

        # val
        model.eval()
        val_losses = []
        val_rmses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]"):
                xb = batch["xb"].to(device)
                xobs = batch["xobs"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)
                wobs = batch.get("wobs")
                wobs = wobs.to(device) if (wobs is not None and bool(args.use_obs_error_weight)) else None
                x = make_input_tensor(xb, xobs, mask, wobs)
                y_hat = model(x)
                valid_y = batch.get("valid_y")
                if valid_y is None:
                    w = torch.ones_like(y)
                else:
                    w = valid_y.to(device) if args.ignore_invalid else torch.ones_like(y)
                num = (w * (y_hat - y) ** 2).sum()
                den = w.sum().clamp(min=1.0)
                val_losses.append(float((num / den).item()))
                val_rmses.append(rmse_masked(y_hat, y, w).mean().item())
        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_rmse = sum(val_rmses) / max(1, len(val_rmses))
        # ---- epoch summary ----
        tr_loss = sum(tr_losses) / max(1, len(tr_losses))
        elapsed = time.time() - t0
        logger.info(f"[TRAIN] epoch={epoch} loss={tr_loss:.4e}")
        logger.info(f"[VAL]   epoch={epoch} mse={val_loss:.4e} rmse={val_rmse:.4e}")

        append_csv_row(history_csv, history_header, [epoch, tr_loss, val_loss, val_rmse, best_val, round(elapsed, 3)])
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "meta": meta.to_jsonable(),
                "pred_slices": pred_slices,
                "norm": stats.to_dict() if stats is not None else None,
            }, best_path)
            logger.info(f"[OK] saved best checkpoint to {best_path}")
    logger.info(f"[DONE] run_dir = {run_dir}")
    # close file handles
    try:
        train_loader.dataset.close()
        val_loader.dataset.close()
        test_loader.dataset.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
