from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

# ---- imports (support running as both module and script) ----
# If you run `python -m src.evaluate`, __package__ == 'src' and relative imports work.
# If you click-run this file directly in PyCharm, __package__ is empty and relative imports break.
if __package__ is None or __package__ == "":
    import sys as _sys
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))
    from src.data_loader import make_dataloaders
    from src.models.base_model import make_input_tensor
    from src.models.cnn_model import CNNModel
    from src.models.unet_model import UNetModel
    from src.models.lstm_flat_model import LSTMFlatModel
    from src.models.pinn_model import PINNModel
    from src.models.cnn_lstm_model import CNNLSTMModel
    from src.models.cnn_lstm_pinn_model import CNNLSTMPINNModel
    from src.models.unet_lstm_pinn_model import UNetLSTMPINNModel
    from src.utils.metrics import metrics_dict
else:
    from .data_loader import make_dataloaders
    from .models.base_model import make_input_tensor
    from .models.cnn_model import CNNModel
    from .models.unet_model import UNetModel
    from .models.lstm_flat_model import LSTMFlatModel
    from .models.pinn_model import PINNModel
    from .models.cnn_lstm_model import CNNLSTMModel
    from .models.cnn_lstm_pinn_model import CNNLSTMPINNModel
    from .models.unet_lstm_pinn_model import UNetLSTMPINNModel
    from .utils.metrics import metrics_dict


def parse_args():
    p = argparse.ArgumentParser("MarineDLAssimilation evaluate")
    p.add_argument("--run_dir", type=str, required=True, help="runs/yyyymmdd_hhmmss")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(model_name: str, in_channels: int, out_channels: int, *, C: int, pred_slices: dict, depth_max: int, pinn_backbone: str):
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


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_args = ckpt["args"]
    pred_slices = ckpt["pred_slices"]

    # rebuild dataloaders with same args
    _, _, test_loader, meta, _ = make_dataloaders(
        ckpt_args["data_dir"],
        model=ckpt_args["model"],
        include_vars=ckpt_args["include_vars"],
        target_vars=ckpt_args["target_vars"],
        depth_max=ckpt_args["depth_max"],
        patch_size=ckpt_args["patch_size"],
        obs_ratio=ckpt_args["obs_ratio"],
        obs_mode=ckpt_args.get("obs_mode", "random"),
        sst_ratio=ckpt_args.get("sst_ratio", None),
        sla_ratio=ckpt_args.get("sla_ratio", None),
        argo_profiles=ckpt_args.get("argo_profiles", 1),
        argo_depth_stride=ckpt_args.get("argo_depth_stride", 1),
        use_obs_error_weight=bool(ckpt_args.get("use_obs_error_weight", 0)),
        sigma_sst=ckpt_args.get("sigma_sst", 0.4),
        sigma_sla=ckpt_args.get("sigma_sla", 0.03),
        sigma_argo_temp=ckpt_args.get("sigma_argo_temp", 0.1),
        sigma_argo_sal=ckpt_args.get("sigma_argo_sal", 0.02),
        batch_size=ckpt_args["batch_size"],
        num_workers=0,
        seed=ckpt_args["seed"],
        per_channel_mask=bool(ckpt_args.get("per_channel_mask", 0)) or bool(ckpt_args.get("use_obs_error_weight", 0)) or (ckpt_args.get("obs_mode", "random") == "multisource"),
        compute_norm=True,
    )

    C = len(meta.channels)
    eff_per_channel = bool(ckpt_args.get("per_channel_mask", 0)) or bool(ckpt_args.get("use_obs_error_weight", 0)) or (ckpt_args.get("obs_mode", "random") == "multisource")
    mask_ch = C if eff_per_channel else 1
    wobs_ch = mask_ch if bool(ckpt_args.get("use_obs_error_weight", 0)) else 0
    in_channels = 2 * C + mask_ch + wobs_ch
    out_channels = sum(meta.var_slices[v]["ndepth"] for v in meta.target_vars)

    model = build_model(
        ckpt_args["model"], in_channels, out_channels,
        C=C, pred_slices=pred_slices, depth_max=ckpt_args["depth_max"],
        pinn_backbone=ckpt_args.get("pinn_backbone", "unet"),
    )
    model.to(args.device)
    # warmup for lazy models (e.g., LSTMFlatModel)
    with torch.no_grad():
        dummy = torch.zeros((1, in_channels, ckpt_args["patch_size"], ckpt_args["patch_size"]), device=args.device)
        _ = model(dummy)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            xb = batch["xb"].to(args.device)
            xobs = batch["xobs"].to(args.device)
            mask = batch["mask"].to(args.device)
            y = batch["y"].to(args.device)
            wobs = batch.get("wobs")
            wobs = wobs.to(args.device) if (wobs is not None and bool(ckpt_args.get("use_obs_error_weight", 0))) else None
            x = make_input_tensor(xb, xobs, mask, wobs)
            y_hat = model(x)
            valid_y = batch.get("valid_y")
            w = valid_y.to(args.device) if valid_y is not None else None
            all_metrics.append(metrics_dict(y_hat, y, weight=w))

    # aggregate
    import numpy as np

    def _avg(key):
        vals = np.array([m[key] for m in all_metrics], dtype=np.float64)
        return vals.mean(axis=0).tolist()

    out = {"rmse": _avg("rmse"), "mae": _avg("mae"), "corr": _avg("corr")}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[OK] saved", run_dir / "metrics.json")

    try:
        test_loader.dataset.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
