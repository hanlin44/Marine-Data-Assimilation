# src/utils/plots.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metrics_figures(run_dir: str, out_dir: Optional[str] = None, show: bool = False) -> Dict[str, str]:
    """
    Read run_dir/meta.json + run_dir/metrics.json and save:
      - rmse_vs_depth.png
      - corr_vs_depth.png
      - rmse_2d_bar.png (if 2D vars exist)
      - corr_2d_bar.png (if 2D vars exist)
    """
    run_dir = str(run_dir)
    run_path = Path(run_dir)
    meta = _read_json(run_path / "meta.json")
    metrics = _read_json(run_path / "metrics.json")

    rmse = np.asarray(metrics["rmse"], dtype=float)
    corr = np.asarray(metrics["corr"], dtype=float)

    # v4+ meta has pred_slices like: var -> {start,end,is3d,ndepth}
    pred_slices = meta.get("pred_slices", None)
    if pred_slices is None:
        raise RuntimeError("meta.json missing 'pred_slices'. Please use the newer project version.")

    vars3d = [v for v, d in pred_slices.items() if int(d.get("is3d", 0)) == 1]
    vars2d = [v for v, d in pred_slices.items() if int(d.get("is3d", 0)) == 0]

    if out_dir is None:
        out_path = _ensure_dir(run_path / "figures")
    else:
        out_path = _ensure_dir(Path(out_dir))

    def _get(var: str, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        info = pred_slices[var]
        sl = slice(int(info["start"]), int(info["end"]))
        ndepth = int(info["ndepth"])
        return arr[sl], ndepth

    saved = {}

    # RMSE vs depth
    plt.figure(figsize=(9, 5))
    for v in vars3d:
        a, n = _get(v, rmse)
        plt.plot(np.arange(n), a, marker="o", label=v)
    plt.xlabel("depth index (0=surface ...)")
    plt.ylabel("RMSE (normalized)")
    plt.title("RMSE vs depth")
    plt.legend()
    p1 = out_path / "rmse_vs_depth.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    if show:
        plt.show()
    plt.close()
    saved["rmse_vs_depth"] = str(p1)

    # Corr vs depth
    plt.figure(figsize=(9, 5))
    for v in vars3d:
        a, n = _get(v, corr)
        plt.plot(np.arange(n), a, marker="o", label=v)
    plt.xlabel("depth index (0=surface ...)")
    plt.ylabel("Correlation (Pearson)")
    plt.title("Correlation vs depth")
    plt.legend()
    p2 = out_path / "corr_vs_depth.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    if show:
        plt.show()
    plt.close()
    saved["corr_vs_depth"] = str(p2)

    # 2D bars
    if len(vars2d) > 0:
        names, rm, co = [], [], []
        for v in vars2d:
            a, _ = _get(v, rmse)
            b, _ = _get(v, corr)
            names.append(v)
            rm.append(float(a[0]))
            co.append(float(b[0]))

        x = np.arange(len(names))

        plt.figure(figsize=(9, 4))
        plt.bar(x, rm)
        plt.xticks(x, names)
        plt.ylabel("RMSE (normalized)")
        plt.title("2D variables RMSE")
        plt.tight_layout()
        p3 = out_path / "rmse_2d_bar.png"
        plt.savefig(p3, dpi=200)
        if show:
            plt.show()
        plt.close()
        saved["rmse_2d_bar"] = str(p3)

        plt.figure(figsize=(9, 4))
        plt.bar(x, co)
        plt.xticks(x, names)
        plt.ylabel("Correlation")
        plt.title("2D variables Correlation")
        plt.tight_layout()
        p4 = out_path / "corr_2d_bar.png"
        plt.savefig(p4, dpi=200)
        if show:
            plt.show()
        plt.close()
        saved["corr_2d_bar"] = str(p4)

    return saved


def plot_patch_5panel(
    xb: np.ndarray,
    y_obs: np.ndarray,
    obs_mask: np.ndarray,
    x_pred: np.ndarray,
    x_true: np.ndarray,
    out_path: Optional[str] = None,
    title: str = "",
    show: bool = False,
):
    """
    5-panel: background | sparse obs | prediction | truth | error
    All inputs are 2D arrays (H,W). obs_mask is 0/1.
    """
    xb = np.asarray(xb)
    y_obs = np.asarray(y_obs)
    obs_mask = np.asarray(obs_mask)
    x_pred = np.asarray(x_pred)
    x_true = np.asarray(x_true)

    obs_show = np.where(obs_mask > 0.5, y_obs, np.nan)
    err = x_pred - x_true

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(xb, origin="lower")
    axes[0].set_title("background")

    axes[1].imshow(obs_show, origin="lower")
    axes[1].set_title("sparse obs")

    axes[2].imshow(x_pred, origin="lower")
    axes[2].set_title("pred (analysis)")

    axes[3].imshow(x_true, origin="lower")
    axes[3].set_title("truth")

    axes[4].imshow(err, origin="lower")
    axes[4].set_title("error")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200)

    if show:
        plt.show()
    plt.close(fig)
