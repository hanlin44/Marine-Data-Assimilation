# run_plot_history.py
# python run_plot_history.py --run_dir runs/yyyymmdd_hhmmss
"""
Plot training history curves from <run_dir>/history.csv.

PyCharm usage:
- Right-click this file -> Run (adjust RUN_DIR below if needed)

CLI usage:
    python run_plot_history.py --run_dir runs/20251227_103452
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/20251227_103452")
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/figures")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    hist_path = run_dir / "history.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"history.csv not found: {hist_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(hist_path)

    # 1) train loss + val mse
    plt.figure()
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_mse" in df.columns:
        plt.plot(df["epoch"], df["val_mse"], label="val_mse")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training Loss / Validation MSE")
    plt.tight_layout()
    plt.savefig(out_dir / "history_loss.png", dpi=150)
    plt.close()

    # 2) val rmse
    if "val_rmse" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["val_rmse"], label="val_rmse")
        plt.xlabel("epoch")
        plt.ylabel("rmse")
        plt.legend()
        plt.title("Validation RMSE")
        plt.tight_layout()
        plt.savefig(out_dir / "history_val_rmse.png", dpi=150)
        plt.close()

    print(f"[OK] Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
