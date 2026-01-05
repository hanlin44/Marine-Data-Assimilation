# run_plot_metrics.py
# python run_plot_metrics.py --run_dir runs/yyyymmdd_hhmmss
import argparse
from src.utils.plots import save_metrics_figures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/20251218_213943")
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/figures")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    saved = save_metrics_figures(args.run_dir, out_dir=args.out_dir, show=args.show)
    for k, v in saved.items():
        print(f"[OK] {k}: {v}")

if __name__ == "__main__":
    main()

# eg: python run_plot_metrics.py --run_dir runs/20251218_213943
