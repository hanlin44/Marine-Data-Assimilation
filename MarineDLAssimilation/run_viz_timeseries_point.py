# run_viz_timeseries_point.py
# python run_viz_timeseries_point.py --data_dir data/06 --var zos --lat 0 --lon 150 --depth 0 --out outputs/timeseries --show
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _guess_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="e.g. data/06")
    ap.add_argument("--var", required=True, help="e.g. zos/thetao/so/uo/vo")
    ap.add_argument("--lat", type=float, required=True, help="target latitude, nearest")
    ap.add_argument("--lon", type=float, required=True, help="target longitude, nearest (dataset uses -180~180 in your case)")
    ap.add_argument("--depth", type=int, default=0, help="depth idx for 3D vars")
    ap.add_argument("--out", default="outputs/timeseries")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files in {data_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = []
    values = []

    for fp in files:
        ds = xr.open_dataset(fp, engine="netcdf4")
        if args.var not in ds:
            continue

        tname = _guess_name(ds, ["time"])
        zname = _guess_name(ds, ["depth", "lev", "z"])
        latn = _guess_name(ds, ["latitude", "lat"])
        lonn = _guess_name(ds, ["longitude", "lon"])

        da = ds[args.var]
        if tname and tname in da.dims:
            da = da.isel({tname: 0})
            t = ds[tname].isel({tname: 0}).values
        else:
            t = fp.name  # fallback

        if zname and zname in da.dims:
            da = da.isel({zname: args.depth})

        # nearest point
        da_pt = da.sel({latn: args.lat, lonn: args.lon}, method="nearest").squeeze()
        v = float(da_pt.values)
        if np.isfinite(v):
            times.append(t)
            values.append(v)

    if not values:
        raise RuntimeError("No finite values extracted. Check lon/lat range or var name.")

    plt.figure(figsize=(9, 4))
    plt.plot(values, marker="o")
    plt.title(f"time series at (lat={args.lat}, lon={args.lon}) var={args.var}")
    plt.xlabel("time index (sorted by filename)")
    plt.ylabel(args.var)
    p = out_dir / f"ts_{args.var}_lat{args.lat}_lon{args.lon}_z{args.depth}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    if args.show:
        plt.show()
    plt.close()
    print("[OK] saved", p)


if __name__ == "__main__":
    main()

# eg: python run_viz_timeseries_point.py --data_dir data/06 --var thetao --lat 0 --lon 150 --depth 0
