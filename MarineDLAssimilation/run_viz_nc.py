# run_viz_nc.py
# python run_viz_nc.py --nc data/06/mercatorglorys12v1_gl12_mean_19930601_R19930602.nc --var thetao --depth 0 --stride 6 --out outputs/nc_viz --show
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


def _slice_for_coord(coord, vmin, vmax):
    # robust slice for ascending/descending coords
    a0 = float(coord.values[0])
    a1 = float(coord.values[-1])
    if a0 <= a1:
        return slice(vmin, vmax)
    else:
        return slice(vmax, vmin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="path to one nc file")
    ap.add_argument("--var", required=True, help="e.g. thetao/so/uo/vo/zos")
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--depth", type=int, default=0, help="depth index for 3D vars")
    ap.add_argument("--lat_min", type=float, default=None)
    ap.add_argument("--lat_max", type=float, default=None)
    ap.add_argument("--lon_min", type=float, default=None)
    ap.add_argument("--lon_max", type=float, default=None)
    ap.add_argument("--stride", type=int, default=4, help="downsample factor for fast plotting")
    ap.add_argument("--out", default="outputs/nc_viz")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.nc, engine="netcdf4")  # _FillValue -> NA handled by xarray

    tname = _guess_name(ds, ["time"])
    zname = _guess_name(ds, ["depth", "lev", "z"])
    lat = _guess_name(ds, ["latitude", "lat"])
    lon = _guess_name(ds, ["longitude", "lon"])

    if args.var not in ds:
        raise KeyError(f"var '{args.var}' not in dataset. Available: {list(ds.data_vars.keys())}")

    da = ds[args.var]

    # select time
    if tname is not None and tname in da.dims:
        da = da.isel({tname: args.time})

    # select depth for 3D
    depth_info = ""
    if zname is not None and zname in da.dims:
        da = da.isel({zname: args.depth})
        if zname in ds.coords:
            depth_val = float(ds[zname].isel({zname: args.depth}).values)
            depth_info = f"_z{args.depth}_({depth_val:.3f}m)"
        else:
            depth_info = f"_z{args.depth}"

    # spatial subset
    if (args.lat_min is not None) and (args.lat_max is not None) and (lat is not None) and (lat in da.dims):
        da = da.sel({lat: _slice_for_coord(ds[lat], args.lat_min, args.lat_max)})
    if (args.lon_min is not None) and (args.lon_max is not None) and (lon is not None) and (lon in da.dims):
        da = da.sel({lon: _slice_for_coord(ds[lon], args.lon_min, args.lon_max)})

    # downsample for speed
    if args.stride and args.stride > 1:
        if lat is not None and lat in da.dims:
            da = da.isel({lat: slice(None, None, args.stride)})
        if lon is not None and lon in da.dims:
            da = da.isel({lon: slice(None, None, args.stride)})

    da2 = da.squeeze()

    # 1) field map
    plt.figure(figsize=(10, 4))
    da2.plot()  # xarray quick plotting
    plt.title(f"{args.var}{depth_info}")
    p1 = out_dir / f"{args.var}{depth_info}_map.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # 2) valid mask (finite)
    valid = np.isfinite(da2.values)
    plt.figure(figsize=(10, 4))
    plt.imshow(valid.astype(float), origin="lower")
    plt.title(f"{args.var}{depth_info} valid_mask (1=ocean/valid, 0=invalid)")
    p2 = out_dir / f"{args.var}{depth_info}_mask.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # 3) histogram of finite values
    vals = da2.values
    vals = vals[np.isfinite(vals)]
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=80)
    plt.title(f"{args.var}{depth_info} histogram (finite only)")
    p3 = out_dir / f"{args.var}{depth_info}_hist.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    print("[OK] saved:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)


if __name__ == "__main__":
    main()

# eg: python run_viz_nc.py --nc "data/06/mercatorglorys12v1_gl12_mean_19930630_R19930707.nc" --var thetao --depth 0 --stride 6
