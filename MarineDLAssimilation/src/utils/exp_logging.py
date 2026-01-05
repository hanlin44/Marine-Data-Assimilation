# src/utils/exp_logging.py
from __future__ import annotations
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

def make_run_dir(root: str = "runs", run_name: Optional[str] = None) -> Path:
    """
    Create runs/YYYYMMDD_HHMMSS (or runs/<run_name>) and return Path.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = run_name or ts
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def setup_logger(run_dir: Path, name: str = "MarineDL", level: int = logging.INFO) -> logging.Logger:
    """
    Logger -> both console and run_dir/train.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging

    # Clear existing handlers (important when re-running in PyCharm)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_csv_row(path: Path, header: list[str], row: list[Any]) -> None:
    import csv
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)

def log_environment(logger: logging.Logger) -> Dict[str, Any]:
    """Collect environment info and log it in a readable form."""
    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cwd": str(Path.cwd()),
    }
    logger.info("env:")
    for k in sorted(info.keys()):
        logger.info(f"  {k}: {info[k]}")
    return info

# def log_environment(logger: logging.Logger) -> Dict[str, Any]:
#     info = {
#         "platform": platform.platform(),
#         "python": sys.version.replace("\n", " "),
#         "torch": torch.__version__,
#         "cuda_available": torch.cuda.is_available(),
#         "cuda_version": torch.version.cuda,
#     }
#     if torch.cuda.is_available():
#         try:
#             info["gpu_name"] = torch.cuda.get_device_name(0)
#         except Exception:
#             info["gpu_name"] = "unknown"
#     logger.info(f"ENV: {info}")
#     return info

def log_args(logger: logging.Logger, args: Any) -> Dict[str, Any]:
    # args: argparse.Namespace 以前是这个
    """Log all CLI hyperparameters in a readable form and return as dict."""
    d: Dict[str, Any] = vars(args).copy()
    logger.info("hyperparams:")
    for k in sorted(d.keys()):
        logger.info(f"  {k}: {d[k]}")
    return d

