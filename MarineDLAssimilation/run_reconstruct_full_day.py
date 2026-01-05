"""PyCharm-friendly full-grid inference entrypoint.

Example:
    python run_reconstruct_full_day.py --run_dir runs/20250101_120000 --nc_path data/06/xxx.nc
"""

from src.reconstruct_full_day import main


if __name__ == "__main__":
    main()
