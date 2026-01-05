"""PyCharm-friendly entrypoint for evaluation.

Example:
  python run_evaluate.py --run_dir runs/20251218_213943
"""
# python run_evaluate.py --run_dir runs/yyyymmdd_hhmmss
from src.evaluate import main


if __name__ == "__main__":
    main()
