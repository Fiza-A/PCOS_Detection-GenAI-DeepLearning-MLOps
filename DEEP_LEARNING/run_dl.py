"""Run the one-time Phase 3.3 REAL_ONLY_BASELINE training experiment."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from train import run_training  # noqa: E402


if __name__ == "__main__":
    run_training()
