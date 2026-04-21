from datetime import datetime
from pathlib import Path

from prompt_injection.config import ARTIFACTS_DIR


CACHE_DIR = ARTIFACTS_DIR / "cache"
RUNS_DIR = ARTIFACTS_DIR / "runs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def create_run_dir() -> Path:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir