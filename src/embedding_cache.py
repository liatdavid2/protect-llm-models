from pathlib import Path
from typing import Tuple

import numpy as np

from src.config import DATASET_CONFIG
from src.run_paths import CACHE_DIR


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace(":", "_")


def get_cache_path(model_name: str) -> Path:
    safe_name = _safe_model_name(model_name)
    return CACHE_DIR / f"embeddings_{DATASET_CONFIG}_{safe_name}.npz"


def save_embeddings_cache(
    cache_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )


def load_embeddings_cache(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_valid"],
        data["y_valid"],
        data["X_test"],
        data["y_test"],
    )