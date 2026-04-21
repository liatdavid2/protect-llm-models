from typing import Tuple, Dict

import pandas as pd
from datasets import load_dataset

from src.config import DATASET_NAME, DATASET_CONFIG


REQUIRED_COLUMNS = ["text", "label"]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    train_df = ds["train"].to_pandas()
    valid_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()

    for df in (train_df, valid_df, test_df):
        _validate_columns(df)
        df["text"] = df["text"].astype(str).fillna("")
        df["label"] = df["label"].astype(int)

    return train_df, valid_df, test_df


def build_label_map(train_df: pd.DataFrame) -> Dict[int, str]:
    labels = sorted(train_df["label"].unique().tolist())
    return {int(label): f"class_{int(label)}" for label in labels}