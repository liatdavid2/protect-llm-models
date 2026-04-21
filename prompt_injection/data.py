from typing import Tuple, Dict
import pandas as pd
from datasets import load_dataset
from prompt_injection.config import DATASET_NAME, DATASET_CONFIG
import os
from dotenv import load_dotenv
from datasets import load_dataset
from prompt_injection.config import DATASET_NAME, DATASET_CONFIG

REQUIRED_COLUMNS = ["text", "label"]

load_dotenv()

def load_splits():
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        token=os.getenv("HF_TOKEN"),
    )

    train_df = ds["train"].to_pandas()
    valid_df = ds["validation"].to_pandas()
    test_df = ds["test"].to_pandas()

    return train_df, valid_df, test_df


def build_label_map(train_df: pd.DataFrame) -> Dict[int, str]:
    labels = sorted(train_df["label"].unique().tolist())
    return {int(label): f"class_{int(label)}" for label in labels}