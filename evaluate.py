import numpy as np

from prompt_injection.config import EMBEDDING_MODEL_NAME
from prompt_injection.data import load_splits
from prompt_injection.features import EmbeddingEncoder
from prompt_injection.model import evaluate_model, load_model
from prompt_injection.latest_run import get_latest_run_dir
from prompt_injection.embedding_cache import get_cache_path, load_embeddings_cache, save_embeddings_cache


def main():
    latest_run_dir = get_latest_run_dir()
    model_path = latest_run_dir / "xgb_prompt_injection.joblib"

    print(f"Loading model from latest run: {model_path}")

    cache_path = get_cache_path(EMBEDDING_MODEL_NAME)

    if cache_path.exists():
        print(f"Loading cached embeddings from: {cache_path}")
        _, _, X_valid, y_valid, X_test, y_test = load_embeddings_cache(cache_path)
    else:
        print("Loading dataset...")
        train_df, valid_df, test_df = load_splits()

        print("Encoding text with sentence embeddings...")
        encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)

        X_train = encoder.encode(train_df["text"].tolist())
        y_train = train_df["label"].to_numpy()

        X_valid = encoder.encode(valid_df["text"].tolist())
        y_valid = valid_df["label"].to_numpy()

        X_test = encoder.encode(test_df["text"].tolist())
        y_test = test_df["label"].to_numpy()

        print(f"Saving embeddings cache to: {cache_path}")
        save_embeddings_cache(
            cache_path,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
        )

    model = load_model(model_path)

    valid_metrics = evaluate_model(model, X_valid, y_valid)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("Validation metrics:")
    print(valid_metrics)

    print("\nTest metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()