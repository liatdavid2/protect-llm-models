from src.config import EMBEDDING_MODEL_NAME
from src.data import load_splits, build_label_map
from src.features import EmbeddingEncoder
from src.model import build_model, evaluate_model, save_model
from src.utils import save_json
from src.embedding_cache import get_cache_path, load_embeddings_cache, save_embeddings_cache
from src.run_paths import create_run_dir


def main():
    run_dir = create_run_dir()

    model_path = run_dir / "xgb_prompt_injection.joblib"
    metrics_path = run_dir / "metrics.json"
    label_map_path = run_dir / "label_map.json"

    print("Loading dataset...")
    train_df, valid_df, test_df = load_splits()

    print("Building label map...")
    label_map = build_label_map(train_df)
    save_json(label_map_path, label_map)

    cache_path = get_cache_path(EMBEDDING_MODEL_NAME)

    if cache_path.exists():
        print(f"Loading cached embeddings from: {cache_path}")
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_embeddings_cache(cache_path)
    else:
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

    print("Training XGBoost...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Evaluating on validation...")
    valid_metrics = evaluate_model(model, X_valid, y_valid)

    print("Evaluating on test...")
    test_metrics = evaluate_model(model, X_test, y_test)

    all_metrics = {
        "dataset_name": "neuralchemy/Prompt-injection-dataset",
        "dataset_config": "core",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "validation": valid_metrics,
        "test": test_metrics,
    }

    print("Saving model...")
    save_model(model, model_path)
    save_json(metrics_path, all_metrics)

    print("Done.")
    print(f"Run directory: {run_dir}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()