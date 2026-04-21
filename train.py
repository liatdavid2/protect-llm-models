from src.config import EMBEDDING_MODEL_NAME, MODEL_PATH, METRICS_PATH, LABEL_MAP_PATH
from src.data import load_splits, build_label_map
from src.features import EmbeddingEncoder
from src.model import build_model, evaluate_model, save_model
from src.utils import save_json


def main():
    print("Loading dataset...")
    train_df, valid_df, test_df = load_splits()

    print("Building label map...")
    label_map = build_label_map(train_df)
    save_json(LABEL_MAP_PATH, label_map)

    print("Encoding text with sentence embeddings...")
    encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)

    X_train = encoder.encode(train_df["text"].tolist())
    y_train = train_df["label"].to_numpy()

    X_valid = encoder.encode(valid_df["text"].tolist())
    y_valid = valid_df["label"].to_numpy()

    X_test = encoder.encode(test_df["text"].tolist())
    y_test = test_df["label"].to_numpy()

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
    save_model(model, MODEL_PATH)
    save_json(METRICS_PATH, all_metrics)

    print("Done.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()