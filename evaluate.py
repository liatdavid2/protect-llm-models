from src.config import EMBEDDING_MODEL_NAME, MODEL_PATH
from src.data import load_splits
from src.features import EmbeddingEncoder
from src.model import evaluate_model, load_model


def main():
    _, valid_df, test_df = load_splits()

    encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)
    model = load_model(MODEL_PATH)

    X_valid = encoder.encode(valid_df["text"].tolist())
    y_valid = valid_df["label"].to_numpy()

    X_test = encoder.encode(test_df["text"].tolist())
    y_test = test_df["label"].to_numpy()

    valid_metrics = evaluate_model(model, X_valid, y_valid)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("Validation metrics:")
    print(valid_metrics)

    print("\nTest metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()