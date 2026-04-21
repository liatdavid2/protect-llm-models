import sys

from src.config import EMBEDDING_MODEL_NAME, MODEL_PATH
from src.features import EmbeddingEncoder
from src.model import load_model


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python infer.py "your prompt text here"')

    text = sys.argv[1]

    encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)
    model = load_model(MODEL_PATH)

    X = encoder.encode([text])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    result = {
        "text": text,
        "predicted_label": pred,
        "malicious_probability": proba,
    }

    print(result)


if __name__ == "__main__":
    main()