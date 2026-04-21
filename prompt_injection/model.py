from typing import Dict, Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier


def build_model() -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_binary": float(f1_score(y, y_pred, average="binary")),
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    return metrics


def save_model(model, path) -> None:
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)