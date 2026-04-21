from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "core"  # "core" first, later you can try "full"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_PATH = ARTIFACTS_DIR / "xgb_prompt_injection.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"