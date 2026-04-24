import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import boto3


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

GUARD_TRAINING_MODULES = [
    "prompt_injection_input_guard.train",
    "harmful_content_input_guard.train",
    "pii_output_guard.train",
    "system_prompt_leakage_output_guard.train",
]


def get_guard_artifact_dir(module_name: str) -> Path:
    guard_name = module_name.replace(".train", "")
    return ARTIFACTS_DIR / guard_name


def find_latest_metrics_file_for_guard(module_name: str) -> Optional[Path]:
    guard_dir = get_guard_artifact_dir(module_name)
    runs_dir = guard_dir / "runs"

    if not runs_dir.exists():
        print(f"No runs directory found for {module_name}: {runs_dir}", flush=True)
        return None

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]

    if not run_dirs:
        print(f"No run folders found for {module_name}: {runs_dir}", flush=True)
        return None

    latest_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    metrics_file = latest_run_dir / "metrics.json"

    if not metrics_file.exists():
        print(f"No metrics.json found in latest run folder: {latest_run_dir}", flush=True)
        return None

    return metrics_file

def flatten_metrics(data: dict, prefix: str = "") -> dict:
    flat = {}

    for key, value in data.items():
        clean_key = str(key).replace(" ", "_").replace("-", "_")

        full_key = f"{prefix}_{clean_key}" if prefix else clean_key

        if isinstance(value, dict):
            flat.update(flatten_metrics(value, full_key))
        elif isinstance(value, (int, float)):
            flat[full_key] = float(value)

    return flat


def log_guard_metrics_to_mlflow(module_name: str) -> dict:
    metrics_file = find_latest_metrics_file_for_guard(module_name)

    if metrics_file is None:
        print(f"No metrics file found for {module_name}", flush=True)
        return {}

    print(f"Found metrics file for {module_name}: {metrics_file}", flush=True)

    with metrics_file.open("r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    module_prefix = module_name.replace(".", "_")
    flat_metrics = flatten_metrics(metrics_data)

    logged_metrics = {}

    wanted_names = {
        "accuracy",
        "f1",
        "f1_score",
        "macro_f1",
        "weighted_f1",
        "precision",
        "recall",
        "macro_precision",
        "macro_recall",
        "weighted_precision",
        "weighted_recall",
    }

    for key, value in flat_metrics.items():
        normalized_key = key.lower()

        should_log = any(name in normalized_key for name in wanted_names)

        if should_log:
            metric_name = f"{module_prefix}_{key}"
            mlflow.log_metric(metric_name, value)
            logged_metrics[metric_name] = value

    mlflow.log_artifact(str(metrics_file), artifact_path=f"metrics/{module_prefix}")

    return logged_metrics

def get_run_name() -> str:
    return datetime.utcnow().strftime("guard_training_%Y-%m-%d_%H-%M-%S")


def upload_directory_to_s3(local_dir: Path, bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")

    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir).as_posix()
            s3_key = f"{prefix.rstrip('/')}/{relative_path}"
            print(f"Uploading {file_path} to s3://{bucket}/{s3_key}", flush=True)
            s3.upload_file(str(file_path), bucket, s3_key)


def run_training_module(module_name: str) -> dict:
    start_time = time.time()

    print("=" * 80, flush=True)
    print(f"Starting training module: {module_name}", flush=True)
    print("=" * 80, flush=True)

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=False,
    )

    duration_seconds = round(time.time() - start_time, 2)

    status = "success" if result.returncode == 0 else "failed"

    module_prefix = module_name.replace(".", "_")

    mlflow.log_metric(f"{module_prefix}_duration_seconds", duration_seconds)
    mlflow.log_param(f"{module_prefix}_status", status)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed for module: {module_name}")

    logged_metrics = log_guard_metrics_to_mlflow(module_name)

    print(f"Finished training module: {module_name}", flush=True)

    return {
        "module": module_name,
        "status": status,
        "duration_seconds": duration_seconds,
        "mlflow_metrics": logged_metrics,
    }

def write_manifest(run_name: str, results: list[dict]) -> Path:
    manifest = {
        "project": "secure-llm-gateway",
        "run_name": run_name,
        "training_mode": "offline_training_worker",
        "created_at_utc": datetime.utcnow().isoformat(),
        "models": GUARD_TRAINING_MODULES,
        "results": results,
        "artifacts_dir": str(ARTIFACTS_DIR),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ARTIFACTS_DIR / "training_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def main() -> None:
    run_name = get_run_name()

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "secure-llm-guard-training")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    print(f"MLflow tracking URI: {mlflow_tracking_uri}", flush=True)
    print(f"MLflow experiment: {mlflow_experiment_name}", flush=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("project", "secure-llm-gateway")
        mlflow.log_param("training_entrypoint", "train_all_guards.py")
        mlflow.log_param("number_of_guard_models", len(GUARD_TRAINING_MODULES))

        results = []

        for module_name in GUARD_TRAINING_MODULES:
            result = run_training_module(module_name)
            results.append(result)

        manifest_path = write_manifest(run_name, results)

        if ARTIFACTS_DIR.exists():
            mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")

        mlflow.log_artifact(str(manifest_path), artifact_path="manifest")

        s3_bucket = os.getenv("S3_BUCKET")
        s3_prefix = os.getenv(
            "S3_PREFIX",
            f"secure-llm-gateway/guard-training/{run_name}",
        )

        if s3_bucket:
            print(f"Uploading artifacts to s3://{s3_bucket}/{s3_prefix}", flush=True)
            upload_directory_to_s3(ARTIFACTS_DIR, s3_bucket, s3_prefix)
            mlflow.log_param("s3_bucket", s3_bucket)
            mlflow.log_param("s3_prefix", s3_prefix)
        else:
            print("S3_BUCKET is not set. Skipping S3 upload.", flush=True)

        print("=" * 80, flush=True)
        print("All guard models were trained successfully.", flush=True)
        print("=" * 80, flush=True)


if __name__ == "__main__":
    main()