from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import typer

from creditcard_fraud.config import MLFLOW_FILE_STORE, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _load_train_test(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feature_columns = [c for c in train.columns if c != "Class"]
    return (
        train[feature_columns],
        train["Class"],
        test[feature_columns],
        test["Class"],
        feature_columns,
    )


def _log_and_train_model(
    name: str,
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, object]:
    with mlflow.start_run(run_name=name):
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        mlflow.log_metrics(
            {
                "recall_fraud": recall,
                "precision_fraud": precision,
                "f1_fraud": f1,
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info(
            f"{name}: recall={recall:.3f}, precision={precision:.3f}, f1={f1:.3f}"
        )
    return recall, model


def _resolve_tracking_uri(explicit: Optional[str]) -> str:
    """Переменная MLFLOW_TRACKING_URI или каталог .mlruns в PROJ_ROOT (writable)."""
    if explicit:
        return explicit
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    MLFLOW_FILE_STORE.mkdir(parents=True, exist_ok=True)
    return f"file:{MLFLOW_FILE_STORE.resolve().as_posix()}"


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    tracking_uri: Optional[str] = typer.Option(
        None,
        help="MLflow URI. По умолчанию: $MLFLOW_TRACKING_URI или <проект>/.mlruns",
    ),
    experiment_name: str = "creditcard_fraud_baseline",
    model_path: Path = MODELS_DIR / "model.pkl",
    meta_path: Path = MODELS_DIR / "model_meta.json",
):
    """
    LogisticRegression vs RandomForest в MLflow; на диск — модель с max recall_fraud (приоритет fraud).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test, feature_columns = _load_train_test(
        train_path, test_path
    )
    uri = _resolve_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {uri}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    candidates: list[tuple[str, float, object]] = []

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", n_jobs=-1
    )
    r, m = _log_and_train_model(
        "LogisticRegression", lr, x_train, y_train, x_test, y_test
    )
    candidates.append(("LogisticRegression", r, m))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    r, m = _log_and_train_model(
        "RandomForest", rf, x_train, y_train, x_test, y_test
    )
    candidates.append(("RandomForest", r, m))

    best_name, best_recall, best_model = max(candidates, key=lambda t: t[1])
    joblib.dump(best_model, model_path)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": best_name,
                "recall_fraud": best_recall,
                "feature_columns": feature_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.success(f"Лучшая по recall_fraud: {best_name} → {model_path}")


if __name__ == "__main__":
    app()
