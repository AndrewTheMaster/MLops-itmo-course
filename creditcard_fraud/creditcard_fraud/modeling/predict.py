from __future__ import annotations

import json
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from creditcard_fraud.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    meta_path: Path = MODELS_DIR / "model_meta.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    limit: int | None = typer.Option(None, help="Макс. число строк."),
):
    """Пакетный инференс по CSV (колонка Class удаляется, если есть)."""
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    columns = meta["feature_columns"]
    df = pd.read_csv(features_path)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    df = df[columns]
    if limit is not None:
        df = df.head(limit)
    out = pd.DataFrame({"prediction": model.predict(df)})
    if hasattr(model, "predict_proba"):
        out["fraud_proba"] = model.predict_proba(df)[:, 1]
    out.to_csv(predictions_path, index=False)
    logger.success(f"→ {predictions_path} ({len(out)} строк)")


if __name__ == "__main__":
    app()
