from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from creditcard_fraud.config import MODELS_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _bundle()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Fraud classification API",
    lifespan=lifespan,
)


class PredictBody(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "Time": 0.0,
                        "V1": -1.359807,
                        "V2": -0.072781,
                        "V3": 2.536347,
                        "V4": 1.378155,
                        "V5": -0.338321,
                        "V6": 0.462388,
                        "V7": 0.239599,
                        "V8": 0.098698,
                        "V9": 0.363787,
                        "V10": 0.090794,
                        "V11": -0.551600,
                        "V12": -0.617801,
                        "V13": -0.991390,
                        "V14": -0.311169,
                        "V15": 1.468177,
                        "V16": -0.470401,
                        "V17": 0.207971,
                        "V18": 0.025791,
                        "V19": 0.403993,
                        "V20": 0.251412,
                        "V21": -0.018307,
                        "V22": 0.277838,
                        "V23": -0.110474,
                        "V24": 0.066928,
                        "V25": 0.128539,
                        "V26": -0.189115,
                        "V27": 0.133558,
                        "V28": -0.021053,
                        "Amount": 149.62,
                    }
                }
            ]
        }
    }

    features: dict[str, float] = Field(
        ..., description="Все признаки из model_meta.json (train без Class): Time, V1…V28, Amount."
    )


class PredictOut(BaseModel):
    prediction: int
    fraud_proba: float | None = None


@lru_cache(maxsize=1)
def _bundle():
    mp, jp = MODELS_DIR / "model.pkl", MODELS_DIR / "model_meta.json"
    if not mp.is_file() or not jp.is_file():
        raise FileNotFoundError(
            "Нет models/model.pkl или model_meta.json. Выполните: dvc repro или "
            "python -m creditcard_fraud.modeling.train"
        )
    return joblib.load(mp), json.loads(jp.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    try:
        _bundle()
        return {"status": "ok"}
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing"}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictBody):
    try:
        model, meta = _bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    cols = meta["feature_columns"]
    missing = [c for c in cols if c not in body.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_features",
                "message": "В features должны быть все ключи из model_meta.json (без Class).",
                "missing": missing[:15],
                "missing_count": len(missing),
                "expected_columns": cols,
            },
        )
    row = pd.DataFrame([[body.features[c] for c in cols]], columns=cols)
    pred = int(model.predict(row)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row)[0, 1])
    return PredictOut(prediction=pred, fraud_proba=proba)
