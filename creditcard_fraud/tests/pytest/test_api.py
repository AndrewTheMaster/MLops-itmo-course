import json
from pathlib import Path
import time

import joblib
from fastapi.testclient import TestClient
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from creditcard_fraud.api import main as api_main


def _dummy_model(dirpath: Path) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    m = LogisticRegression().fit(X, [0, 1])
    joblib.dump(m, dirpath / "model.pkl")
    (dirpath / "model_meta.json").write_text(
        json.dumps({"feature_columns": ["a", "b"], "model_name": "t", "recall_fraud": 1.0}),
        encoding="utf-8",
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    _dummy_model(tmp_path)
    monkeypatch.setattr(api_main, "MODELS_DIR", tmp_path)
    api_main._bundle.cache_clear()
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_predict_happy_path(client):
    assert client.get("/health").status_code == 200
    r = client.post("/predict", json={"features": {"a": 0.5, "b": 0.5}})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert isinstance(body["fraud_probability"], float)
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_missing_required_field(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_invalid_amount_type_returns_422(client):
    r = client.post("/predict", json={"features": {"a": "abc", "b": 0.5}})
    assert r.status_code == 422


@pytest.mark.parametrize(
    "features",
    [
        {"a": 0.5},  # missing b
        {"a": 0.5, "b": 0.5, "c": 1.0},  # extra feature
    ],
)
def test_predict_wrong_feature_dimensionality(client, features):
    r = client.post("/predict", json={"features": features})
    assert r.status_code in (400, 422)
    body = r.json()
    assert "detail" in body


def test_predict_median_latency_under_200ms(client):
    elapsed = []
    payload = {"features": {"a": 0.5, "b": 0.5}}
    for _ in range(100):
        start = time.perf_counter()
        r = client.post("/predict", json=payload)
        elapsed.append((time.perf_counter() - start) * 1000)
        assert r.status_code == 200

    elapsed.sort()
    median_ms = elapsed[len(elapsed) // 2]
    assert median_ms <= 200.0
