import json
from pathlib import Path

import joblib
from fastapi.testclient import TestClient
import pandas as pd
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


def test_predict(tmp_path, monkeypatch):
    _dummy_model(tmp_path)
    monkeypatch.setattr(api_main, "MODELS_DIR", tmp_path)
    api_main._bundle.cache_clear()
    c = TestClient(api_main.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json={"features": {"a": 0.5, "b": 0.5}})
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)
