from pathlib import Path

import pandas as pd

from creditcard_fraud.preprocess import run_preprocess


def test_preprocess_skip_ge(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    df = pd.DataFrame(
        {
            "Time": range(300),
            "Amount": [10.0] * 300,
            "V1": [0.1] * 300,
            "Class": [0] * 290 + [1] * 10,
        }
    )
    df.to_csv(p, index=False)
    tr = tmp_path / "train.csv"
    te = tmp_path / "test.csv"
    run_preprocess(input_path=p, train_path=tr, test_path=te, skip_ge=True)
    assert tr.is_file() and pd.read_csv(tr).shape[0] > 0
