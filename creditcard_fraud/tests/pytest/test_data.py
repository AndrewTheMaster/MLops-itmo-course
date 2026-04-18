import pandas as pd
import pytest

from creditcard_fraud import preprocess


def _base_df(rows: int = 300) -> pd.DataFrame:
    classes = ([0, 1] * (rows // 2 + 1))[:rows]
    return pd.DataFrame(
        {
            "Time": list(range(rows)),
            "Amount": [10.0 + i * 0.01 for i in range(rows)],
            "V1": [0.1] * rows,
            "Class": classes,
        }
    )


@pytest.mark.parametrize(
    ("name", "df", "expected_exception"),
    [
        ("empty_dataframe", pd.DataFrame(columns=["Time", "Amount", "V1", "Class"]), ValueError),
        (
            "single_class_only_fraud",
            pd.DataFrame(
                {"Time": range(100), "Amount": [10.0] * 100, "V1": [0.1] * 100, "Class": [1] * 100}
            ),
            ValueError,
        ),
        (
            "amount_all_nan",
            pd.DataFrame(
                {"Time": range(100), "Amount": [float("nan")] * 100, "V1": [0.1] * 100, "Class": [0, 1] * 50}
            ),
            ValueError,
        ),
        (
            "amount_wrong_type",
            pd.DataFrame(
                {"Time": range(100), "Amount": ["abc"] * 100, "V1": [0.1] * 100, "Class": [0, 1] * 50}
            ),
            ValueError,
        ),
    ],
)
def test_run_preprocess_edge_cases(tmp_path, name, df, expected_exception):
    input_path = tmp_path / f"{name}.csv"
    train_path = tmp_path / f"{name}_train.csv"
    test_path = tmp_path / f"{name}_test.csv"
    df.to_csv(input_path, index=False)

    with pytest.raises(expected_exception):
        preprocess.run_preprocess(
            input_path=input_path,
            train_path=train_path,
            test_path=test_path,
            skip_ge=True,
        )


def test_run_preprocess_success_skip_ge(tmp_path):
    input_path = tmp_path / "dataset.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _base_df().to_csv(input_path, index=False)

    preprocess.run_preprocess(
        input_path=input_path,
        train_path=train_path,
        test_path=test_path,
        skip_ge=True,
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    assert set(train.columns) == {"Time", "Amount", "V1", "Class"}
    assert set(test.columns) == {"Time", "Amount", "V1", "Class"}
    assert len(train) + len(test) > 0


def test_main_calls_run_preprocess(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def _fake_run_preprocess(input_path, train_path, test_path, test_size, random_state, skip_ge):
        calls["input_path"] = input_path
        calls["train_path"] = train_path
        calls["test_path"] = test_path
        calls["test_size"] = test_size
        calls["random_state"] = random_state
        calls["skip_ge"] = skip_ge

    monkeypatch.setattr(preprocess, "run_preprocess", _fake_run_preprocess)

    preprocess.main(
        input_path=tmp_path / "i.csv",
        train_path=tmp_path / "tr.csv",
        test_path=tmp_path / "te.csv",
        test_size=0.25,
        random_state=7,
        skip_ge=True,
    )

    assert calls["test_size"] == 0.25
    assert calls["random_state"] == 7
    assert calls["skip_ge"] is True
