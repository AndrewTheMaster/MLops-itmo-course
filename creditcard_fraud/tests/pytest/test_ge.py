import pandas as pd
import pytest

from creditcard_fraud.ge_validation import validate_interim_fraud_dataset


def test_ge_ok():
    df = pd.DataFrame(
        {
            "Time": range(200),
            "Amount": [1.0] * 200,
            "V1": [0.0] * 200,
            "Class": [0] * 199 + [1],
        }
    )
    validate_interim_fraud_dataset(df)


def test_ge_bad_class():
    df = pd.DataFrame(
        {
            "Time": range(200),
            "Amount": [1.0] * 200,
            "V1": [0.0] * 200,
            "Class": [2] * 200,
        }
    )
    with pytest.raises(ValueError, match="качества данных"):
        validate_interim_fraud_dataset(df)
