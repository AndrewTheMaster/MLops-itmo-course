"""GE + очистка + масштабирование + train/test (как в EDA / dataset)."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import typer

from creditcard_fraud.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from creditcard_fraud.ge_validation import validate_interim_fraud_dataset

app = typer.Typer()


def run_preprocess(
    input_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    test_size: float = 0.3,
    random_state: int = 42,
    skip_ge: bool = False,
) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input dataset is empty.")
    if "Amount" not in df.columns or "Class" not in df.columns:
        raise ValueError("Input dataset must contain 'Amount' and 'Class' columns.")

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if df["Amount"].isna().all():
        raise ValueError("Column 'Amount' is invalid: all values are NaN after conversion.")

    if df["Class"].nunique(dropna=False) < 2:
        raise ValueError("Target column 'Class' must contain at least two classes.")

    df["Class"] = df["Class"].astype(int)
    if not skip_ge:
        validate_interim_fraud_dataset(df)

    q1 = df["Amount"].quantile(0.25)
    q3 = df["Amount"].quantile(0.75)
    iqr = q3 - q1
    df = df.loc[
        ~((df["Amount"] < (q1 - 1.5 * iqr)) | (df["Amount"] > (q3 + 1.5 * iqr)))
    ].copy()

    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    x = df.drop("Class", axis=1)
    y = df["Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train = x_train.copy()
    train["Class"] = y_train
    test = x_test.copy()
    test["Class"] = y_test

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logger.success(f"Сохранены {train_path}, {test_path}")


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    test_size: float = 0.3,
    random_state: int = 42,
    skip_ge: bool = typer.Option(False, "--skip-ge"),
):
    run_preprocess(input_path, train_path, test_path, test_size, random_state, skip_ge)


if __name__ == "__main__":
    app()
