"""Загрузка сырого ARFF (OpenML) → data/interim/dataset.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from scipy.io.arff import loadarff
import typer

from creditcard_fraud.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def _bytes_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna().head(20)
        if len(sample) == 0:
            continue
        if all(isinstance(v, (bytes, bytearray)) for v in sample):
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8")
                if isinstance(x, (bytes, bytearray))
                else x
            )
    return df


def run_load(
    input_path: Path = RAW_DATA_DIR / "dataset",
    output_path: Path = INTERIM_DATA_DIR / "dataset.csv",
) -> None:
    logger.info(f"Загрузка ARFF из {input_path}")
    data, _ = loadarff(input_path)
    df = pd.DataFrame(data)
    df = _bytes_columns_to_str(df)
    if "Class" in df.columns:
        df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
        if df["Class"].isna().any():
            raise ValueError("Некорректные значения в Class.")
        df["Class"] = df["Class"].astype(int)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Сохранено {output_path} ({len(df)} строк).")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset",
    output_path: Path = INTERIM_DATA_DIR / "dataset.csv",
):
    run_load(input_path, output_path)


if __name__ == "__main__":
    app()
