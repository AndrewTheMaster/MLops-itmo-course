from pathlib import Path

from loguru import logger
import typer
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from creditcard_fraud.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Загрузка исходного ARFF-датаcета, базовая предобработка и разбиение
    на train/test, совместимое с EDA-ноутбуком.
    """
    logger.info(f"Loading raw data from {input_path}")
    data, _ = loadarff(input_path)
    df = pd.DataFrame(data)

    # Удаляем выбросы по признаку Amount (как в ноутбуке)
    q1 = df["Amount"].quantile(0.25)
    q3 = df["Amount"].quantile(0.75)
    iqr = q3 - q1
    df = df[
        ~(
            (df["Amount"] < (q1 - 1.5 * iqr))
            | (df["Amount"] > (q3 + 1.5 * iqr))
        )
    ]

    # Масштабирование Time и Amount
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    # Разделение на признаки и целевую переменную
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

    logger.info(f"Saving train data to {train_path}")
    train.to_csv(train_path, index=False)

    logger.info(f"Saving test data to {test_path}")
    test.to_csv(test_path, index=False)

    logger.success("Train/test split created successfully.")


if __name__ == "__main__":
    app()
