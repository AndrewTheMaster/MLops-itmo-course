from pathlib import Path

import mlflow
import mlflow.sklearn
from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import typer

from creditcard_fraud.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _load_train_test(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info(f"Loading train data from {train_path}")
    train = pd.read_csv(train_path)

    logger.info(f"Loading test data from {test_path}")
    test = pd.read_csv(test_path)

    x_train = train.drop("Class", axis=1)
    y_train = train["Class"]

    x_test = test.drop("Class", axis=1)
    y_test = test["Class"]

    return x_train, y_train, x_test, y_test


def _log_and_train_model(
    name: str,
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    logger.info(f"Training model: {name}")

    with mlflow.start_run(run_name=name):
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        mlflow.log_metrics(
            {
                "recall_fraud": recall,
                "precision_fraud": precision,
                "f1_fraud": f1,
            }
        )

        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            f"{name}: recall={recall:.3f}, precision={precision:.3f}, f1={f1:.3f}"
        )


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "creditcard_fraud_baseline",
):
    """
    Базовое сравнение LogisticRegression и RandomForest с логированием в MLflow.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = _load_train_test(train_path, test_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    _log_and_train_model("LogisticRegression", logreg, x_train, y_train, x_test, y_test)
    _log_and_train_model("RandomForest", rf, x_train, y_train, x_test, y_test)

    logger.success("Training and MLflow logging complete.")


if __name__ == "__main__":
    app()
