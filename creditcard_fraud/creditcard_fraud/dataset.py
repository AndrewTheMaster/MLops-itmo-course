"""Последовательно: load_data → preprocess (для ручного запуска; DVC — отдельные стадии)."""

from __future__ import annotations

from loguru import logger
import typer

from creditcard_fraud.load_data import run_load
from creditcard_fraud.preprocess import run_preprocess

app = typer.Typer()


@app.command()
def main():
    logger.info("load_data…")
    run_load()
    logger.info("preprocess…")
    run_preprocess()


if __name__ == "__main__":
    app()
