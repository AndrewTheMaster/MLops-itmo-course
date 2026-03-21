"""
Проверки качества interim-данных (аналог минимального suite в Great Expectations).

Пакет ``great-expectations==0.18.x`` в метаданных PyPI требует ``numpy<2``, что
несовместимо с Python 3.13 + NumPy 2.x + MLflow 2.20. Логика ожиданий сохранена
явно — те же правила, что ``PandasDataset.expect_*`` в учебном suite.
"""

from __future__ import annotations

import pandas as pd


def validate_interim_fraud_dataset(df: pd.DataFrame) -> None:
    errors: list[str] = []

    if len(df) < 100:
        errors.append(f"ожидалось >= 100 строк, получено {len(df)}")

    for col in ("Class", "Time", "Amount"):
        if col not in df.columns:
            errors.append(f"нет колонки {col!r}")

    if "Class" in df.columns:
        bad = ~df["Class"].isin([0, 1])
        if bad.any():
            errors.append(f"Class вне {{0,1}}: {df.loc[bad, 'Class'].unique()[:5]!r}")

    if errors:
        raise ValueError("Проверки качества данных: " + "; ".join(errors))
