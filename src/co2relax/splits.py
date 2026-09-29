"""Разбиения выборки для честной оценки.

Данные лежат на регулярной сетке 9x9x9. Случайный split даёт утечку:
у каждой тестовой точки соседи в 500 K остаются в обучении, и любая
модель покажет почти нулевую ошибку. Поэтому разбиения структурные.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEMP_COLS = ["T", "T12", "T3"]


def split_subsample(df: pd.DataFrame, step: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прореживание сетки: обучение на каждом step-м узле, тест на остальных.

    При step=2 сетка 9x9x9 даёт 5x5x5=125 узлов в обучении и 604 в тесте.
    Тестовые точки лежат между обучающими — это честная проверка
    интерполяции с вдвое более крупным шагом (1000 K вместо 500 K).
    """
    mask = np.ones(len(df), dtype=bool)
    for c in TEMP_COLS:
        vals = np.sort(df[c].unique())
        mask &= df[c].isin(vals[::step]).to_numpy()
    return df[mask].copy(), df[~mask].copy()


def split_holdout_slices(
    df: pd.DataFrame, col: str = "T", values: list | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Исключение целых срезов по одной переменной.

    Проверяет интерполяцию по col: тестовые значения лежат внутри
    обучающего диапазона, но ни одна точка с такой col в обучение не попала.
    """
    if values is None:
        vals = np.sort(df[col].unique())
        values = [vals[len(vals) // 3], vals[2 * len(vals) // 3]]
    m = df[col].isin(values).to_numpy()
    return df[~m].copy(), df[m].copy()


def split_extrapolate(
    df: pd.DataFrame, col: str = "T3", n_out: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Экстраполяция: верхние n_out значений col уходят в тест.

    Самая жёсткая проверка — модель не видела этой области вовсе.
    """
    vals = np.sort(df[col].unique())
    m = df[col].isin(vals[-n_out:]).to_numpy()
    return df[~m].copy(), df[m].copy()


SPLITS = {
    "прореживание сетки": lambda d: split_subsample(d, 2),
    "срезы по T": lambda d: split_holdout_slices(d, "T"),
    "срезы по T12": lambda d: split_holdout_slices(d, "T12"),
    "экстраполяция по T3": lambda d: split_extrapolate(d, "T3", 2),
}
