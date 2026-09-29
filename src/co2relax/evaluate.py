"""Единая оценка моделей: точность, ограничения, скорость."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .baselines import GridSpline, ScatteredSpline, extract_A, metrics
from .data import TARGETS
from .physics import AFFINITY, factor


def constraint_checks(true: np.ndarray, pred: np.ndarray, scale: float,
                      is_equilibrium: np.ndarray | None = None) -> dict:
    """Проверки физических ограничений.

    В отличие от метрик точности агрегируются по максимуму: ограничение
    либо выполняется везде, либо нарушено.
    """
    out = {}
    if is_equilibrium is not None and is_equilibrium.any():
        out["равновесие"] = float(np.abs(pred[is_equilibrium]).max() / scale)

    # доля точек с неверным знаком (нули не учитываем)
    nz = true != 0
    out["доля_неверный_знак"] = float(
        (np.sign(pred[nz]) != np.sign(true[nz])).mean()) if nz.any() else 0.0
    return out


def eval_spline(train: pd.DataFrame, test: pd.DataFrame, target: str,
                on_A: bool, scale: float) -> dict:
    """Сплайн-бейзлайн в одном из двух режимов."""
    col, tr = target, train
    if on_A:
        tr = train.assign(_A=extract_A(train, target))
        col = "_A"
    try:
        model = GridSpline().fit(tr, col)
    except ValueError:
        model = ScatteredSpline().fit(tr, col)
    pred = model.predict(test)
    if on_A:
        pred = pred * factor(AFFINITY[target](test["T"], test["T12"], test["T3"]))

    true = test[target].to_numpy()
    eq = ((test["T"] == test["T12"]) & (test["T12"] == test["T3"])).to_numpy()
    return {**metrics(true, pred), **constraint_checks(true, pred, scale, eq)}


def eval_net(net, test: pd.DataFrame, scales: np.ndarray,
             targets: list[str] = TARGETS) -> dict[str, dict]:
    """Сеть по всем каналам сразу."""
    from .train import predict

    pred = predict(net, test, targets)
    eq = ((test["T"] == test["T12"]) & (test["T12"] == test["T3"])).to_numpy()
    out = {}
    for j, t in enumerate(targets):
        true = test[t].to_numpy()
        out[t] = {**metrics(true, pred[:, j]),
                  **constraint_checks(true, pred[:, j], scales[j], eq)}
    return out


def timing(net, test: pd.DataFrame, n_repeat: int = 20,
           targets: list[str] = TARGETS) -> dict:
    """Время одного вызова модели, мкс на точку.

    Сравнивается с временем интерполяции на решётке. Обе величины —
    накладные расходы суррогата; прямой расчёт на порядки дороже,
    но здесь недоступен.
    """
    from .train import predict

    predict(net, test.head(8), targets)          # прогрев
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        predict(net, test, targets)
    dt_net = (time.perf_counter() - t0) / n_repeat / len(test) * 1e6

    return {"сеть_мкс_на_точку": dt_net, "точек": len(test)}


def compare_all(core: pd.DataFrame, splits: dict, cfgs: dict,
                targets: list[str] = TARGETS, seed: int = 0) -> pd.DataFrame:
    """Полное сравнение: сети и сплайны на всех разбиениях.

    cfgs — словарь {имя: Config} для нейросетевых моделей.
    Возвращает длинную таблицу: строка на (разбиение, канал, модель).
    """
    from .train import train_model

    scales = np.array([np.median(np.abs(core[t][core[t] != 0])) for t in targets])
    rows = []

    for sname, fn in splits.items():
        tr, te = fn(core)

        for mname, cfg in cfgs.items():
            cfg.seed = seed
            net, hist = train_model(tr, cfg, targets)
            res = eval_net(net, te, scales, targets)
            for t in targets:
                rows.append(dict(разбиение=sname, канал=t, модель=mname,
                                 эпох=len(hist), **res[t]))

        for mname, on_A in [("сплайн по R", False), ("сплайн по A", True)]:
            for j, t in enumerate(targets):
                r = eval_spline(tr, te, t, on_A, scales[j])
                rows.append(dict(разбиение=sname, канал=t, модель=mname,
                                 эпох=np.nan, **r))

    return pd.DataFrame(rows)
