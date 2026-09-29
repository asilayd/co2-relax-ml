"""Загрузка, проверка и подготовка данных по релаксационным членам CO2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .physics import THETA_2, THETA_3

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "CO2_CO2_relaxation_results_large.csv"
CORE_PATH = ROOT / "data" / "core.csv"
SAMPLE_PATH = ROOT / "data" / "sample.csv"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

TEMP_COLS = ["T", "T12", "T3"]
KEY_COLS = ["n_CO2", "partner", "T", "T12", "T3"]
R_COLS = ["R_VT2_12", "R_VT2_3", "R_VV23_12", "R_VV23_3", "R_VV123_12", "R_VV123_3"]
TARGETS = ["R_VT2_12", "R_VV23_12", "R_VV123_12"]
REDUNDANT = ["R_VT2_3", "R_VV23_3", "R_VV123_3"]


def load(path: Path | str | None = None) -> pd.DataFrame:
    """Сырые данные со столбцом плотности.

    Нужны только для проверки инварианта R ~ n^2 и построения core.
    Для всего остального используйте load_core().
    """
    if path is None:
        for p in (RAW_PATH, SAMPLE_PATH):
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError(
                f"Нет сырых данных. Ожидается {RAW_PATH} или {SAMPLE_PATH}.\n"
                f"Если нужен сжатый датасет — используйте load_core()."
            )
    df = pd.read_csv(path)
    df["partner"] = df["partner"].astype(str)
    return df


def load_core(rebuild: bool = False) -> pd.DataFrame:
    """Сжатый датасет: 729 физически различных точек, без плотности.

    Читает data/core.csv, если он есть. Иначе строит его из сырых данных
    и сохраняет — файл весит ~60 КБ и хранится в репозитории, поэтому
    после первого запуска полные данные больше не нужны.
    """
    if CORE_PATH.exists() and not rebuild:
        df = pd.read_csv(CORE_PATH)
        df["partner"] = df["partner"].astype(str)
        return df

    core = to_core(load())
    CORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(CORE_PATH, index=False)
    return core


def to_core(df: pd.DataFrame) -> pd.DataFrame:
    """Сжимает данные до физически различных точек.

    Плотность факторизуется точно (R ~ n^2), поэтому строки, отличающиеся
    только n_CO2, несут одну и ту же информацию. Оставляем по одной,
    делим таргеты на n^2 (получается эффективный коэффициент, м^3/с)
    и выбрасываем столбец плотности.
    """
    keys = [c for c in ("partner", *TEMP_COLS) if c in df.columns]
    core = df.drop_duplicates(subset=keys).copy()
    if "n_CO2" in core.columns:
        n2 = core["n_CO2"] ** 2
        for c in R_COLS:
            if c in core.columns:
                core[c] = core[c] / n2
        core = core.drop(columns=["n_CO2"])
    return core.sort_values(keys).reset_index(drop=True)


def grid_summary(df: pd.DataFrame) -> dict:
    """Какие значения принимают ключевые столбцы и полна ли сетка."""
    keys = [c for c in KEY_COLS if c in df.columns]
    out: dict = {c: sorted(df[c].unique().tolist()) for c in keys}
    out["rows"] = len(df)
    out["full_grid_rows"] = int(np.prod([df[c].nunique() for c in keys]))
    out["grid_is_full"] = out["rows"] == out["full_grid_rows"]
    out["duplicates_by_key"] = int(df.duplicated(subset=keys).sum())
    out["nan_total"] = int(df.isna().sum().sum())
    return out


def check_invariants(df: pd.DataFrame, rtol: float = 1e-9) -> dict:
    """Четыре точных соотношения, которые должны выполняться в данных.

    Четвёртое (R ~ n^2) проверяемо только на сырых данных с несколькими
    плотностями; на core пропускается.
    """
    res: dict = {}

    # 1. VT2 не трогает моду 3
    res["VT2_3_all_zero"] = bool((df["R_VT2_3"] == 0).all())

    # 2. Сохранение квантов: R_3 = -R_12 / 3
    for p in ("VV23", "VV123"):
        a, b = df[f"R_{p}_12"], df[f"R_{p}_3"]
        res[f"{p}_ratio_exact"] = bool(np.allclose(b, -a / 3, rtol=rtol, atol=0))
        m = a.abs() > 0
        r = b[m] / a[m]
        res[f"{p}_ratio_min_max"] = (float(r.min()), float(r.max()))

    # 3. Ноль на равновесии T = T12 = T3
    eq = df[(df["T"] == df["T12"]) & (df["T12"] == df["T3"])]
    res["equilibrium_rows"] = len(eq)
    if len(eq):
        arr = np.abs(df[TARGETS].to_numpy())
        nonzero = arr[arr > 0]
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
        eq_max = float(eq[R_COLS].abs().to_numpy().max())
        res["equilibrium_max_abs_R"] = eq_max
        res["equilibrium_rel_to_median"] = eq_max / scale
    else:
        res["equilibrium_max_abs_R"] = None

    # 4. R ~ n^2
    if "n_CO2" not in df.columns:
        res["n2_scaling"] = "нет столбца n_CO2 (это core) — проверяется на сырых данных"
        return res

    n_vals = sorted(df["n_CO2"].unique().tolist())
    res["n_CO2_count"] = len(n_vals)
    res["n_CO2_range"] = (n_vals[0], n_vals[-1])
    if len(n_vals) > 1:
        scaled = df[TARGETS].div(df["n_CO2"] ** 2, axis=0)
        scaled[["partner", *TEMP_COLS]] = df[["partner", *TEMP_COLS]]
        g = scaled.groupby(["partner", *TEMP_COLS])[TARGETS]
        spread = g.agg(lambda s: np.ptp(s.to_numpy()) / max(np.abs(s).max(), 1e-300))
        res["n2_scaling_max_rel_spread"] = float(spread.to_numpy().max())
    else:
        res["n2_scaling"] = "одна плотность — проверить нельзя"

    return res


def _group_keys(df: pd.DataFrame) -> list[str]:
    """Ключи группировки: всё, что есть в таблице, кроме T3."""
    return [c for c in ("n_CO2", "partner", "T", "T12") if c in df.columns]


def vt2_t3_spread(df: pd.DataFrame) -> pd.Series:
    """Насколько R_VT2_12 меняется по T3 при фиксированных остальных параметрах.

    Относительный размах. Если везде ~1e-6 — VT2 фактически не зависит от T3.
    """
    g = df.groupby(_group_keys(df))["R_VT2_12"]
    return g.agg(lambda s: np.ptp(s.to_numpy()) / max(np.abs(s).max(), 1e-300))


def sign_crossings(df: pd.DataFrame, target: str = "R_VV23_12") -> pd.DataFrame:
    """Где по T3 меняется знак target при фиксированных остальных параметрах.

    Ноль уточняется линейной интерполяцией между соседними узлами —
    грубо при шаге 500 K, но для разведки достаточно.
    Работает и на сырых данных, и на core без n_CO2.
    """
    keys = _group_keys(df)
    rows = []
    for vals, g in df.groupby(keys):
        vals = vals if isinstance(vals, tuple) else (vals,)
        g = g.sort_values("T3")
        t3 = g["T3"].to_numpy()
        r = g[target].to_numpy()
        idx = np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
        for i in idx:
            t0 = t3[i] - r[i] * (t3[i + 1] - t3[i]) / (r[i + 1] - r[i])
            rows.append({**dict(zip(keys, vals)),
                         "T3_lo": t3[i], "T3_hi": t3[i + 1], "T3_zero": t0})
    return pd.DataFrame(rows)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет обратные безразмерные температуры как признаки.

    Аррениусовская структура линейна по 1/T, поэтому обратные температуры
    работают лучше сырых кельвинов.
    """
    out = df.copy()
    out["x_T"] = 1000.0 / out["T"]
    out["x_12"] = THETA_2 / out["T12"]
    out["x_3"] = THETA_3 / out["T3"]
    return out