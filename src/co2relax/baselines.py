"""Метрики и базовые модели, с которыми сравнивается нейросеть."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata

from .physics import AFFINITY, factor

TEMP_COLS = ["T", "T12", "T3"]


def metrics(true: np.ndarray, pred: np.ndarray, rel_floor: float = 1e-3) -> dict:
    """Ошибки предсказания.

    Относительная ошибка считается только там, где |true| заметно больше нуля:
    у нулевой поверхности она взрывается по построению и ничего не говорит
    о качестве. Окрестность нуля оценивается абсолютной ошибкой,
    нормированной на характерный масштаб величины.
    """
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    err = np.abs(pred - true)
    scale = np.median(np.abs(true[true != 0])) if np.any(true != 0) else 1.0

    big = np.abs(true) > rel_floor * np.abs(true).max()
    rel = err[big] / np.abs(true[big])

    return {
        "n": int(len(true)),
        "rel_median": float(np.median(rel)) if big.any() else np.nan,
        "rel_p95": float(np.percentile(rel, 95)) if big.any() else np.nan,
        "rel_max": float(rel.max()) if big.any() else np.nan,
        "abs_max_over_scale": float(err.max() / scale),
        "r2": float(1 - (err**2).sum() / ((true - true.mean()) ** 2).sum()),
    }



def extract_A(df: pd.DataFrame, target: str, tol: float = 1e-9) -> pd.Series:
    """Предэкспоненциальный множитель A = R / [1 - exp(-phi)].
 
    На равновесии обращаются в ноль и числитель, и знаменатель, поэтому там
    A не определён и возвращается NaN. Равновесие определяется по допуску:
    phi должен быть точным нулём, но в арифметике с плавающей точкой выходит
    ~1e-16, и без допуска получилось бы A = 0/1e-16 = 0 вместо NaN.
    Физические значения |phi| — порядка единиц, так что tol=1e-9 их не заденет.
    """
    phi = np.asarray(AFFINITY[target](df["T"], df["T12"], df["T3"]), dtype=float)
    f = factor(phi)
    A = np.where(np.abs(phi) > tol, df[target].to_numpy() / np.where(f != 0, f, 1.0), np.nan)
    return pd.Series(A, index=df.index, name=f"A_{target}")

def _fill_nan(axes: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Заполняет пропуски в решётке ближайшими значениями.

    Нужно для A: на равновесии phi = 0, множитель обращается в ноль,
    и A формально не определён, хотя предел конечен.
    """
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    flat = grid.ravel()
    ok = ~np.isnan(flat)
    flat = flat.copy()
    flat[~ok] = griddata(pts[ok], flat[ok], pts[~ok], method="nearest")
    return flat.reshape(grid.shape)


class GridSpline:
    """Интерполяция на регулярной сетке — базовая модель для сравнения.

    Работает только если обучающие точки образуют полную решётку.
    Вне выпуклой оболочки узлов значения не определены; для таких точек
    используется ближайший узел, что честно отражает поведение таблицы
    при экстраполяции.
    """

    def __init__(self, method: str = "cubic"):
        self.method = method

    def fit(self, train: pd.DataFrame, target: str):
        axes = [np.sort(train[c].unique()) for c in TEMP_COLS]
        shape = tuple(len(a) for a in axes)
        if np.prod(shape) != len(train):
            raise ValueError(
                f"обучающие точки не образуют полную решётку: "
                f"{np.prod(shape)} узлов против {len(train)} строк"
            )
        grid = train.sort_values(TEMP_COLS)[target].to_numpy().reshape(shape)
        if np.isnan(grid).any():  # A не определён на равновесии — заполняем соседями
            grid = _fill_nan(axes, grid)
        # Значения порядка 1e-17: без нормировки кубический сплайн молча
        # возвращает нули — система решается на грани машинной точности.
        nz = np.abs(grid[grid != 0])
        self._scale = float(np.median(nz)) if nz.size else 1.0
        self._interp = RegularGridInterpolator(
            axes, grid / self._scale, method=self.method,
            bounds_error=False, fill_value=None,
        )
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self._interp(test[TEMP_COLS].to_numpy()) * self._scale


class ScatteredSpline:
    """Интерполяция по неструктурированным точкам (когда решётки нет)."""

    def __init__(self, method: str = "linear"):
        self.method = method

    def fit(self, train: pd.DataFrame, target: str):
        self._pts = train[TEMP_COLS].to_numpy()
        self._val = train[target].to_numpy()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        out = griddata(self._pts, self._val, test[TEMP_COLS].to_numpy(),
                       method=self.method)
        nan = np.isnan(out)
        if nan.any():  # вне оболочки — ближайший узел
            out[nan] = griddata(self._pts, self._val,
                                test[TEMP_COLS].to_numpy()[nan], method="nearest")
        return out


def run_baseline(train, test, target, model=None, on_A: bool = False) -> dict:
    """Обучает базовую модель и считает метрики в единицах исходного R.

    on_A=True — интерполируется множитель A, затем R восстанавливается
    умножением на [1 - exp(-phi)]. Так знак и нулевая поверхность
    воспроизводятся точно даже базовой моделью.
    """
    model = model or ScatteredSpline()
    col = target

    if on_A:
        train = train.assign(_A=extract_A(train, target))
        col = "_A"

    model.fit(train, col)
    pred = model.predict(test)

    if on_A:
        phi = AFFINITY[target](test["T"], test["T12"], test["T3"])
        pred = pred * factor(phi)

    return metrics(test[target].to_numpy(), pred)
