"""Модели-суррогаты релаксационных членов.

Ключевая идея: множитель детального баланса [1 - exp(-phi)] входит в модель
структурно, а не извлекается из данных делением. Сеть предсказывает
предэкспоненциальный множитель A, а на выходе он умножается на factor(phi).

Это даёт:
  * точное обращение R в ноль на равновесии при любых весах сети;
  * правильный знак и положение нулевой поверхности;
  * отсутствие деления на малую величину при подготовке меток.

Функция потерь вычисляется по R, а не по A, поэтому плохо обусловленная
окрестность нулевой поверхности не портит обучение: там множитель мал,
и вклад этих точек в градиент естественно подавлен.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .physics import AFFINITY, THETA_2, THETA_3

TEMP_COLS = ["T", "T12", "T3"]


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_features(df) -> np.ndarray:
    """Обратные безразмерные температуры.

    Аррениусовская структура линейна по 1/T, поэтому обратные температуры
    работают лучше сырых кельвинов. Масштабы подобраны так, чтобы признаки
    были порядка единицы.
    """
    return np.stack([
        1000.0 / df["T"].to_numpy(),
        THETA_2 / df["T12"].to_numpy(),
        THETA_3 / df["T3"].to_numpy(),
    ], axis=1).astype(np.float32)


def make_factors(df, targets: list[str]) -> np.ndarray:
    """Множители [1 - exp(-phi)] по каждому каналу."""
    out = []
    for t in targets:
        phi = AFFINITY[t](df["T"], df["T12"], df["T3"])
        out.append(-np.expm1(-np.asarray(phi, dtype=float)))
    return np.stack(out, axis=1).astype(np.float32)


class PhysicsInformedMLP(nn.Module):
    """R = A(x) * [1 - exp(-phi)], где A предсказывается сетью.

    Сеть выдаёт log A, поэтому A положителен по построению, а выход
    охватывает несколько порядков без насыщения.
    """

    def __init__(self, n_out: int = 3, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = 3
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers.append(nn.Linear(d_in, n_out))
        self.net = nn.Sequential(*layers)

        # Нормировка входов и масштаб выхода заполняются в fit_scalers.
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))
        self.register_buffer("log_a0", torch.zeros(n_out))

    def fit_scalers(self, x: torch.Tensor, a_scale: torch.Tensor) -> None:
        """Запоминает нормировку входов и характерный масштаб A."""
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-8))
        self.log_a0.copy_(torch.log(a_scale.clamp_min(1e-300)))

    def log_A(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.x_mean) / self.x_std
        return self.net(z) + self.log_a0

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """x — признаки (N, 3), f — множители (N, n_out). Возвращает R."""
        return torch.exp(self.log_A(x)) * f


class PlainMLP(nn.Module):
    """Модель без физической структуры — для сравнения.

    Предсказывает R напрямую в нормированных единицах. Ничего не гарантирует
    на равновесии: именно этот разрыв и должен быть виден в метриках.
    """

    def __init__(self, n_out: int = 3, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = 3
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers.append(nn.Linear(d_in, n_out))
        self.net = nn.Sequential(*layers)
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))
        self.register_buffer("y_scale", torch.ones(n_out))

    def fit_scalers(self, x: torch.Tensor, y_scale: torch.Tensor) -> None:
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-8))
        self.y_scale.copy_(y_scale.clamp_min(1e-300))

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        """f не используется — сигнатура общая с PhysicsInformedMLP."""
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_scale


MODELS = {"physics": PhysicsInformedMLP, "plain": PlainMLP}