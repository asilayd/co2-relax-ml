"""Обучение и оценка суррогатных моделей."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch

from .data import TARGETS
from .models import MODELS, make_factors, make_features, device
from .physics import THETA_2, THETA_3


@dataclass
class Config:
    model: str = "physics"
    hidden: int = 64
    depth: int = 3
    epochs: int = 3000
    lr: float = 3e-3
    weight_decay: float = 1e-6
    seed: int = 0
    patience: int = 400       # ранняя остановка по обучающей потере
    val_frac: float = 0.15    # доля обучающих точек под валидацию


def _relative_loss(pred: torch.Tensor, true: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    """Потери, нормированные на масштаб каждого канала.

    Без нормировки функция потерь определялась бы каналом с наибольшими
    значениями, и остальные модель бы проигнорировала. Знаменатель
    сглажен масштабом, чтобы окрестность нуля не взрывала градиент.
    """
    return (((pred - true) / (true.abs() + 0.05 * scale)) ** 2).mean()


def train_model(train: pd.DataFrame, cfg: Config = Config(),
                targets: list[str] = TARGETS, verbose: bool = False):
    """Обучает модель на train. Возвращает (модель, история потерь)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = device()

    X = torch.tensor(make_features(train), device=dev)
    F = torch.tensor(make_factors(train, targets), device=dev)
    Y = torch.tensor(train[targets].to_numpy(dtype=np.float32), device=dev)

    # масштаб каждого канала — медиана модуля ненулевых значений
    scale = torch.tensor(
        [np.median(np.abs(train[t][train[t] != 0])) for t in targets],
        dtype=torch.float32, device=dev,
    )

    net = MODELS[cfg.model](n_out=len(targets), hidden=cfg.hidden,
                            depth=cfg.depth).to(dev)
    if cfg.model == "physics":
        # характерный масштаб A: R / f вдали от нулевой поверхности
        big = F.abs() > 0.3
        a0 = torch.stack([
            (Y[:, j][big[:, j]] / F[:, j][big[:, j]]).abs().median()
            if big[:, j].any() else scale[j]
            for j in range(len(targets))
        ])
        net.fit_scalers(X, a0)
    else:
        net.fit_scalers(X, scale)

    # валидация для ранней остановки
    n = len(train)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
    n_val = max(1, int(cfg.val_frac * n))
    vi, ti = idx[:n_val].to(dev), idx[n_val:].to(dev)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(cfg.epochs):
        net.train()
        opt.zero_grad()
        loss = _relative_loss(net(X[ti], F[ti]), Y[ti], scale)
        loss.backward()
        opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            vl = _relative_loss(net(X[vi], F[vi]), Y[vi], scale).item()
        hist.append((loss.item(), vl))

        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if verbose and ep % 500 == 0:
            print(f"  эпоха {ep:5d}  train {loss.item():.3e}  val {vl:.3e}")

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, np.array(hist)


@torch.no_grad()
def predict(net, df: pd.DataFrame, targets: list[str] = TARGETS) -> np.ndarray:
    dev = next(net.parameters()).device
    X = torch.tensor(make_features(df), device=dev)
    F = torch.tensor(make_factors(df, targets), device=dev)
    return net(X, F).cpu().numpy()


def jacobian(net, df: pd.DataFrame, targets: list[str] = TARGETS) -> np.ndarray:
    """dR/d(T, T12, T3) — то, что нужно неявным схемам в солвере.

    Считается автоматическим дифференцированием по всей цепочке, включая
    множитель [1 - exp(-phi)]: температуры входят и в признаки сети, и в
    сродство, обе зависимости учитываются. Табличная интерполяция такой
    возможности не даёт. Форма результата (N, n_out, 3).
    """
    from .physics import AFFINITY_TORCH

    dev = next(net.parameters()).device
    Tn = torch.tensor(df[["T", "T12", "T3"]].to_numpy(dtype=np.float32),
                      device=dev, requires_grad=True)

    x = torch.stack([1000.0 / Tn[:, 0],
                     THETA_2 / Tn[:, 1],
                     THETA_3 / Tn[:, 2]], dim=1)
    phi = torch.stack([AFFINITY_TORCH[t](Tn[:, 0], Tn[:, 1], Tn[:, 2])
                       for t in targets], dim=1)
    f = -torch.expm1(-phi)
    r = net(x, f)

    out = np.zeros((len(df), len(targets), 3), dtype=np.float32)
    for j in range(len(targets)):
        g = torch.autograd.grad(r[:, j].sum(), Tn, retain_graph=True)[0]
        out[:, j] = g.detach().cpu().numpy()
    return out