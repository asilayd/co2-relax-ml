"""Рисунки и таблицы отчёта.

Скрипт занимается только отрисовкой и сохранением: вся физика, модели,
разбиения и метрики берутся из пакета co2relax. Имена файлов совпадают с
теми, на которые ссылается report/sections/*.tex, поэтому достаточно
скопировать полученные PDF в report/figures/ — править текст не нужно.

    python scripts/make_figures.py                 # всё
    python scripts/make_figures.py --only phys res
    python scripts/make_figures.py --quick         # быстрая проверка, ~1 мин
    python scripts/make_figures.py --recompute     # пересчитать сравнение и свип

Недостающие reports/03_comparison.csv и reports/04_sweep.csv скрипт
считает сам, так что порядок запуска скриптов значения не имеет.

Кроме рисунков сохраняются:
    reports/TABLES.tex    таблицы отчёта, вставить в sections/05_results.tex
    reports/NUMBERS.txt   числа, встречающиеся в тексте прозой
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))          # пакет co2relax
sys.path.insert(0, str(Path(__file__).resolve().parent))   # соседний figstyle.py

import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

from co2relax.baselines import extract_A, metrics, spline_predict  # noqa: E402
from co2relax.data import (RAW_PATH, REPORT_DIR, TARGETS,   # noqa: E402
                           load, load_core, vt2_t3_spread)
from co2relax.evaluate import (compare_all, constraint_checks,  # noqa: E402
                               eval_net, timing)
from co2relax.physics import (AFFINITY, DELTA_VV123, DELTA_VV23,  # noqa: E402
                              THETA_1, THETA_2, THETA_3)
from co2relax.splits import SPLITS                   # noqa: E402
from co2relax.train import Config, jacobian, predict, train_model  # noqa: E402
from figstyle import (BLUE, CHANNEL, DIV, GRID, INK, MODEL,  # noqa: E402
                      MUTED, ORANGE, SEQ, out_dir, save, set_out, use_style)

BEST = dict(hidden=128, depth=4, epochs=6000, lr=3e-3)
QUICK = dict(hidden=32, depth=2, epochs=800, lr=3e-3)
SWEEP_GRID = dict(hidden=[16, 32, 64, 128], depth=[2, 3, 4],
                  epochs=[1500, 6000], lr=[3e-3])
SWEEP_GRID_QUICK = dict(hidden=[16, 32], depth=[2, 3], epochs=[800], lr=[3e-3])

CMP_PATH = REPORT_DIR / "03_comparison.csv"
SWEEP_PATH = REPORT_DIR / "04_sweep.csv"

PHI_EDGES = [0, 0.05, 0.15, 0.4, 1.0, 3.0]
PHI_LBL = ["<0.05", "0.05–\n0.15", "0.15–\n0.4", "0.4–1", "1–3"]

NUMBERS: dict[str, object] = {}     # числа для NUMBERS.txt


def phi_of(df, target):
    return np.asarray(AFFINITY[target](df["T"], df["T12"], df["T3"]), dtype=float)


def scales_of(core):
    return np.array([np.median(np.abs(core[t][core[t] != 0])) for t in TARGETS])


# =======================================================================
#  ДАННЫЕ
# =======================================================================

def fig_density(core):
    """Плотность выносится из R точно: R ~ n^2. Основание сжатия датасета."""
    if not RAW_PATH.exists():
        print("  [пропуск] data_density — нет data/raw")
        return
    raw = load(RAW_PATH)
    states = (raw[["T", "T12", "T3"]].drop_duplicates()
              .query("T != T12 or T12 != T3").sample(14, random_state=0))

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for _, s in states.iterrows():
        g = raw[(raw["T"] == s["T"]) & (raw["T12"] == s["T12"])
                & (raw["T3"] == s["T3"])].sort_values("n_CO2")
        ax[0].plot(g.n_CO2, np.abs(g.R_VV23_12), color=BLUE, alpha=0.45, lw=1.1)
        ax[1].plot(g.n_CO2, np.abs(g.R_VV23_12) / g.n_CO2 ** 2,
                   color=BLUE, alpha=0.45, lw=1.1)
    ax[0].set(xscale="log", yscale="log", xlabel="$n$, м$^{-3}$",
              ylabel="$|R_{VV23}|$, м$^{-3}$с$^{-1}$", title="исходные значения")
    ax[1].set(xscale="log", yscale="log", xlabel="$n$, м$^{-3}$",
              ylabel="$|R_{VV23}|\\,/\\,n^2$", title="делённые на $n^2$")

    res = []
    for t in TARGETS:
        g = raw.assign(_v=raw[t] / raw.n_CO2 ** 2).groupby(["T", "T12", "T3"])["_v"]
        rel = (g.max() - g.min()) / g.apply(lambda s: np.abs(s).max()).replace(0, np.nan)
        res.append(rel.dropna())
    allres = pd.concat(res)
    med = float(np.median(allres))
    pos = allres[allres > 0]
    if len(pos):
        ax[2].hist(np.log10(pos), bins=30, color=BLUE, edgecolor="white", lw=0.4)
    ax[2].set(xlabel="$\\lg$ относительной невязки $R/n^2$", ylabel="состояний",
              title=f"невязка факторизации, медиана {med:.1e}")
    ax[2].grid(axis="x", alpha=0)
    save(fig, "data_density")
    NUMBERS["данные: строк в сыром файле"] = len(raw)
    NUMBERS["данные: значений плотности"] = int(raw["n_CO2"].nunique())
    NUMBERS["данные: невязка факторизации R/n^2"] = f"{med:.1e}"


def fig_domain(core):
    """Область определения и её особые множества."""
    phi = {t: phi_of(core, t) for t in TARGETS}
    eq = (core["T"] == core["T12"]) & (core["T12"] == core["T3"])
    small = np.abs(phi["R_VV23_12"]) < 0.05

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax[0].scatter(core.T12, core.T3, s=12, color=MUTED, alpha=0.22,
                  edgecolor="none", label="узлы сетки")
    ax[0].scatter(core.T12[eq], core.T3[eq], s=40, color=ORANGE, marker="D",
                  zorder=3, label="равновесие $T{=}T_{12}{=}T_3$")
    ax[0].scatter(core.T12[small], core.T3[small], s=28, facecolor="none",
                  edgecolor=BLUE, lw=1.1, zorder=2, label="$|\\varphi_{VV23}|<0.05$")
    lo, hi = core.T3.min(), core.T3.max()
    ax[0].set(xlabel="$T_{12}$, K", ylabel="$T_3$, K",
              ylim=(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo)),
              title="сетка в проекции на $(T_{12},T_3)$")
    ax[0].legend(loc="upper left", handletextpad=0.3, borderpad=0.2)

    for t in TARGETS:
        ax[1].hist(phi[t], bins=40, histtype="step", lw=1.6,
                   color=CHANNEL[t]["c"], label=CHANNEL[t]["label"])
    ax[1].axvline(0, color=INK, lw=1.0, ls="--")
    ax[1].set(xlabel="сродство $\\varphi$", ylabel="узлов",
              title="распределение сродства")
    ax[1].legend()

    los, his, rs = [], [], np.random.RandomState(0)
    for i, t in enumerate(TARGETS):
        v = np.abs(core[t][core[t] != 0])
        ax[2].scatter(np.full(len(v), i) + rs.uniform(-.16, .16, len(v)), v,
                      s=6, color=CHANNEL[t]["c"], alpha=0.35, edgecolor="none")
        los.append(np.percentile(v, 1)); his.append(v.max())
    span = np.log10(max(his) / min(los))
    ax[2].set(yscale="log", ylim=(min(los) / 3, max(his) * 3), xticks=range(3),
              xticklabels=[CHANNEL[t]["label"] for t in TARGETS],
              ylabel="$|R|$, м$^{-3}$с$^{-1}$",
              title="распределение $|R|$")
    ax[2].grid(axis="x", alpha=0)
    save(fig, "data_domain")
    NUMBERS["область: узлов равновесия"] = int(eq.sum())
    NUMBERS["область: узлов с |phi|<0.05"] = int(small.sum())
    NUMBERS["область: динамический диапазон, порядков"] = f"{span:.1f}"


def fig_vt2_t3(core):
    """VT2 не зависит от T3 — вход канала эффективно двумерный."""
    spread = vt2_t3_spread(core)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))

    pairs = core[["T", "T12"]].drop_duplicates().query("T != T12")
    for _, s in pairs.sample(min(9, len(pairs)), random_state=1).iterrows():
        g = core[(core["T"] == s["T"]) & (core["T12"] == s["T12"])].sort_values("T3")
        ax[0].plot(g.T3, g.R_VT2_12, marker="o", ms=3.5, color=BLUE, alpha=0.5)
    ax[0].set(xlabel="$T_3$, K", ylabel="$R_{VT2}$, м$^{-3}$с$^{-1}$",
              title="$R_{VT2}$ при фиксированных $(T, T_{12})$")

    t3min = core["T3"].min()
    tlo, thi = core["T"].min(), core["T"].max()
    for T in sorted(core["T"].unique())[::2]:
        g = core[(core["T"] == T) & (core["T3"] == t3min)].sort_values("T12")
        ax[1].plot(g.T12, g.R_VT2_12, marker="o", ms=3.5,
                   color=SEQ((T - tlo + 200) / (thi - tlo + 400)), label=f"{T:.0f} K")
    ax[1].axhline(0, color=GRID, lw=1.0)
    ax[1].set(xlabel="$T_{12}$, K", ylabel="$R_{VT2}$, м$^{-3}$с$^{-1}$",
              title="$R_{VT2}$ при наименьшем $T_3$")
    ax[1].legend(title="$T$", ncol=2, fontsize=7.5, title_fontsize=8)

    n_zero, med = int((spread == 0).sum()), float(np.median(spread))
    tail = (f"размах по $T_3$ тождественно нулевой во всех {n_zero} парах"
            if n_zero == len(spread) else f"медианный размах по $T_3$ равен {med:.1e}")
    save(fig, "data_vt2_t3")
    NUMBERS["VT2: пар (T,T12)"] = len(spread)
    NUMBERS["VT2: пар с нулевым размахом по T3"] = n_zero
    NUMBERS["VT2: медианный размах по T3"] = f"{med:.1e}"


# =======================================================================
#  ФИЗИКА
# =======================================================================

def fig_zero_surface(core):
    """Линия phi=0 по спектроскопии поверх смены знака в данных."""
    vals = np.sort(core["T"].unique())
    Ts = [vals[len(vals) // 4], vals[len(vals) // 2], vals[3 * len(vals) // 4]]
    t12 = np.linspace(core.T12.min(), core.T12.max(), 400)
    lo3, hi3 = core.T3.min(), core.T3.max()

    fig, axes = plt.subplots(len(Ts), 3, figsize=(10.2, 8.4), sharex=True, sharey=True)
    for r, T in enumerate(Ts):
        sl = core[core["T"] == T]
        for c, t in enumerate(TARGETS):
            ax = axes[r, c]
            piv = sl.pivot(index="T3", columns="T12", values=t)
            v = piv.to_numpy(); lim = np.abs(v).max()
            ax.pcolormesh(piv.columns, piv.index, np.sign(v) * np.abs(v) ** 0.35,
                          cmap=DIV, vmin=-lim ** 0.35, vmax=lim ** 0.35,
                          shading="nearest")
            if t == "R_VT2_12":
                for col, lw, ls in [(INK, 2.0, "-"), ("white", 0.7, (0, (4, 4)))]:
                    ax.axvline(T, color=col, lw=lw, ls=ls, zorder=4)
            else:
                den = (3 * THETA_2 / t12 + DELTA_VV23 / T) if t == "R_VV23_12" else \
                      ((THETA_1 + THETA_2) / t12 + DELTA_VV123 / T)
                t3 = THETA_3 / den
                ok = (t3 >= lo3) & (t3 <= hi3)
                ax.plot(t12[ok], t3[ok], color=INK, lw=2.0, zorder=4)
                ax.plot(t12[ok], t3[ok], color="white", lw=0.7, zorder=5, ls=(0, (4, 4)))
            if r == 0:
                ax.set_title(CHANNEL[t]["label"])
            if c == 0:
                ax.set_ylabel(f"$T = {T:.0f}$ K\n$T_3$, K")
            if r == len(Ts) - 1:
                ax.set_xlabel("$T_{12}$, K")
            ax.grid(False)
    fig.legend(handles=[Line2D([], [], color=INK, lw=2.0,
                               label="$\\varphi=0$ по спектроскопии"),
                        Line2D([], [], marker="s", ls="none", ms=9, color=BLUE,
                               label="$R>0$"),
                        Line2D([], [], marker="s", ls="none", ms=9, color=ORANGE,
                               label="$R<0$")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "phys_zero_surface")


def fig_affinity_residual(core):
    """Количественная проверка сродства.

    Нуль уточняется интерполяцией по 1/T3, а не по T3: сродство линейно
    именно по обратной температуре, поэтому вблизи нулевой поверхности R
    почти линейна в этой переменной и смещение интерполяции мало.
    """
    rows = []
    for (T_, T12_), g in core.groupby(["T", "T12"]):
        g = g.sort_values("T3")
        x, r = 1.0 / g["T3"].to_numpy(), g["R_VV23_12"].to_numpy()
        for i in np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]:
            x0 = x[i] - r[i] * (x[i + 1] - x[i]) / (r[i + 1] - r[i])
            rows.append(dict(T=T_, T12=T12_, T3_zero=1.0 / x0))
    cr = pd.DataFrame(rows)
    if len(cr) < 5:
        print("  [пропуск] phys_affinity_residual — мало переходов знака")
        return

    T, T12, z = cr["T"].to_numpy(), cr["T12"].to_numpy(), cr["T3_zero"].to_numpy()
    th = THETA_3 / (3 * THETA_2 / T12 + DELTA_VV23 / T)
    resid = z - th
    grid = np.linspace(200, 900, 2801)
    curve = np.array([np.sqrt(np.mean((z - THETA_3 / (3 * THETA_2 / T12 + d / T)) ** 2))
                      for d in grid])
    d_fit = float(grid[curve.argmin()])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    relmed = float(np.median(np.abs(resid) / th))
    step = float(np.diff(np.sort(core["T3"].unique())).min())

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax[0].scatter(th, z, s=22, color=BLUE, alpha=0.7, edgecolor="none")
    lo, hi = core.T3.min(), core.T3.max()
    ax[0].plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1.1)
    ax[0].set(xlabel="$T_3$ нуля по формуле, K", ylabel="$T_3$ нуля по данным, K",
              title=f"RMS {rms:.0f} K при шаге сетки {step:.0f} K")
    ax[1].hist(resid, bins=22, color=BLUE, edgecolor="white", lw=0.4)
    ax[1].axvline(0, color=INK, lw=1.0, ls="--")
    ax[1].set(xlabel="невязка $T_3$, K", ylabel="переходов",
              title=f"невязка положения нуля, медиана {relmed*100:.2f}\\%")
    ax[1].grid(axis="x", alpha=0)
    ax[2].plot(grid, curve, color=BLUE)
    ax[2].axvline(DELTA_VV23, color=ORANGE, lw=1.4, ls="--",
                  label=f"теория {DELTA_VV23:.0f} K")
    ax[2].axvline(d_fit, color=INK, lw=1.2, label=f"подгонка {d_fit:.1f} K")
    ax[2].set(xlabel="дефект энергии $\\Delta$, K", ylabel="RMS невязки, K",
              title="подгонка дефекта $\\Delta$")
    ax[2].legend()
    save(fig, "phys_affinity_residual")
    NUMBERS["сродство: RMS невязки нуля, K"] = f"{rms:.1f}"
    NUMBERS["сродство: шаг сетки по T3, K"] = f"{step:.0f}"
    NUMBERS["сродство: медианная отн. ошибка"] = f"{relmed*100:.3f}%"
    NUMBERS["сродство: подгонка Delta, K"] = f"{d_fit:.1f} (теория {DELTA_VV23:.0f})"
    NUMBERS["сродство: переходов знака"] = len(cr)


def fig_A_vs_R(core):
    """Зачем факторизация: A знакопостоянна там, где R меняет знак."""
    vals = np.sort(core["T"].unique())
    T = vals[len(vals) // 2]
    sl = core[core["T"] == T]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), sharex=True, sharey=True)
    for c, t in enumerate(TARGETS):
        piv = sl.pivot(index="T3", columns="T12", values=t)
        v = piv.to_numpy(); lim = np.abs(v).max()
        axes[0, c].pcolormesh(piv.columns, piv.index, np.sign(v) * np.abs(v) ** 0.35,
                              cmap=DIV, vmin=-lim ** 0.35, vmax=lim ** 0.35,
                              shading="nearest")
        axes[0, c].set_title(CHANNEL[t]["label"])
        pA = sl.assign(_A=extract_A(sl, t)).pivot(index="T3", columns="T12", values="_A")
        axes[1, c].pcolormesh(pA.columns, pA.index, np.log10(np.abs(pA.to_numpy())),
                              cmap=SEQ, shading="nearest")
        axes[1, c].set_xlabel("$T_{12}$, K")
        for r in (0, 1):
            axes[r, c].grid(False)
    axes[0, 0].set_ylabel("$R$ (знак меняется)\n$T_3$, K")
    axes[1, 0].set_ylabel("$\\lg A = \\lg\\,[R/(1-e^{-\\varphi})]$\n$T_3$, K")
    save(fig, "phys_A_vs_R")


def fig_conditioning(core):
    """Насколько устойчиво A извлекается делением вблизи нулевой поверхности."""
    rows = []
    for t in ["R_VV23_12", "R_VV123_12"]:
        d = core.assign(_A=extract_A(core, t).to_numpy(), _phi=phi_of(core, t))
        for _, g in d.groupby(["T", "T12"]):
            g = g.sort_values("T3")
            a = g["_A"].to_numpy()
            nb = np.full_like(a, np.nan)
            nb[1:-1] = 0.5 * (a[:-2] + a[2:])
            rows.append(pd.DataFrame(dict(phi=np.abs(g["_phi"].to_numpy()),
                                          dev=np.abs(a - nb) / np.abs(nb), канал=t)))
    d = pd.concat(rows).dropna()
    d = d[np.isfinite(d.dev)]

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for t in ["R_VV23_12", "R_VV123_12"]:
        s = d[d["канал"] == t]
        ax[0].scatter(s.phi, s.dev, s=9, color=CHANNEL[t]["c"], alpha=0.35,
                      edgecolor="none", label=CHANNEL[t]["label"])
    ax[0].set(xscale="log", yscale="log", xlabel="$|\\varphi|$",
              ylabel="отклонение $A$ от соседей по $T_3$",
              title="отклонение против $|\\varphi|$")
    for h in ax[0].legend().legend_handles:
        h.set_alpha(1.0)

    edges = [0, 0.05, 0.15, 0.4, 1.0, np.inf]
    lbl = ["<0.05", "0.05–0.15", "0.15–0.4", "0.4–1", ">1"]
    med, cnt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = d[(d.phi >= lo) & (d.phi < hi)]["dev"]
        med.append(np.median(s) * 100 if len(s) else np.nan); cnt.append(len(s))
    ax[1].bar(range(len(med)), med, color=BLUE, width=0.6)
    for i, (v, n) in enumerate(zip(med, cnt)):
        if np.isfinite(v):
            ax[1].text(i, v, f"{v:.2f}%\nn={n}", va="bottom", ha="center", fontsize=7.5)
    ax[1].set(xticks=range(len(med)), xticklabels=lbl, xlabel="$|\\varphi|$",
              ylabel="медианное отклонение, %", ylim=(0, np.nanmax(med) * 1.45),
              title="то же по корзинам")
    ax[1].grid(axis="x", alpha=0)
    save(fig, "phys_conditioning")
    NUMBERS["обусловленность: отклонение A по корзинам, %"] = \
        ", ".join(f"{l}: {v:.2f}" for l, v in zip(lbl, med))


# =======================================================================
#  ОБУЧЕНИЕ
# =======================================================================

def fig_splits(core):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1), sharex=True, sharey=True)
    for ax, (name, fn) in zip(axes, SPLITS.items()):
        tr, te = fn(core)
        ax.scatter(tr.T12, tr.T3, s=13, color=MUTED, alpha=0.3, edgecolor="none")
        ax.scatter(te.T12, te.T3, s=16, color=BLUE, alpha=0.75, edgecolor="none")
        ax.set(title=f"{name}\n{len(tr)} / {len(te)}", xlabel="$T_{12}$, K")
        ax.grid(False)
    axes[0].set_ylabel("$T_3$, K")
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", color=MUTED,
                               alpha=0.5, label="обучение"),
                        Line2D([], [], marker="o", ls="none", color=BLUE,
                               label="контроль")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    save(fig, "ml_splits")


def fig_learning(core, sweep, cfg):
    tr, _ = SPLITS["прореживание сетки"](core)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for mname, key in [("сеть с физикой", "physics"), ("сеть без физики", "plain")]:
        for seed in range(3):
            _, hist = train_model(tr, Config(model=key, seed=seed, **cfg))
            ax[0].plot(hist[:, 0], color=MODEL[mname]["c"],
                       alpha=0.6 if seed else 1.0, lw=1.3,
                       label=mname if seed == 0 else None)
    ax[0].set(yscale="log", xlabel="эпоха", ylabel="нормированная потеря",
              title="потери на обучении")
    ax[0].legend()

    if sweep is not None and len(sweep):
        g = sweep[sweep.epochs == sweep.epochs.max()].groupby("hidden")["эпох"] \
            .agg(["mean", "std"])
        ax[1].errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o",
                       color=BLUE, capsize=3)
        ax[1].set(xscale="log", xticks=g.index, xticklabels=g.index,
                  xlabel="ширина слоя", ylabel="эпох до ранней остановки",
                  title="эпох до ранней остановки")
    else:
        ax[1].axis("off")
    save(fig, "ml_learning")


def fig_pred_true(net, te):
    P = predict(net, te)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for j, (ax, t) in enumerate(zip(axes, TARGETS)):
        y, p = te[t].to_numpy(), P[:, j]
        m = y != 0
        lim = np.abs(y[m]).max() * 1.4
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, ls="--", lw=1.0, zorder=1)
        ax.scatter(y[m], p[m], s=11, color=CHANNEL[t]["c"], alpha=0.45,
                   edgecolor="none", zorder=2)
        ax.set(xscale="symlog", yscale="symlog",
               xlabel="данные, м$^{-3}$с$^{-1}$", title=CHANNEL[t]["label"])
        ax.grid(alpha=0.5)
    axes[0].set_ylabel("предсказание сети")
    save(fig, "ml_pred_true")


def fig_profiles(core, net):
    tv, t3v = np.sort(core["T"].unique()), np.sort(core["T3"].unique())
    T, T3 = tv[len(tv) // 2], t3v[2 * len(t3v) // 3]
    sl = core[(core["T"] == T) & (core["T3"] == T3)].sort_values("T12")
    fine = pd.DataFrame(dict(T12=np.linspace(core.T12.min(), core.T12.max(), 300)))
    fine["T"], fine["T3"] = T, T3
    P = predict(net, fine)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for j, (ax, t) in enumerate(zip(axes, TARGETS)):
        ax.axhline(0, color=GRID, lw=1.0)
        ax.plot(fine.T12, P[:, j], color=CHANNEL[t]["c"], lw=1.8, label="сеть")
        ax.scatter(sl.T12, sl[t], s=30, facecolor="none", edgecolor=INK, lw=1.0,
                   zorder=3, label="данные")
        phi0 = np.asarray(AFFINITY[t](T, fine.T12.to_numpy(), T3), dtype=float)
        for i in np.where(np.sign(phi0[:-1]) * np.sign(phi0[1:]) < 0)[0]:
            ax.axvline(fine.T12.to_numpy()[i], color=MUTED, ls=":", lw=1.2)
        ax.set(xlabel="$T_{12}$, K", title=CHANNEL[t]["label"])
    axes[0].set_ylabel("$R$, м$^{-3}$с$^{-1}$")
    axes[0].legend()
    save(fig, "ml_profiles")


# =======================================================================
#  РЕЗУЛЬТАТЫ
# =======================================================================

def fig_accuracy(cmp):
    splits = [s for s in SPLITS if s in set(cmp["разбиение"])]
    models = [m for m in MODEL if m in set(cmp["модель"])]
    w, off = 0.8 / len(models), (len(models) - 1) / 2
    fig, axes = plt.subplots(2, len(splits), figsize=(12.6, 6.2), sharey="row")
    for r, (col, ylab) in enumerate([("rel_median", "медианная отн. ошибка"),
                                     ("rel_p95", "отн. ошибка, 95-й процентиль")]):
        for c, s in enumerate(splits):
            ax = axes[r, c]
            sub = cmp[cmp["разбиение"] == s]
            for k, m in enumerate(models):
                v = [sub[(sub["модель"] == m) & (sub["канал"] == t)][col].iloc[0]
                     for t in TARGETS]
                ax.bar(np.arange(3) + (k - off) * w, v, width=w * 0.88,
                       color=MODEL[m]["c"], label=m if (r == 0 and c == 0) else None)
            ax.set(yscale="log", xticks=range(3),
                   xticklabels=[CHANNEL[t]["label"] for t in TARGETS])
            ax.grid(axis="x", alpha=0)
            if r == 0:
                ax.set_title(s)
        axes[r, 0].set_ylabel(ylab)
    fig.legend(loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.05))
    save(fig, "res_accuracy")


def fig_constraints(cmp):
    models = [m for m in MODEL if m in set(cmp["модель"])]
    eq = [cmp[cmp["модель"] == m]["равновесие"].max() for m in models]
    sg = [cmp[cmp["модель"] == m]["доля_неверный_знак"].max() for m in models]
    fl = 1e-17
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax[0].bar(range(len(models)), np.maximum(eq, fl),
              color=[MODEL[m]["c"] for m in models], width=0.55)
    ax[0].set(yscale="log", ylim=(fl, 3e2), xticks=range(len(models)),
              xticklabels=[m.replace(" ", "\n", 1) for m in models],
              ylabel="$\\max |R_{\\rm пред}| / R_*$ на равновесии",
              title="невязка равновесия $R(T,T,T)=0$")
    for i, v in enumerate(eq):
        ax[0].text(i, max(v, fl) * 1.7, f"{v:.0e}", ha="center", va="bottom", fontsize=8)
    ax[0].axhline(1e-15, color=MUTED, ls=":", lw=1.1)
    ax[0].grid(axis="x", alpha=0)
    ax[1].bar(range(len(models)), np.maximum(sg, 1e-4),
              color=[MODEL[m]["c"] for m in models], width=0.55)
    ax[1].set(yscale="log", ylim=(1e-4, 3), xticks=range(len(models)),
              xticklabels=[m.replace(" ", "\n", 1) for m in models],
              ylabel="доля точек", title="доля неверного знака $R$")
    for i, v in enumerate(sg):
        ax[1].text(i, max(v, 1e-4) * 1.7, "нет" if v == 0 else f"{v:.1e}",
                   ha="center", va="bottom", fontsize=8)
    ax[1].grid(axis="x", alpha=0)
    save(fig, "res_constraints")


def fig_error_vs_phi(core, cfg):
    """Ошибка как функция сродства: где сосредоточены отказы каждой модели."""
    scales = scales_of(core)
    rows = []
    for _, fn in SPLITS.items():
        tr, te = fn(core)
        nets = {}
        for mn, key in [("сеть с физикой", "physics"), ("сеть без физики", "plain")]:
            net, _ = train_model(tr, Config(model=key, seed=0, **cfg))
            nets[mn] = predict(net, te)
        for j, t in enumerate(TARGETS):
            true, phi = te[t].to_numpy(), np.abs(phi_of(te, t))
            cur = {mn: nets[mn][:, j] for mn in nets}
            for mn, on_A in [("сплайн по R", False), ("сплайн по A", True)]:
                cur[mn] = spline_predict(tr, te, t, on_A)
            for mn, p in cur.items():
                rel = np.abs(p - true) / np.maximum(np.abs(true), 1e-3 * scales[j])
                rows.append(pd.DataFrame(dict(модель=mn, phi=phi, rel=rel)))
    d = pd.concat(rows, ignore_index=True)

    x = np.arange(len(PHI_LBL))
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.5))
    for mn in [m for m in MODEL if m in set(d["модель"])]:
        s = d[d["модель"] == mn]
        med, p95, bad = [], [], []
        for lo, hi in zip(PHI_EDGES[:-1], PHI_EDGES[1:]):
            v = s[(s.phi >= lo) & (s.phi < hi)]["rel"]
            if len(v):
                med.append(np.median(v)); p95.append(np.percentile(v, 95))
                bad.append((v > 0.1).mean())
            else:
                med.append(np.nan); p95.append(np.nan); bad.append(np.nan)
        kw = dict(color=MODEL[mn]["c"], marker=MODEL[mn]["m"], label=mn)
        ax[0].plot(x, med, **kw); ax[1].plot(x, p95, **kw)
        ax[2].plot(x, np.maximum(bad, 5e-4), **kw)
        NUMBERS[f"по сродству: {mn}, p95 при |phi|<0.05"] = f"{p95[0]:.2e}"
        NUMBERS[f"по сродству: {mn}, доля промахов >10% при |phi|<0.05"] = f"{bad[0]:.3f}"
    for a, ttl, yl in [(ax[0], "медиана ошибки", "медианная отн. ошибка"),
                       (ax[1], "95-й процентиль ошибки", "отн. ошибка, p95"),
                       (ax[2], "доля ошибок $>10\\%$", "доля точек")]:
        a.set(yscale="log", xticks=x, xticklabels=PHI_LBL, xlabel="$|\\varphi|$",
              ylabel=yl, title=ttl)
    ax[2].set_ylim(4e-4, 1)
    ax[2].axhline(5e-4, color=GRID, lw=1.0)
    ax[2].text(0.05, 5.6e-4, "нулевые значения", fontsize=7.5, color=MUTED)
    ax[0].legend(loc="best", fontsize=7.5)
    save(fig, "res_error_vs_phi")


def fig_sweep(sw):
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.3))
    piv = sw.pivot_table(index="depth", columns="hidden", values="среднее", aggfunc="mean")
    im = ax[0].imshow(np.log10(piv.to_numpy()), cmap=SEQ.reversed(), aspect="auto")
    ax[0].set(xticks=range(len(piv.columns)), xticklabels=piv.columns,
              yticks=range(len(piv.index)), yticklabels=piv.index,
              xlabel="ширина слоя", ylabel="число слоёв", title="$\\lg$ средней ошибки")
    mean = piv.values.mean()
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax[0].text(j, i, f"{piv.iloc[i, j]:.1e}", ha="center", va="center",
                       fontsize=7.5, color="white" if piv.iloc[i, j] < mean else INK)
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    g = sw.groupby(["hidden", "depth", "epochs"])["среднее"].mean().sort_values()
    ax[1].plot(range(len(g)), g.to_numpy(), marker="o", ms=4, color=BLUE)
    ax[1].set(yscale="log", xlabel="конфигурация (отсортированы)",
              ylabel="средняя ошибка",
              title="конфигурации по возрастанию ошибки")

    ax[2].scatter(sw["среднее"] * 1e3, np.maximum(sw["равновесие"], 1e-17),
                  s=16, color=BLUE, alpha=0.6, edgecolor="none")
    ax[2].set(yscale="log", ylim=(1e-18, 1e-1),
              xlabel="средняя ошибка, $\\times 10^{-3}$", ylabel="невязка равновесия",
              title=f"ограничение против точности, {len(sw)} прогонов")
    ax[2].axhline(1e-15, color=ORANGE, ls="--", lw=1.2)
    save(fig, "res_sweep")
    NUMBERS["свип: прогонов"] = len(sw)
    best = tuple(int(v) for v in g.index[0])
    NUMBERS["свип: лучшая конфигурация (ширина, слоёв, эпох)"] = f"{best} -> {g.iloc[0]:.2e}"
    NUMBERS["свип: разброс по точности"] = f"{g.iloc[-1]/g.iloc[0]:.1f}x"
    NUMBERS["свип: макс. невязка равновесия"] = f"{sw['равновесие'].max():.1e}"
    NUMBERS["свип: макс. доля неверного знака"] = f"{sw['знак'].max():.1e}"


# =======================================================================
#  ИСТОЧНИКИ ЧИСЕЛ
# =======================================================================

def get_comparison(core, cfg, seeds, recompute):
    """Читает reports/03_comparison.csv, а если его нет — считает."""
    if CMP_PATH.exists() and not recompute:
        print(f"  сравнение: читаю {CMP_PATH.name}")
        return pd.read_csv(CMP_PATH)
    print(f"  сравнение: считаю ({seeds} сидов)")
    frames = []
    for seed in range(seeds):
        cfgs = {"сеть с физикой": Config(model="physics", **cfg),
                "сеть без физики": Config(model="plain", **cfg)}
        frames.append(compare_all(core, SPLITS, cfgs, TARGETS, seed=seed).assign(seed=seed))
    full = pd.concat(frames, ignore_index=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(REPORT_DIR / "03_comparison_all_seeds.csv", index=False)
    num = full.select_dtypes(include=[np.number]).columns.drop("seed")
    agg = full.groupby(["разбиение", "канал", "модель"])[list(num)].median().reset_index()
    agg.to_csv(CMP_PATH, index=False)
    return agg


def get_sweep(core, grid, seeds, recompute):
    """Читает reports/04_sweep.csv, а если его нет — считает."""
    if SWEEP_PATH.exists() and not recompute:
        print(f"  свип: читаю {SWEEP_PATH.name}")
        return pd.read_csv(SWEEP_PATH)
    tr, te = SPLITS["прореживание сетки"](core)
    scales = scales_of(core)
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"  свип: считаю, {len(combos)} конфигураций x {seeds} сидов")
    rows = []
    for i, vals in enumerate(combos, 1):
        prm = dict(zip(keys, vals))
        for seed in range(seeds):
            t0 = time.time()
            net, hist = train_model(tr, Config(model="physics", seed=seed, **prm))
            res = eval_net(net, te, scales)
            rows.append(dict(**prm, seed=seed, эпох=len(hist), сек=time.time() - t0,
                             среднее=float(np.mean([res[t]["rel_median"] for t in TARGETS])),
                             равновесие=max(res[t].get("равновесие", 0.0) for t in TARGETS),
                             знак=max(res[t]["доля_неверный_знак"] for t in TARGETS)))
        print(f"    [{i}/{len(combos)}] {prm} -> {rows[-1]['среднее']:.2e}")
    sw = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sw.to_csv(SWEEP_PATH, index=False)
    return sw


# =======================================================================
#  ТАБЛИЦЫ
# =======================================================================

NAMES = {"R_VT2_12": "VT2", "R_VV23_12": "VV23", "R_VV123_12": "VV123"}
SPN = {"прореживание сетки": "Прореживание сетки", "срезы по T": "Срезы по $T$",
       "срезы по T12": "Срезы по $\\Tv$", "экстраполяция по $T_3$": "Экстраполяция",
       "экстраполяция по T3": "Экстраполяция по $\\Tt$"}


def _cap(s):
    return s[:1].upper() + s[1:]


def _num(v, d=2):
    return "$" + f"{v:.{d}f}".replace(".", "{,}") + "$"


def _sci(v, d=1):
    if not np.isfinite(v):
        return "---"
    if v == 0:
        return "$0$"
    m, e = f"{v:.{d}e}".split("e")
    return f"${m.replace('.', '{,}')}\\cdot10^{{{int(e)}}}$"


def write_tables(cmp, path):
    L = ["% Сгенерировано scripts/make_figures.py.",
         "% Вставить вместо соответствующих tabular в sections/05_results.tex.\n"]
    mods = [m for m in MODEL if m in set(cmp["модель"])]

    L += ["\n% --- медианная ошибка сети с физикой ---",
          "\\begin{tabular}{@{}lccc@{}}\n\\toprule",
          "Схема валидации & VT2 & VV23 & VV123 \\\\\n\\midrule"]
    p = cmp[cmp["модель"] == "сеть с физикой"]
    for s in SPLITS:
        r = [p[(p["разбиение"] == s) & (p["канал"] == t)]["rel_median"] for t in TARGETS]
        if any(len(x) == 0 for x in r):
            continue
        L.append(f"{SPN[s]} & " + " & ".join(_sci(x.iloc[0]) for x in r) + " \\\\")
    L += ["\\bottomrule\n\\end{tabular}\n"]

    L += ["\n% --- худший случай по моделям ---",
          "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule",
          "Модель & медиана & p95 & макс. & мин. $R^2$ & невязка равн. "
          "& неверный знак \\\\\n\\midrule"]
    w = cmp.groupby("модель").agg(med=("rel_median", "max"), p95=("rel_p95", "max"),
                                  mx=("rel_max", "max"), r2=("r2", "min"),
                                  eq=("равновесие", "max"),
                                  sg=("доля_неверный_знак", "max"))
    for m in mods:
        r = w.loc[m]
        L.append(f"{_cap(m)} & {_sci(r['med'])} & {_sci(r['p95'])} & {_num(r['mx'])} & "
                 f"{_num(r['r2'], 4)} & {_sci(r['eq'])} & "
                 f"{'нет' if r['sg'] == 0 else _sci(r['sg'])} \\\\")
    L += ["\\bottomrule\n\\end{tabular}\n"]

    L += ["\n% --- кто первый в каждой из 12 комбинаций ---",
          "\\begin{tabular}{@{}l" + "c" * len(mods) + "@{}}\n\\toprule",
          "Критерий & " + " & ".join(_cap(m) for m in mods) + " \\\\\n\\midrule"]
    for col, nm in [("rel_median", "Медиана ошибки"), ("rel_p95", "95-й процентиль"),
                    ("rel_max", "Максимум ошибки"), ("r2", "$R^2$")]:
        f = (lambda g: g.loc[g[col].idxmax(), "модель"]) if col == "r2" \
            else (lambda g: g.loc[g[col].idxmin(), "модель"])
        vc = cmp.groupby(["разбиение", "канал"]).apply(f, include_groups=False).value_counts()
        L.append(f"{nm} & " + " & ".join(str(int(vc.get(m, 0))) for m in mods) + " \\\\")
    L += ["\\bottomrule\n\\end{tabular}\n"]

    if {"сеть с физикой", "сеть без физики"} <= set(cmp["модель"]):
        L += ["\n% --- вклад физики: без физики / с физикой ---",
              "\\begin{tabular}{@{}lccc@{}}\n\\toprule",
              "Схема валидации & VT2 & VV23 & VV123 \\\\\n\\midrule"]
        a = cmp[cmp["модель"] == "сеть с физикой"].set_index(["разбиение", "канал"])["rel_median"]
        b = cmp[cmp["модель"] == "сеть без физики"].set_index(["разбиение", "канал"])["rel_median"]
        g = b / a
        for s in SPLITS:
            if (s, TARGETS[0]) not in g.index:
                continue
            L.append(f"{SPN[s]} & " + " & ".join(_num(g[(s, t)], 1) for t in TARGETS) + " \\\\")
        L += ["\\bottomrule\n\\end{tabular}\n"]
        NUMBERS["вклад физики: диапазон"] = f"{g.min():.1f}–{g.max():.1f}"

    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  {path.name}")


def write_numbers(path, groups):
    lines = ["Числа, встречающиеся в тексте отчёта прозой.",
             "Сверить с sections/*.tex и заменить где разошлось.",
             f"Этот прогон покрывал: {', '.join(groups)}. "
             "Полный набор даёт запуск без --only.", "=" * 64, ""]
    lines += [f"{k}: {v}" for k, v in NUMBERS.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {path.name}")


# =======================================================================

def main(only, out, quick, seeds, skip_sweep, recompute):
    use_style()
    set_out(Path(out) if out else None)
    core = load_core()
    todo = only or ["data", "phys", "ml", "res"]
    cfg = QUICK if quick else BEST
    ns = 1 if quick else seeds
    t0 = time.time()

    if "data" in todo:
        print("данные:")
        fig_density(core); fig_domain(core); fig_vt2_t3(core)
    if "phys" in todo:
        print("физика:")
        fig_zero_surface(core); fig_affinity_residual(core)
        fig_A_vs_R(core); fig_conditioning(core)

    sweep = None
    if {"ml", "res"} & set(todo) and not skip_sweep:
        print("свип:")
        grid = SWEEP_GRID_QUICK if quick else SWEEP_GRID
        sweep = get_sweep(core, grid, 1 if quick else 2, recompute)

    if "ml" in todo:
        print("обучение:")
        fig_splits(core)
        fig_learning(core, sweep, cfg)
        tr, te = SPLITS["прореживание сетки"](core)
        net, _ = train_model(tr, Config(model="physics", seed=0, **cfg))
        fig_pred_true(net, te)
        fig_profiles(core, net)
        NUMBERS["скорость: мкс на точку"] = f"{timing(net, te)['сеть_мкс_на_точку']:.2f}"
        pt = te.iloc[::max(1, len(te) // 30)].copy()
        nums, anas = [], []
        for ai, axn in enumerate(["T", "T12", "T3"]):
            p1, p0 = pt.copy(), pt.copy()
            p1[axn] += 1.0; p0[axn] -= 1.0
            nums.append((predict(net, p1) - predict(net, p0)) / 2.0)
            anas.append(jacobian(net, pt)[:, :, ai])
        num, ana = np.stack(nums, -1), np.stack(anas, -1)
        e = np.abs(num - ana) / np.abs(num).max(axis=(0, 2), keepdims=True)
        NUMBERS["якобиан: медианное расхождение"] = f"{np.median(e)*100:.3f}%"
        NUMBERS["якобиан: максимальное расхождение"] = f"{e.max()*100:.3f}%"

    if "res" in todo:
        print("результаты:")
        cmp = get_comparison(core, cfg, ns, recompute)
        fig_accuracy(cmp); fig_constraints(cmp); fig_error_vs_phi(core, cfg)
        if sweep is not None:
            fig_sweep(sweep)
        print("таблицы:")
        write_tables(cmp, REPORT_DIR / "TABLES.tex")

    write_numbers(REPORT_DIR / "NUMBERS.txt", todo)
    print(f"\nготово за {time.time() - t0:.0f} c")
    print(f"рисунки: {out_dir()}")
    print(f"скопировать в отчёт:  cp {out_dir()}/*.pdf report/figures/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=["data", "phys", "ml", "res"])
    p.add_argument("--out", default=None, help="куда класть рисунки")
    p.add_argument("--quick", action="store_true", help="быстрая проверка")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--recompute", action="store_true",
                   help="пересчитать сравнение и свип, даже если CSV есть")
    main(**vars(p.parse_args()))