"""Подбор гиперпараметров нейросетевой модели.

Перебирает конфигурации на одном разбиении, ранжирует по среднему по каналам,
затем проверяет лучшую на всех разбиениях. Результат — reports/04_sweep.csv.

Запуск:  python scripts/sweep.py
         python scripts/sweep.py --split "экстраполяция по T3" --seeds 3
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from co2relax.data import REPORT_DIR, TARGETS, load_core
from co2relax.evaluate import eval_net
from co2relax.splits import SPLITS
from co2relax.train import Config, train_model

GRID = dict(
    hidden=[16, 32, 64, 128],
    depth=[2, 3, 4],
    epochs=[1500, 6000],
    lr=[3e-3],
)


def run_sweep(core: pd.DataFrame, split_name: str, seeds: int) -> pd.DataFrame:
    tr, te = SPLITS[split_name](core)
    scales = np.array([np.median(np.abs(core[t][core[t] != 0])) for t in TARGETS])

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"конфигураций: {len(combos)} × {seeds} сидов = {len(combos)*seeds} обучений")

    rows = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        for seed in range(seeds):
            cfg = Config(model="physics", seed=seed, **params)
            t0 = time.time()
            net, hist = train_model(tr, cfg)
            res = eval_net(net, te, scales)
            rows.append(dict(
                **params, seed=seed, эпох=len(hist), сек=time.time() - t0,
                **{t[2:-3]: res[t]["rel_median"] for t in TARGETS},
                равновесие=max(res[t].get("равновесие", 0.0) for t in TARGETS),
                знак=max(res[t]["доля_неверный_знак"] for t in TARGETS),
            ))
        print(f"  [{i}/{len(combos)}] {params}  ->  "
              f"{np.mean([rows[-1][t[2:-3]] for t in TARGETS]):.2e}")

    df = pd.DataFrame(rows)
    df["среднее"] = df[[t[2:-3] for t in TARGETS]].mean(axis=1)
    return df


def main(split: str, seeds: int) -> None:
    core = load_core()
    df = run_sweep(core, split, seeds)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "04_sweep.csv", index=False)

    agg = (df.groupby(["hidden", "depth", "epochs"])
             .agg(среднее=("среднее", "mean"), разброс=("среднее", "std"),
                  сек=("сек", "mean"))
             .sort_values("среднее"))
    pd.set_option("display.width", 200)
    print("\n=== лучшие 10 конфигураций ===")
    print(agg.head(10).to_string(float_format=lambda v: f"{v:.3e}"))
    print("\n=== худшие 3 ===")
    print(agg.tail(3).to_string(float_format=lambda v: f"{v:.3e}"))

    best = agg.index[0]
    print(f"\nлучшая: hidden={best[0]}, depth={best[1]}, epochs={best[2]}")
    print(f"улучшение относительно худшей: "
          f"{agg['среднее'].iloc[-1] / agg['среднее'].iloc[0]:.1f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="прореживание сетки", choices=list(SPLITS))
    p.add_argument("--seeds", type=int, default=2)
    main(**vars(p.parse_args()))
