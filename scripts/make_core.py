"""Строит data/core.csv — сжатый датасет из 729 физически различных точек.

Плотность факторизуется точно (R ~ n^2), поэтому сырые 3.6 млн строк
сводятся к 729 без потери информации. 
"""
from __future__ import annotations

import argparse
from pathlib import Path

from co2relax.data import CORE_PATH, RAW_PATH, load, to_core


def main(raw: Path) -> None:
    if not raw.exists():
        raise SystemExit(f"Нет сырых данных: {raw}")

    df = load(raw)
    core = to_core(df)

    CORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(CORE_PATH, index=False)

    kb = CORE_PATH.stat().st_size / 1024
    print(f"{len(df):,} строк -> {len(core)} точек")
    print(f"{CORE_PATH} ({kb:.0f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=RAW_PATH)
    main(**vars(p.parse_args()))