import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "CO2_CO2_relaxation_results_large.csv"
OUT = ROOT / "data" / "sample.csv"


def main(n: int, seed: int) -> None:
    if not RAW.exists():
        raise SystemExit(f"Нет данных: {RAW}")

    df = pd.read_csv(RAW)
    sample = df.sample(min(n, len(df)), random_state=seed).sort_index()
    sample.to_csv(OUT, index=False)
    print(f"{len(sample)} строк -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(**vars(p.parse_args()))