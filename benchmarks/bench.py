"""Wall-clock and peak-memory benchmarks on synthetic data. Run: python benchmarks/bench.py"""

from __future__ import annotations

import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_data_scientist import load_dataset, profile_frame, run_query, train_supervised


def make(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "id": np.arange(n), "date": pd.Timestamp("2022-01-01") + pd.to_timedelta(
            rng.integers(0, 730, n), unit="D"),
        "region": rng.choice(["W", "E", "N", "S"], n), "x": x, "y": rng.gamma(2, 20, n),
        "z": rng.integers(0, 100, n), "label": (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)})


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


if __name__ == "__main__":
    print("| rows | load CSV (s) | profile (s) | group-by query (ms) | model (s) |")
    print("|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for n in (10_000, 100_000, 1_000_000):
            path = Path(tmp) / f"d{n}.csv"
            make(n).to_csv(path, index=False)
            ds, t_load = timed(lambda p=path: load_dataset(p))
            df = ds.frame()
            _, t_prof = timed(lambda d=df: profile_frame(d))
            _, t_sql = timed(lambda d=ds: run_query(
                d, f"SELECT region, avg(y), count(*) FROM d{n} GROUP BY 1"))  # noqa: B023, S608
            t_model = "-"
            if n <= 100_000:
                _, tm = timed(lambda d=df: train_supervised(d, "label"))
                t_model = f"{tm:.1f}"
            print(f"| {n:,} | {t_load:.2f} | {t_prof:.2f} | {t_sql * 1000:.1f} | {t_model} |")
            ds.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nPeak resident memory of the whole run: {peak:,.0f} MB")
