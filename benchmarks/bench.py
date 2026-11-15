"""Wall-clock and peak-memory benchmarks on synthetic data. Run: python benchmarks/bench.py

The model column uses the default protocol: on large tables the candidates are compared on a
seeded sample of ``SELECTION_ROWS`` training rows, the selected model is refitted on all
training rows and scored on the whole hold-out. For 100,000 rows the same data is also
trained with the sample switched off, so the cost and the effect of the sample are measured
side by side on the same machine. Pass ``--quick`` to skip that slow comparison.
"""

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
from auto_data_scientist.modeling import SELECTION_ROWS

COMPARE_AT = 100_000


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


def describe(report) -> str:
    return (f"{report.best_model}, hold-out AUC {report.test_metrics['roc_auc']:.4f} on "
            f"{report.n_test:,} rows, candidates compared on {report.n_selection:,} of "
            f"{report.n_train:,} training rows")


if __name__ == "__main__":
    quick = "--quick" in sys.argv[1:]
    notes = []
    print("| rows | load CSV (s) | profile (s) | group-by query (ms) | model (s) "
          "| model, sample off (s) |")
    print("|---|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        for n in (10_000, 100_000, 1_000_000):
            path = Path(tmp) / f"d{n}.csv"
            make(n).to_csv(path, index=False)
            ds, t_load = timed(lambda p=path: load_dataset(p))
            df = ds.frame()
            _, t_prof = timed(lambda d=df: profile_frame(d))
            _, t_sql = timed(lambda d=ds: run_query(
                d, f"SELECT region, avg(y), count(*) FROM d{n} GROUP BY 1"))  # noqa: B023, S608
            report, t_model = timed(lambda d=df: train_supervised(d, "label"))
            notes.append(f"{n:,} rows: {describe(report)}.")
            t_full = "not run"
            if report.n_selection == report.n_train:
                t_full = "same (no sample needed)"
            if n == COMPARE_AT and not quick:
                full, tf = timed(lambda d=df: train_supervised(d, "label",
                                                               max_selection_rows=None))
                t_full = f"{tf:.1f}"
                notes.append(f"{n:,} rows, sample off: {describe(full)}.")
            print(f"| {n:,} | {t_load:.2f} | {t_prof:.2f} | {t_sql * 1000:.1f} | {t_model:.1f} "
                  f"| {t_full} |")
            ds.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nSelection sample: at most {SELECTION_ROWS:,} training rows.")
    for line in notes:
        print(f"- {line}")
    print(f"\nPeak resident memory of the whole run: {peak:,.0f} MB")
