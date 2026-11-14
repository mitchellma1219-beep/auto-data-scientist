"""Built-in synthetic sample data with known properties (no real people or companies).

The generators are seeded, so every call returns the same rows. The tests use the planted
structure as ground truth, and the dashboard offers the samples so it works without a file
and without a network connection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from auto_data_scientist.errors import DataError


def make_orders(n: int = 6000, seed: int = 11) -> pd.DataFrame:
    """Monthly revenue has a linear trend and a 12-month season; West is the largest region."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2021-01-01", periods=48, freq="MS")
    weight = 1 + 0.01 * np.arange(48) + 0.25 * np.sin(2 * np.pi * np.arange(48) / 12)
    month = rng.choice(48, n, p=weight / weight.sum())
    dates = months[month] + pd.to_timedelta(rng.integers(0, 28, n), unit="D")
    region = rng.choice(["West", "East", "North", "South"], n, p=[0.4, 0.25, 0.2, 0.15])
    qty = rng.integers(1, 6, n)
    price = rng.choice([19.0, 49.0, 99.0], n, p=[0.5, 0.35, 0.15])
    return pd.DataFrame({
        "order_id": np.arange(1, n + 1), "customer_id": rng.integers(1, 1200, n),
        "order_date": dates.strftime("%Y-%m-%d"), "region": region,
        "channel": rng.choice(["web", "store", "partner"], n, p=[0.6, 0.3, 0.1]),
        "quantity": qty, "unit_price": price, "revenue": (qty * price).round(2),
    }).sort_values("order_date").reset_index(drop=True)


def make_churn(n: int = 3000, seed: int = 12) -> pd.DataFrame:
    """Churn follows a logistic model in tenure, support tickets and plan; ``noise`` is noise."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, n)
    tickets = rng.poisson(1.5, n)
    plan = rng.choice(["basic", "pro", "team"], n)
    logit = -0.5 - 0.04 * tenure + 0.35 * tickets + np.where(plan == "basic", 0.6, 0.0)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1), "tenure_months": tenure,
        "monthly_spend": rng.gamma(2, 30, n).round(2), "plan": plan,
        "support_tickets": tickets, "noise": rng.normal(size=n).round(4), "churned": churned,
    })


@dataclass(frozen=True)
class Sample:
    """A built-in dataset plus the starting points that make sense for it."""

    key: str
    title: str
    description: str
    make: Callable[[], pd.DataFrame]
    questions: tuple[str, ...]
    sql: str
    target: str | None = None
    date_col: str | None = None
    value_col: str | None = None
    user_col: str | None = None


SAMPLES: dict[str, Sample] = {
    "orders": Sample(
        key="orders", title="Retail orders",
        description="6,000 synthetic orders over four years. Revenue has a trend and a yearly "
                    "season; West is the largest region.",
        make=make_orders,
        questions=("total revenue by region", "top 2 regions by total revenue",
                   "average revenue per month", "how many orders in West?"),
        sql='SELECT channel, count(*) AS orders, avg(revenue) AS avg_revenue\n'
            'FROM orders\nGROUP BY channel\nORDER BY avg_revenue DESC',
        date_col="order_date", value_col="revenue", user_col="customer_id"),
    "churn": Sample(
        key="churn", title="Subscription churn",
        description="3,000 synthetic customers. Churn depends on tenure, support tickets and "
                    "plan; the column named noise is unrelated to it.",
        make=make_churn,
        questions=("how many customers by plan", "average monthly_spend by plan",
                   "which plan has the highest average support_tickets?"),
        sql='SELECT plan, avg(churned) AS churn_rate, count(*) AS customers\n'
            'FROM churn\nGROUP BY plan\nORDER BY churn_rate DESC',
        target="churned"),
}


def write_sample(key: str, directory: str | Path) -> Path:
    """Write a built-in sample to ``directory`` as ``<key>.csv`` and return the path."""
    if key not in SAMPLES:
        raise DataError(f"unknown sample {key!r}; available: {sorted(SAMPLES)}")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.csv"
    SAMPLES[key].make().to_csv(path, index=False)
    return path
