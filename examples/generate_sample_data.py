"""Generate synthetic retail and churn data with known properties (no real people or companies)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    here = Path(__file__).parent
    make_orders().to_csv(here / "orders.csv", index=False)
    make_churn().to_csv(here / "churn.csv", index=False)
    print(f"wrote {here / 'orders.csv'} and {here / 'churn.csv'}")
