import numpy as np
import pandas as pd
import pytest

from auto_data_scientist.profiling import WEIGHTS, profile_frame


def test_kinds_and_clean_score(orders):
    p = profile_frame(orders, "orders")
    kinds = {c.name: c.kind for c in p.columns}
    assert kinds == {"order_id": "id", "customer_id": "id", "order_date": "datetime",
                     "region": "categorical", "channel": "categorical", "quantity": "numeric",
                     "unit_price": "numeric", "revenue": "numeric"}
    assert p.quality.score == 100.0 and p.issues == [] and p.n_duplicate_rows == 0


def test_numeric_stats_match_numpy(orders):
    c = profile_frame(orders).column("revenue")
    v = orders["revenue"].to_numpy()
    assert c.stats["mean"] == pytest.approx(v.mean())
    assert c.stats["std"] == pytest.approx(v.std(ddof=1))
    assert [c.stats[k] for k in ("q1", "median", "q3")] == pytest.approx(
        list(np.quantile(v, [0.25, 0.5, 0.75])))
    assert c.stats["skew"] == pytest.approx(pd.Series(v).skew())


def test_known_missing_rate_gives_exact_completeness(orders):
    df = orders.copy()
    df.loc[df.index[:600], "region"] = None            # 10% of one column
    df["channel"] = df["channel"].astype(object)
    df.loc[df.index[:1200], "channel"] = "N/A"         # 20% of another, as a null token
    p = profile_frame(df)
    assert p.column("region").n_missing == 600 and p.column("channel").n_missing == 1200
    assert p.quality.completeness == pytest.approx(1 - 1800 / (6000 * 8))


@pytest.mark.parametrize("token", ["", "  ", "NA", "n/a", "NULL", "None", "nan", "#N/A"])
def test_null_tokens_count_as_missing(token):
    p = profile_frame(pd.DataFrame({"a": ["x", token, "y", "z"]}))
    assert p.column("a").n_missing == 1


def test_duplicates_and_uniqueness(orders):
    df = pd.concat([orders, orders.head(300)], ignore_index=True)
    p = profile_frame(df)
    assert p.n_duplicate_rows == 300
    assert p.quality.uniqueness == pytest.approx(1 - 300 / 6300, abs=1e-4)


def test_invalid_values_in_numeric_column(orders):
    df = orders.copy()
    df["revenue"] = df["revenue"].astype(object)
    df.loc[df.index[:60], "revenue"] = "abc"
    p = profile_frame(df)
    c = p.column("revenue")
    assert c.kind == "numeric" and c.n_invalid == 60
    assert p.quality.validity == pytest.approx(1 - 60 / 48000, abs=1e-4)
    assert any("do not parse as numeric" in i for i in p.issues)


def test_extreme_outliers_are_counted(orders):
    df = orders.copy()
    df.loc[df.index[:30], "revenue"] = 1_000_000.0
    p = profile_frame(df)
    assert p.column("revenue").n_extreme >= 30
    assert p.quality.outliers < 1 and p.quality.score < 100


def test_zero_inflated_column_flags_rare_values():
    c = profile_frame(pd.DataFrame({"refund": [0.0] * 990 + [500.0] * 10})).column("refund")
    assert c.n_extreme == 10


def test_consistency_issues():
    df = pd.DataFrame({"region": ["West", "west", " East", "East"] * 25,
                       "const": ["x"] * 100, "v": range(100)})
    p = profile_frame(df)
    assert p.column("region").inconsistent and p.column("const").inconsistent
    assert p.quality.consistency == pytest.approx(1 - 2 / 3, abs=1e-4)
    text = " ".join(p.issues)
    assert "letter case" in text and "whitespace" in text and "constant column" in text


def test_score_is_the_documented_weighted_sum(orders):
    df = pd.concat([orders, orders.head(500)], ignore_index=True)
    df.loc[df.index[:900], "region"] = None
    q = profile_frame(df).quality
    parts = {k: getattr(q, k) for k in WEIGHTS}
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert q.score == pytest.approx(100 * sum(WEIGHTS[k] * v for k, v in parts.items()), abs=0.02)


def test_rejected_rows_lower_validity(orders):
    assert profile_frame(orders, rejected_rows=600).quality.validity == pytest.approx(
        1 - 600 / 6600, abs=1e-4)


def test_empty_and_all_missing():
    p = profile_frame(pd.DataFrame({"a": [], "b": []}))
    assert p.quality.score is None and "table has no rows" in p.issues
    q = profile_frame(pd.DataFrame({"a": [None, None], "b": [1, 2]}))
    assert q.column("a").kind == "empty" and q.quality.completeness == 0.5


def test_date_strings_boolean_and_text():
    df = pd.DataFrame({
        "d": ["2024-01-05", "2024-02-30", "2024-03-01"] * 20,     # one impossible date in 3
        "flag": [0, 1, 1] * 20, "note": [f"free text {i}" for i in range(60)],
        "session_id": [f"s{i}" for i in range(60)]})
    p = profile_frame(df)
    kinds = {c.name: c.kind for c in p.columns}
    assert kinds == {"d": "datetime", "flag": "boolean", "note": "text", "session_id": "id"}
    assert p.column("d").n_invalid == 20
