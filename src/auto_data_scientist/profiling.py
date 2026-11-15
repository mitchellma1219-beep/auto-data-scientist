"""Column and table profiling with a documented data-quality score.

Quality score (0-100) = 100 * sum(weight * component), all components in [0, 1]:

* completeness (0.35): 1 - missing cells / all cells. Missing = null, blank, or a null token
  such as "NA", "N/A", "null", "none", "nan".
* validity (0.25): 1 - invalid cells / non-missing cells. Invalid = a value that does not parse
  as the column's inferred type, plus every cell of a row rejected while loading.
* uniqueness (0.15): 1 - exact duplicate rows / rows.
* consistency (0.10): share of columns without a constant value, padded whitespace or
  categories that differ only by letter case.
* outliers (0.15): 1 - mean share of extreme values across numeric columns, where extreme
  means beyond 3 x IQR from the quartiles (Tukey's "far out" fence).

The score describes the data as loaded. It is a screening aid, not a statistical test.
"""

from __future__ import annotations

import re
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as pt
from pydantic import BaseModel, Field

NULL_TOKENS = frozenset({"", "na", "n/a", "null", "none", "nan", "#n/a", "nil"})
WEIGHTS = {"completeness": 0.35, "validity": 0.25, "uniqueness": 0.15, "consistency": 0.10,
           "outliers": 0.15}
_DATE_RE = r"^\s*(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})"
_ID_RE = re.compile(r"(^|_)(id|uuid|guid|key)$|^id(_|$)", re.IGNORECASE)

Kind = Literal["numeric", "boolean", "datetime", "categorical", "text", "id", "empty"]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: Kind
    n_missing: int
    pct_missing: float
    n_unique: int
    n_invalid: int = 0
    stats: dict[str, float] = Field(default_factory=dict)
    n_outliers: int = 0
    n_extreme: int = 0
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    inconsistent: bool = False


class QualityScore(BaseModel):
    score: float | None
    completeness: float | None = None
    validity: float | None = None
    uniqueness: float | None = None
    consistency: float | None = None
    outliers: float | None = None
    weights: dict[str, float] = Field(default_factory=lambda: dict(WEIGHTS))


class TableProfile(BaseModel):
    table: str
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    rejected_rows: int
    columns: list[ColumnProfile]
    quality: QualityScore
    issues: list[str]

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def by_kind(self, *kinds: Kind) -> list[str]:
        return [c.name for c in self.columns if c.kind in kinds]


def missing_mask(s: pd.Series) -> pd.Series:
    """True where a value is null, blank or a recognised null token."""
    mask = s.isna()
    if not (pt.is_numeric_dtype(s) or pt.is_datetime64_any_dtype(s) or pt.is_bool_dtype(s)):
        text = s.astype("string").str.strip().str.lower()
        mask = mask | text.isin(NULL_TOKENS).fillna(False).astype(bool)
    return mask.astype(bool)


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse a column to timestamps; unparseable values become NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Parse a column to floats; unparseable values become NaN."""
    if pt.is_bool_dtype(s):
        return s.astype(float)
    if pt.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.strip(), errors="coerce").astype(float)


def _numeric_profile(col: ColumnProfile, x: pd.Series) -> None:
    v = x.to_numpy(dtype=float)
    q1, med, q3 = (float(q) for q in np.quantile(v, [0.25, 0.5, 0.75]))
    col.stats = {
        "min": float(v.min()), "q1": q1, "median": med, "q3": q3, "max": float(v.max()),
        "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "skew": float(pd.Series(v).skew()) if v.size > 2 else 0.0,
    }
    iqr = q3 - q1
    if iqr > 0:
        col.n_outliers = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
        col.n_extreme = int(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).sum())
    else:
        # Zero IQR (for example a zero-inflated column): flag rare departures from the median.
        off = int((v != med).sum())
        if 0 < off < 0.05 * v.size:
            col.n_outliers = col.n_extreme = off
    if col.n_extreme:
        col.issues.append(f"{col.n_extreme} extreme value(s) beyond 3 x IQR")


def _profile_column(s: pd.Series, n_rows: int) -> ColumnProfile:
    name = str(s.name)
    missing = missing_mask(s)
    valid = s[~missing]
    n_valid = len(valid)
    col = ColumnProfile(
        name=name, dtype=str(s.dtype), kind="empty", n_missing=int(missing.sum()),
        pct_missing=float(missing.mean()) if n_rows else 0.0, n_unique=int(valid.nunique()))
    if n_valid == 0:
        col.issues.append("column has no usable values")
        return col
    id_name = bool(_ID_RE.search(name))
    if pt.is_bool_dtype(s):
        col.kind = "boolean"
    elif pt.is_datetime64_any_dtype(s):
        col.kind = "datetime"
    elif pt.is_numeric_dtype(s):
        x = valid.astype(float)
        whole = bool(np.all(np.mod(x, 1) == 0))
        unique = col.n_unique == n_valid and n_valid > 1
        consecutive = unique and n_valid >= 20 and x.max() - x.min() + 1 == n_valid
        if whole and (id_name or (unique and consecutive)):
            col.kind = "id"  # a key: unique row identifier or a foreign key such as customer_id
        elif col.n_unique <= 2 and set(x.unique()) <= {0.0, 1.0}:
            col.kind = "boolean"
        else:
            col.kind = "numeric"
            _numeric_profile(col, x)
    else:
        text = valid.astype("string")
        stripped = text.str.strip()
        as_num = pd.to_numeric(stripped, errors="coerce")
        date_like = float(stripped.str.match(_DATE_RE).mean())
        if float(as_num.notna().mean()) >= 0.9:
            col.kind = "numeric"
            col.n_invalid = int(as_num.isna().sum())
            _numeric_profile(col, as_num.dropna().astype(float))
        elif date_like >= 0.9:
            col.kind = "datetime"
            col.n_invalid = int(parse_dates(stripped).isna().sum())
        elif id_name and col.n_unique > max(20, 0.05 * n_valid):
            col.kind = "id"
        elif col.n_unique <= max(20, 0.05 * n_valid):
            col.kind = "categorical"
        else:
            col.kind = "text"
        if col.n_invalid:
            col.issues.append(f"{col.n_invalid} value(s) do not parse as {col.kind}")
        if col.kind == "categorical":
            padded = int((text != stripped).sum())
            folded = stripped.str.lower().nunique()
            if padded:
                col.issues.append(f"{padded} value(s) have leading or trailing whitespace")
                col.inconsistent = True
            if folded < stripped.nunique():
                col.issues.append("categories differ only by letter case")
                col.inconsistent = True
    if col.kind in ("categorical", "boolean"):
        counts = valid.astype("string").str.strip().value_counts().head(5)
        col.top_values = [(str(k), int(v)) for k, v in counts.items()]
    if col.n_unique == 1 and n_valid > 1:
        col.issues.append("constant column")
        col.inconsistent = True
    if col.pct_missing > 0.5:
        col.issues.append(f"{col.pct_missing:.0%} of values are missing")
    return col


def profile_frame(df: pd.DataFrame, table: str = "data", rejected_rows: int = 0) -> TableProfile:
    """Profile every column and compute the quality score described in the module docstring."""
    n_rows, n_cols = df.shape
    columns = [_profile_column(df[c], n_rows) for c in df.columns]
    issues: list[str] = []
    if rejected_rows:
        issues.append(f"{rejected_rows} malformed row(s) were rejected while loading")
    if n_rows == 0 or n_cols == 0:
        issues.append("table has no rows" if n_cols else "table has no columns")
        return TableProfile(table=table, n_rows=n_rows, n_cols=n_cols, n_duplicate_rows=0,
                            rejected_rows=rejected_rows, columns=columns,
                            quality=QualityScore(score=None), issues=issues)
    n_dup = int(df.duplicated().sum())
    if n_dup:
        issues.append(f"{n_dup} exact duplicate row(s)")
    cells = n_rows * n_cols
    missing_cells = sum(c.n_missing for c in columns)
    rejected_cells = rejected_rows * n_cols
    present = cells - missing_cells + rejected_cells
    invalid = sum(c.n_invalid for c in columns) + rejected_cells
    numeric = [c for c in columns if c.kind == "numeric"]
    extreme = [c.n_extreme / max(n_rows - c.n_missing - c.n_invalid, 1) for c in numeric]
    parts = {
        "completeness": 1 - missing_cells / cells,
        "validity": 1 - invalid / present if present else 0.0,
        "uniqueness": 1 - n_dup / n_rows,
        "consistency": 1 - sum(c.inconsistent for c in columns) / n_cols,
        "outliers": 1 - float(np.mean(extreme)) if extreme else 1.0,
    }
    score = 100 * sum(WEIGHTS[k] * v for k, v in parts.items())
    for c in columns:
        issues.extend(f"{c.name}: {msg}" for msg in c.issues)
    return TableProfile(
        table=table, n_rows=n_rows, n_cols=n_cols, n_duplicate_rows=n_dup,
        rejected_rows=rejected_rows, columns=columns, issues=issues,
        quality=QualityScore(score=round(score, 2), **{k: round(v, 4) for k, v in parts.items()}))
