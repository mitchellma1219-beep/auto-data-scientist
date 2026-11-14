"""Supervised modelling with leakage screening, a baseline and honest evaluation.

Protocol: drop exact duplicates and rows without a target; screen features for leakage;
hold out 20% (the most recent 20% when a time column is given); cross-validate every
candidate *pipeline* on the remaining 80% so imputation, scaling and encoding are fitted
inside each fold; pick the best by the primary metric; score it once on the hold-out next to
a no-skill baseline; report permutation importance measured on the hold-out.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from auto_data_scientist.errors import DataError
from auto_data_scientist.profiling import (
    coerce_numeric,
    missing_mask,
    parse_dates,
    profile_frame,
)

Task = Literal["classification", "regression"]
MIN_ROWS = 50
LEAK_THRESHOLD = 0.995


class LeakageFinding(BaseModel):
    column: str
    reason: str
    excluded: bool


class CandidateScore(BaseModel):
    model: str
    cv_mean: float
    cv_std: float
    fold_scores: list[float]


class ModelReport(BaseModel):
    task: Task
    target: str
    n_rows: int
    n_train: int
    n_test: int
    features: list[str]
    primary_metric: str
    higher_is_better: bool
    candidates: list[CandidateScore]
    best_model: str
    improves_on_baseline: bool
    folds_better_than_baseline: int
    n_folds: int
    test_metrics: dict[str, float]
    baseline_test_metrics: dict[str, float]
    importance: list[tuple[str, float]]
    leakage: list[LeakageFinding]
    dropped: dict[str, str]
    limitations: list[str]
    seed: int
    class_balance: dict[str, float] = Field(default_factory=dict)
    lift_top_decile: float | None = None


def detect_leakage(df: pd.DataFrame, target: str, features: list[str],
                   task: Task) -> list[LeakageFinding]:
    """Flag features that reproduce the target almost exactly or are named after it."""
    y = df[target]
    findings: list[LeakageFinding] = []
    stem = target.lower().rstrip("s")
    stem = stem[:-2] if stem.endswith("ed") and len(stem) > 5 else stem
    for col in features:
        x = df[col]
        absent = missing_mask(x)
        if task == "classification" and 0.0 < float(absent.mean()) < 1.0:
            table = pd.crosstab(absent, y.astype("string"))
            match = float(table.max(axis=1).sum() / table.to_numpy().sum())
            if match >= LEAK_THRESHOLD and float(y.value_counts(normalize=True).iloc[0]) <= 0.95:
                findings.append(LeakageFinding(
                    column=col, excluded=True,
                    reason=f"it is filled in for one outcome only (whether it is missing matches "
                           f"the target for {match:.1%} of rows)"))
                continue
        ok = ~(absent | y.isna())
        if ok.sum() < 20:
            continue
        xv, yv = x[ok], y[ok]
        strength, how = 0.0, ""
        numeric = coerce_numeric(xv)
        if numeric.notna().mean() > 0.9 and numeric.nunique() > 10:
            if task == "regression":
                strength = abs(float(stats.spearmanr(numeric.fillna(numeric.median()),
                                                     yv.astype(float)).statistic))
                how = f"|Spearman correlation| with the target is {strength:.3f}"
            elif yv.nunique() == 2:
                codes = (yv == sorted(yv.unique(), key=str)[-1]).astype(int)
                auc = float(roc_auc_score(codes, numeric.fillna(numeric.median())))
                strength = max(auc, 1 - auc)
                how = f"this single column separates the classes with AUC {strength:.3f}"
        elif xv.nunique() <= 50 and task == "classification":
            table = pd.crosstab(xv.astype("string"), yv.astype("string"))
            strength = float(table.max(axis=1).sum() / table.to_numpy().sum())
            base = float(yv.value_counts(normalize=True).iloc[0])
            how = f"knowing this column fixes the target for {strength:.1%} of rows"
            if base > 0.95:
                strength = 0.0
        if strength >= LEAK_THRESHOLD:
            findings.append(LeakageFinding(
                column=col, excluded=True,
                reason=f"near-perfect predictor ({how}); almost certainly recorded with or "
                       "after the outcome"))
        elif len(stem) >= 4 and stem in col.lower():
            findings.append(LeakageFinding(
                column=col, excluded=True,
                reason=f"its name contains the target name ({target!r}); likely derived from "
                       "the outcome"))
    return findings


def _candidates(task: Task, seed: int) -> dict[str, object]:
    if task == "classification":
        return {
            "baseline": DummyClassifier(strategy="prior"),
            "logistic_regression": LogisticRegression(max_iter=2000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed,
                                                    n_jobs=1, min_samples_leaf=2),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        }
    return {
        "baseline": DummyRegressor(strategy="mean"),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1,
                                               min_samples_leaf=2),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }


def _pipeline(model: object, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                                   sparse_output=False))]), categorical),
    ])
    return Pipeline([("prepare", pre), ("model", model)])


def train_supervised(
    df: pd.DataFrame, target: str, task: Task | None = None, time_col: str | None = None,
    keep: list[str] | None = None, seed: int = 0, n_folds: int = 5,
) -> ModelReport:
    """Train, compare with a baseline and evaluate. See the module docstring for the protocol.

    ``keep`` lists columns that must stay as features even if the leakage screen flags them.
    """
    if target not in df.columns:
        raise DataError(f"unknown target {target!r}; available columns: {list(df.columns)}")
    if time_col is not None and time_col not in df.columns:
        raise DataError(f"unknown time column {time_col!r}")
    keep = keep or []
    limitations: list[str] = []
    dropped: dict[str, str] = {}

    data = df.loc[~missing_mask(df[target])].copy()
    repeat = data.duplicated()
    n_dup = int(repeat.sum())
    if n_dup:
        distinct = data.loc[~repeat]
        others = [c for c in distinct.columns if c != target]
        # A near-unique column means identical rows are repeated records, not chance repeats.
        records = any(distinct[c].nunique() >= max(0.5 * len(distinct), 20) for c in others)
        if records:
            data = distinct
            limitations.append(f"{n_dup} exact duplicate row(s) were removed before splitting so "
                               "copies cannot sit on both sides of a split.")
        else:
            limitations.append(
                f"{n_dup} row(s) repeat another row exactly. They were kept because every column "
                "has few distinct values, so repeats are expected. If they are repeated records, "
                "remove them first: the evaluation would otherwise be optimistic.")
    if len(data) < MIN_ROWS:
        raise DataError(f"only {len(data)} usable row(s) with a target value"
                        + (f" after removing {n_dup} duplicate(s)" if n_dup else "")
                        + f"; at least {MIN_ROWS} are required")

    y_raw = data[target]
    y_num = coerce_numeric(y_raw)
    if task is None:
        task = ("regression" if y_num.notna().all() and y_raw.nunique() > 20
                else "classification")
    if task == "regression":
        if y_num.isna().any():
            raise DataError(f"target {target!r} has non-numeric values; cannot run regression")
        y = y_num.astype(float)
    else:
        y = y_raw.astype("string")
        counts = y.value_counts()
        if len(counts) < 2:
            raise DataError(f"target {target!r} has a single class; nothing to predict")
        if len(counts) > 20:
            raise DataError(f"target {target!r} has {len(counts)} classes; maximum is 20")
        if counts.min() < 2 * 2:
            raise DataError(f"class {counts.idxmin()!r} has only {counts.min()} row(s)")
        n_folds = int(max(2, min(n_folds, counts.min() // 2)))

    profile = profile_frame(data.drop(columns=[target]))
    numeric, categorical = [], []
    for c in profile.columns:
        if c.name == time_col:
            dropped[c.name] = "time column (used for ordering only)"
        elif c.kind == "numeric":
            numeric.append(c.name)
        elif c.kind in ("categorical", "boolean"):
            categorical.append(c.name)
        elif c.kind == "id":
            dropped[c.name] = "identifier: unique per row, no predictive meaning"
        else:
            dropped[c.name] = f"{c.kind} column: not usable as a feature in this version"
    features = numeric + categorical
    screened = [c.name for c in profile.columns if c.name != time_col]
    leakage = detect_leakage(data.assign(**{target: y}), target, screened, task)
    for f in leakage:
        if f.column in keep and f.column in features:
            f.excluded = False
            limitations.append(f"{f.column} was kept on request despite a leakage warning.")
        else:
            dropped[f.column] = f"possible leakage: {f.reason}"
    features = [c for c in features if c not in dropped]
    numeric = [c for c in numeric if c in features]
    categorical = [c for c in categorical if c in features]
    if not features:
        raise DataError("no usable feature columns remain after screening")

    X = data[features].copy()
    for name in numeric:
        X[name] = coerce_numeric(X[name])
    for name in categorical:
        text = X[name].astype("string").str.strip()
        X[name] = text.where(~missing_mask(X[name]), None).astype(object)

    if time_col is not None:
        order = np.argsort(parse_dates(data[time_col]).to_numpy(), kind="stable")
        X, y = X.iloc[order], y.iloc[order]
        cut = int(len(X) * 0.8)
        X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
        cv: object = TimeSeriesSplit(n_splits=n_folds)
        limitations.append("Rows were split by time: the model is tested on the most recent 20%.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed,
            stratify=y if task == "classification" else None)
        cv = (StratifiedKFold(n_folds, shuffle=True, random_state=seed)
              if task == "classification" else KFold(n_folds, shuffle=True, random_state=seed))
    if task == "classification" and (y_tr.nunique() < 2 or y_te.nunique() < 2):
        raise DataError("a split contains a single class; provide more data or omit time_col")

    binary = task == "classification" and y.nunique() == 2
    if task == "regression":
        metric, scoring, higher = "rmse", "neg_root_mean_squared_error", False
    elif binary:
        metric, scoring, higher = "roc_auc", "roc_auc", True
    else:
        metric, scoring, higher = "balanced_accuracy", "balanced_accuracy", True

    scores: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in _candidates(task, seed).items():
            raw = cross_val_score(_pipeline(model, numeric, categorical), X_tr, y_tr, cv=cv,
                                  scoring=scoring)
            scores[name] = raw if higher else -raw
    sign = 1 if higher else -1
    contenders = {k: v for k, v in scores.items() if k != "baseline"}
    best = max(contenders, key=lambda k: sign * float(contenders[k].mean()))
    gain = sign * (scores[best] - scores["baseline"])
    folds_better = int((gain > 0).sum())
    scale = abs(float(scores["baseline"].mean())) or 1.0
    min_gain = 0.02 if higher else 0.02 * scale
    improves = folds_better == len(gain) and float(gain.mean()) > min_gain
    if not improves:
        limitations.append(
            "No candidate beat the no-skill baseline consistently across folds: the features "
            "carry little or no usable signal for this target.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = _pipeline(_candidates(task, seed)[best], numeric, categorical).fit(X_tr, y_tr)
        base = _pipeline(_candidates(task, seed)["baseline"], numeric, categorical).fit(X_tr, y_tr)
        imp = permutation_importance(fitted, X_te, y_te, scoring=scoring, n_repeats=5,
                                     random_state=seed)
    importance = sorted(((f, float(v)) for f, v in zip(features, imp.importances_mean,
                                                       strict=True)), key=lambda t: -t[1])

    lift = None
    balance: dict[str, float] = {}
    if task == "regression":
        def _reg(m: Pipeline) -> dict[str, float]:
            pred = m.predict(X_te)
            return {"rmse": float(root_mean_squared_error(y_te, pred)),
                    "mae": float(mean_absolute_error(y_te, pred)),
                    "r2": float(r2_score(y_te, pred))}
        test, base_test = _reg(fitted), _reg(base)
    else:
        balance = {str(k): float(v) for k, v in y.value_counts(normalize=True).items()}
        if min(balance.values()) < 0.1:
            limitations.append("Classes are imbalanced; accuracy alone is misleading here.")

        def _clf(m: Pipeline) -> dict[str, float]:
            pred = m.predict(X_te)
            out = {"accuracy": float(accuracy_score(y_te, pred)),
                   "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
                   "f1_macro": float(f1_score(y_te, pred, average="macro"))}
            if binary:
                pos = list(m.classes_).index(sorted(m.classes_, key=str)[-1])
                proba = m.predict_proba(X_te)[:, pos]
                truth = (y_te == m.classes_[pos]).astype(int)
                out["roc_auc"] = float(roc_auc_score(truth, proba))
                out["average_precision"] = float(average_precision_score(truth, proba))
            return out
        test, base_test = _clf(fitted), _clf(base)
        if binary and len(X_te) >= 50:
            pos = list(fitted.classes_).index(sorted(fitted.classes_, key=str)[-1])
            proba = fitted.predict_proba(X_te)[:, pos]
            truth = (y_te == fitted.classes_[pos]).to_numpy().astype(int)
            top = np.argsort(-proba)[: max(1, len(proba) // 10)]
            if truth.mean() > 0:
                lift = float(truth[top].mean() / truth.mean())

    if len(X_te) < 100:
        limitations.append(f"The hold-out has only {len(X_te)} rows; its metrics are noisy.")
    limitations += [
        "Feature importance shows what the model relies on, not what causes the outcome.",
        "Performance is measured on a random hold-out from the same data; it may not carry "
        "over to a different period or population." if time_col is None else
        "One chronological hold-out was used; performance can vary across periods.",
    ]
    return ModelReport(
        task=task, target=target, n_rows=len(data), n_train=len(X_tr), n_test=len(X_te),
        features=features, primary_metric=metric, higher_is_better=higher,
        candidates=[CandidateScore(model=k, cv_mean=float(v.mean()), cv_std=float(v.std(ddof=1)),
                                   fold_scores=[float(s) for s in v]) for k, v in scores.items()],
        best_model=best, improves_on_baseline=improves, folds_better_than_baseline=folds_better,
        n_folds=len(gain), test_metrics=test, baseline_test_metrics=base_test,
        importance=importance, leakage=leakage, dropped=dropped, limitations=limitations,
        seed=seed, class_balance=balance, lift_top_decile=lift)
