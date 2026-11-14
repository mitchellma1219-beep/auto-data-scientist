"""Sub-sampling for model selection: seeded, stated, and never applied to the reported score."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_data_scientist import analyze, modeling, render_report, write_artifacts
from auto_data_scientist.cli import main
from auto_data_scientist.modeling import (
    IMPORTANCE_ROWS,
    MIN_CLASS_ROWS,
    MIN_SELECTION_ROWS,
    SELECTION_ROWS,
    sample_rows,
    train_supervised,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from generate_sample_data import make_churn


@pytest.fixture
def small_samples(monkeypatch):
    """Let the mechanism be exercised on small tables; the real floor has its own test."""
    monkeypatch.setattr(modeling, "MIN_SELECTION_ROWS", 200)


@pytest.fixture(scope="module")
def full(churn):
    return train_supervised(churn, "churned")


def test_small_tables_are_not_sampled(full):
    assert full.n_selection == full.n_train == 2400 and full.n_importance == full.n_test == 600
    assert not any("seeded sample" in x for x in full.limitations)
    assert SELECTION_ROWS == 20_000 and IMPORTANCE_ROWS == 10_000 and MIN_SELECTION_ROWS == 5_000


def test_sample_is_seeded_proportional_and_keeps_row_order():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.choice(["a", "b", "c"], 10_000, p=[0.7, 0.25, 0.05]))
    rows = sample_rows(y, 2_000, stratify=True, seed=4)
    assert list(rows) == sorted(set(rows)) and rows.max() < len(y)       # ordered, no repeats
    assert list(rows) == list(sample_rows(y, 2_000, stratify=True, seed=4))
    assert list(rows) != list(sample_rows(y, 2_000, stratify=True, seed=5))
    share, sampled = y.value_counts(normalize=True), y.iloc[rows].value_counts(normalize=True)
    assert abs(len(rows) - 2_000) <= 2
    assert (share - sampled).abs().max() < 0.001                          # proportional
    plain = sample_rows(y, 2_000, stratify=False, seed=4)
    assert len(plain) == len(set(plain)) == 2_000 and list(plain) == sorted(plain)
    assert list(plain) != list(sample_rows(y, 2_000, stratify=False, seed=5))
    assert list(sample_rows(y, 20_000, stratify=True, seed=4)) == list(range(10_000))


def test_rare_classes_cannot_vanish_from_the_sample():
    y = pd.Series(["common"] * 99_000 + ["rare"] * 970 + ["tiny"] * 30).sample(
        frac=1, random_state=0).reset_index(drop=True)
    picked = y.iloc[sample_rows(y, 1_000, stratify=True, seed=0)].value_counts()
    assert picked["tiny"] == 30                       # fewer than the floor: every row is kept
    assert picked["rare"] == MIN_CLASS_ROWS           # a proportional share would be 10 rows
    assert picked["common"] == 990


def test_selection_uses_the_sample_but_the_score_uses_every_row(churn, full, small_samples):
    r = train_supervised(churn, "churned", max_selection_rows=1600)
    assert abs(r.n_selection - 1600) <= 1 and r.n_train == 2400 and r.n_test == 600
    # The same candidate wins, and because it is refitted on all 2,400 training rows the
    # hold-out metrics and the importance are exactly those of the run without a sample.
    assert r.best_model == full.best_model
    assert r.test_metrics == full.test_metrics
    assert r.baseline_test_metrics == full.baseline_test_metrics
    assert r.importance == full.importance
    # The cross-validation scores do come from the sample, so they differ.
    assert [c.fold_scores for c in r.candidates] != [c.fold_scores for c in full.candidates]
    note = next(x for x in r.limitations if "seeded sample" in x)
    assert f"{r.n_selection:,} of the 2,400 training rows (seed 0" in note
    assert "refitted on all training rows and scored on the whole hold-out" in note
    again = train_supervised(churn, "churned", max_selection_rows=1600)
    assert again.model_dump() == r.model_dump()                           # same seed, same result


def test_default_floor_on_a_realistic_table():
    """8,000 rows, 6,400 for training, candidates compared on the smallest sample allowed."""
    df = make_churn(8000, seed=21)
    r = train_supervised(df, "churned", max_selection_rows=MIN_SELECTION_ROWS)
    whole = train_supervised(df, "churned", max_selection_rows=None)
    assert abs(r.n_selection - 5000) <= 1 and r.n_train == 6400 and r.n_test == 1600
    assert r.improves_on_baseline and r.folds_better_than_baseline == r.n_folds == 5
    assert r.best_model == whole.best_model == "logistic_regression"
    assert r.test_metrics == whole.test_metrics
    assert [f for f, _ in r.importance[:3]] == [f for f, _ in whole.importance[:3]]
    cv = {c.model: c.cv_mean for c in r.candidates}
    cv_whole = {c.model: c.cv_mean for c in whole.candidates}
    assert cv["logistic_regression"] == pytest.approx(cv_whole["logistic_regression"], abs=0.02)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_sample_does_not_invent_signal(seed):
    rng = np.random.default_rng(seed)
    n = 8000
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "c": rng.choice(list("xyz"), n), "target": rng.integers(0, 2, n)})
    r = train_supervised(df, "target", seed=seed, max_selection_rows=MIN_SELECTION_ROWS)
    assert r.n_selection == 5000 and not r.improves_on_baseline
    assert abs(r.test_metrics["roc_auc"] - 0.5) < 0.05


@pytest.mark.parametrize("fold_rows, spread", [(100, 0.058), (1000, 0.018), (4000, 0.009)])
def test_documented_fold_noise_of_a_no_skill_model(fold_rows, spread):
    """The sample sizes are justified by how much a no-skill AUC varies between folds.

    Folds of 100 rows (a 500-row sample) vary by about 0.058, far more than the 0.02 margin
    a candidate must clear, and such a sample did report signal on pure noise during
    development. Folds of 1,000 (the smallest sample allowed) and 4,000 (the default) do not.
    """
    rng = np.random.default_rng(0)
    half = fold_rows // 2
    ranks = rng.random((600, fold_rows)).argsort(axis=1).argsort(axis=1) + 1
    auc = (ranks[:, :half].sum(axis=1) - half * (half + 1) / 2) / (half * half)
    assert auc.mean() == pytest.approx(0.5, abs=0.01)
    assert auc.std(ddof=1) == pytest.approx(spread, rel=0.12)
    formula = ((fold_rows + 1) / (12 * half * half)) ** 0.5
    assert auc.std(ddof=1) == pytest.approx(formula, rel=0.12)


def test_regression_and_time_ordered_tables_can_be_sampled(churn, small_samples):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3000, 2))
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1],
                       "y": 3 * x[:, 0] - 2 * x[:, 1] + rng.normal(0, 1.0, 3000)})
    r = train_supervised(df, "y", max_selection_rows=500)
    assert r.n_selection == 500 and r.best_model == "ridge_regression"
    assert r.test_metrics["rmse"] == pytest.approx(1.0, rel=0.1)
    assert "every class in proportion" not in " ".join(r.limitations)
    timed = churn.assign(signup=pd.date_range("2020-01-01", periods=len(churn), freq="D"))
    t = train_supervised(timed, "churned", time_col="signup", max_selection_rows=1000)
    assert abs(t.n_selection - 1000) <= 1 and t.n_test == 600 and t.improves_on_baseline


def test_importance_sample_is_stated(churn, full):
    r = train_supervised(churn, "churned", max_importance_rows=500)
    assert abs(r.n_importance - 500) <= 1 and r.test_metrics == full.test_metrics
    assert any(f"{r.n_importance:,} of the 600 hold-out rows" in x for x in r.limitations)
    assert {f for f, _ in r.importance[:2]} == {"tenure_months", "support_tickets"}


def test_caps_are_validated(churn):
    with pytest.raises(ValueError, match="at least 5,000"):
        train_supervised(churn, "churned", max_selection_rows=4999)
    with pytest.raises(ValueError, match="at least 500"):
        train_supervised(churn, "churned", max_importance_rows=499)
    unlimited = train_supervised(churn, "churned", max_selection_rows=None,
                                 max_importance_rows=None)
    assert unlimited.n_selection == 2400 and unlimited.n_importance == 600


def test_the_report_and_the_cli_state_the_sample(churn, write, tmp_path, capsys, small_samples):
    path = write(churn, name="churn")
    a = analyze(path, target="churned", selection_rows=800)
    text = render_report(a)
    limits = text.split("## Statistical limitations")[1].split("## Recommended actions")[0]
    assert "seeded sample of 800 of the 2,400 training rows" in limits
    assert "Candidates were compared on a seeded sample of 800 training rows" in text

    paths = write_artifacts(a, tmp_path / "out", path)
    assert "--selection-rows 800" in paths["reproduce"].read_text()
    assert json.loads(paths["data"].read_text())["model"]["n_selection"] == 800

    assert main(["model", str(path), "--target", "churned", "--selection-rows", "800"]) == 0
    assert json.loads(capsys.readouterr().out)["n_selection"] == 800
    assert main(["report", str(path), "--target", "churned", "--selection-rows", "50"]) == 1
    assert "must be at least" in capsys.readouterr().err


def test_default_report_does_not_mention_a_sample(churn, write, tmp_path):
    path = write(churn, name="churn")
    a = analyze(path, target="churned")
    assert "seeded sample" not in render_report(a)
    assert "--selection-rows" not in write_artifacts(a, tmp_path, path)["reproduce"].read_text()
