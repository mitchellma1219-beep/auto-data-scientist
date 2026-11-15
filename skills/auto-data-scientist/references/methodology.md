# Methodology

## Loading
Files are copied into an in-memory DuckDB database. CSV type inference scans the whole file;
rows with the wrong number of fields are rejected and counted, never silently repaired.
SQLite and DuckDB sources are opened read-only. After loading, external access is disabled
and the configuration is locked.

## Data quality score (0-100)
`100 x (0.35 completeness + 0.25 validity + 0.15 uniqueness + 0.10 consistency + 0.15 outliers)`

| Component | Definition |
|---|---|
| completeness | 1 - missing cells / all cells. Missing = null, blank or a token such as NA, N/A, null, none, nan |
| validity | 1 - invalid cells / non-missing cells. Invalid = does not parse as the column's inferred type; every cell of a rejected row counts |
| uniqueness | 1 - exact duplicate rows / rows |
| consistency | share of columns without a constant value, padded whitespace or categories differing only by case |
| outliers | 1 - mean share of values beyond 3 x IQR from the quartiles, over numeric columns |

The weights are a stated convention, not an estimate. Report the components with the score.

## SQL
Queries pass four layers: an AST allowlist (one SELECT, known tables only, no file, network,
configuration or dynamic-SQL functions, no recursive CTEs, no joins without a condition), a
locked connection, a transaction that is always rolled back, and a time and row budget.

## Modelling
Duplicates and rows without a target are removed. Identifier, free-text and date columns
are not used as features. A leakage screen drops columns that predict the target almost
perfectly on their own (AUC or |Spearman| or purity >= 0.995) or are named after it. 20% of
rows are held out (the latest 20% with `--time-col`). Candidates (a no-skill baseline, a
linear model, a random forest, gradient boosting) are compared by k-fold cross-validation
with all preprocessing fitted inside each fold. A model "improves on the baseline" only if
it wins in every fold and by a margin (0.02 AUC / balanced accuracy, or 2% of RMSE).
Importance is permutation importance on the hold-out.

On large tables the comparison of candidates is the slow step, so it runs on a seeded
sample: when there are more than 20,000 training rows, cross-validation uses 20,000 of them
(`--selection-rows` changes the number; fewer than 5,000 is refused). Classification
samples keep every class in proportion, with at least 50 rows of each class or all of them.
Only the choice of model uses the sample: the selected model is refitted on all training
rows and scored on the whole hold-out. Permutation importance uses at most 10,000 hold-out
rows. Both samples are drawn with `--seed`, recorded in the output (`n_selection`,
`n_importance`) and listed under the report's statistical limitations.

The size follows from the margin rule above. With 20,000 rows a validation fold has 4,000,
where the AUC of a no-skill model varies by about 0.009 between folds, well inside the 0.02
margin. With 500 rows the spread is about 0.058 and noise can pass the rule, which is why
small samples are refused. The same caution applies to small tables: with a few hundred
rows, "improves on the baseline" is weak evidence.

## Forecasting
The series is aggregated by period (periods without rows count as zero). Naive, seasonal
naive and Holt-Winters forecasts are scored by rolling-origin backtests. Holt-Winters is
used only if it beats the best naive method by at least 5% MAE and in most windows. The
reported range is the forecast +/- the 80th percentile of absolute backtest errors.

## Known limits
- One table at a time for profiling, modelling and the rule-based translator.
- The rule-based translator covers aggregate questions only and refuses the rest.
- No hyperparameter search, probability calibration or class-imbalance handling.
- No causal inference. No hierarchical or multi-seasonal forecasting.
