# Analysis report: churn.csv

## Executive summary

- The table `churn` has 3,000 rows and 7 columns; data quality score 100.0/100 (completeness 100.0%, validity 100.0%, 0 duplicate rows).
- By `plan`, `pro` accounts for 33.8% of total `monthly_spend` (61,474 across 1,004 rows).
- By `churned`, `0` accounts for 70.2% of total `monthly_spend` (127,728 across 2,119 rows).
- A logistic regression model predicts `churned` with hold-out roc_auc 0.726 versus 0.500 for a no-skill baseline (600 held-out rows).

## Business problem

Which customers are most at risk of churning?

## Data overview

Source `churn.csv` (sha256 `5587db71f8c5441f…`), table `churn`: 3,000 rows, 7 columns, 0 malformed row(s) rejected on load.

| column | kind | missing | unique | invalid | extreme |
|---|---|---|---|---|---|
| customer_id | id | 0.0% | 3000 | 0 | 0 |
| tenure_months | numeric | 0.0% | 59 | 0 | 0 |
| monthly_spend | numeric | 0.0% | 2637 | 0 | 4 |
| plan | categorical | 0.0% | 3 | 0 | 0 |
| support_tickets | numeric | 0.0% | 9 | 0 | 17 |
| noise | numeric | 0.0% | 2866 | 0 | 0 |
| churned | boolean | 0.0% | 2 | 0 | 0 |

Likely metrics: `monthly_spend`, `tenure_months`, `support_tickets`, `noise`. Dimensions: `plan`, `churned`. Dates: none.

## Key findings

1. The table `churn` has 3,000 rows and 7 columns; data quality score 100.0/100 (completeness 100.0%, validity 100.0%, 0 duplicate rows).
2. By `plan`, `pro` accounts for 33.8% of total `monthly_spend` (61,474 across 1,004 rows).
3. By `churned`, `0` accounts for 70.2% of total `monthly_spend` (127,728 across 2,119 rows).
4. A logistic regression model predicts `churned` with hold-out roc_auc 0.726 versus 0.500 for a no-skill baseline (600 held-out rows).

## Supporting evidence

### Data quality

Score 100.0/100 = 35% completeness (1.0000) + 25% validity (1.0000) + 15% uniqueness (1.0000) + 10% consistency (1.0000) + 15% outliers (0.9982).

- monthly_spend: 4 extreme value(s) beyond 3 x IQR
- support_tickets: 17 extreme value(s) beyond 3 x IQR

### Numeric columns

| column | count | mean | std | min | 25% | 50% | 75% | max | skew |
|---|---|---|---|---|---|---|---|---|---|
| tenure_months | 3,000 | 30.37 | 17.21 | 1 | 16 | 31 | 46 | 59 | -0.02904 |
| monthly_spend | 3,000 | 60.62 | 41.29 | 0.17 | 28.9 | 51.77 | 83.54 | 309.4 | 1.167 |
| support_tickets | 3,000 | 1.526 | 1.243 | 0 | 1 | 1 | 2 | 8 | 0.8576 |
| noise | 3,000 | 0.02837 | 1.003 | -3.501 | -0.6457 | 0.0242 | 0.7236 | 3.527 | 0.009381 |

### `monthly_spend` by `plan`

| plan | rows | total | mean | median | share_of_total |
|---|---|---|---|---|---|
| pro | 1004 | 61,474 | 61.23 | 52.27 | 0.3381 |
| basic | 984 | 60,328 | 61.31 | 53.5 | 0.3318 |
| team | 1012 | 60,043 | 59.33 | 50.7 | 0.3302 |

### `monthly_spend` by `churned`

| churned | rows | total | mean | median | share_of_total |
|---|---|---|---|---|---|
| 0 | 2119 | 127,728 | 60.28 | 51.36 | 0.7024 |
| 1 | 881 | 54,117 | 61.43 | 53.11 | 0.2976 |

### Correlations (strongest first)

| a | b | r | n | p_value |
|---|---|---|---|---|
| monthly_spend | noise | -0.037 | 3000 | 0.0427 |
| support_tickets | noise | 0.01655 | 3000 | 0.3648 |
| tenure_months | support_tickets | 0.01295 | 3000 | 0.4784 |
| monthly_spend | support_tickets | 0.009581 | 3000 | 0.5999 |
| tenure_months | monthly_spend | -0.007789 | 3000 | 0.6698 |
| tenure_months | noise | -0.001437 | 3000 | 0.9373 |

### Query

```sql
SELECT plan, avg(churned) AS churn_rate, count(*) AS customers
FROM churn
GROUP BY plan
ORDER BY churn_rate DESC
```

Reads from: churn. Aggregates: AVG(churned), COUNT(\*). Groups by: plan. Sorted by: churn_rate DESC.

| plan | churn_rate | customers |
|---|---|---|
| basic | 0.373 | 984 |
| pro | 0.2629 | 1004 |
| team | 0.247 | 1012 |

### Model for `churned` (classification)

3,000 rows; 2,400 for training with 5-fold cross-validation, 600 held out. Selected: **logistic_regression** (better than the baseline in 5 of 5 folds). Seed 0.

| model | cv roc_auc | cv std |
|---|---|---|
| baseline | 0.5 | 0 |
| logistic_regression | 0.7282 | 0.01414 |
| random_forest | 0.6736 | 0.01485 |
| gradient_boosting | 0.6733 | 0.0255 |

| index | selected model | no-skill baseline |
|---|---|---|
| accuracy | 0.7233 | 0.7067 |
| balanced_accuracy | 0.5766 | 0.5 |
| f1_macro | 0.573 | 0.4141 |
| roc_auc | 0.7258 | 0.5 |
| average_precision | 0.5226 | 0.2933 |

| feature | permutation importance |
|---|---|
| tenure_months | 0.1326 |
| support_tickets | 0.06227 |
| plan | 0.01933 |
| monthly_spend | -8.844e-05 |
| noise | -0.001764 |

Columns left out:

- `customer_id`: identifier: unique per row, no predictive meaning

## Statistical limitations

- The data is observational: differences between groups can reflect who is in each group rather than the effect of any attribute.
- 6 correlation pair(s) were examined without correction for multiple comparisons; p-values are descriptive and no significance is claimed.
- Feature importance shows what the model relies on, not what causes the outcome.
- Performance is measured on a random hold-out from the same data; it may not carry over to a different period or population.

## Recommended actions

1. Fix the 2 data issue(s) listed under Supporting evidence at the source before relying on downstream numbers.
2. Pilot the `churned` model on a small group and measure the outcome against a randomly selected control group before any wider roll-out.
3. Review the largest segments with the business owner to check that the concentration is expected, then decide where a deeper analysis is worth it.
4. Treat every relationship in this report as a hypothesis; confirm the ones that matter with a controlled experiment.

## Potential business impact

- Ranking by the model, the top 10% of held-out rows contain 2.10 times the average rate of the positive `churned` class (measured on 600 rows).
- **No monetary impact is estimated.** The data holds no costs, margins or measured effect of any intervention. To quantify impact, supply the value of the outcome per unit, the cost of acting and an effect size from an experiment.
