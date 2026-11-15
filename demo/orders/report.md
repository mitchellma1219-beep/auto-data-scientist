# Analysis report: orders.csv

## Executive summary

- The table `orders` has 6,000 rows and 8 columns; data quality score 100.0/100 (completeness 100.0%, validity 100.0%, 0 duplicate rows).
- By `region`, `West` accounts for 40.9% of total `revenue` (303,582 across 2,439 rows).
- By `channel`, `web` accounts for 60.7% of total `revenue` (450,614 across 3,622 rows).
- `unit_price` and `revenue` move together (r = +0.76, n = 6,000). This is an association, not evidence of cause.

## Business problem

Where does revenue come from, and what range should we plan for over the next six months?

## Data overview

Source `orders.csv` (sha256 `d8c03863d91d3a61…`), table `orders`: 6,000 rows, 8 columns, 0 malformed row(s) rejected on load.

| column | kind | missing | unique | invalid | extreme |
|---|---|---|---|---|---|
| order_id | id | 0.0% | 6000 | 0 | 0 |
| customer_id | id | 0.0% | 1187 | 0 | 0 |
| order_date | datetime | 0.0% | 1319 | 0 | 0 |
| region | categorical | 0.0% | 4 | 0 | 0 |
| channel | categorical | 0.0% | 3 | 0 | 0 |
| quantity | numeric | 0.0% | 5 | 0 | 0 |
| unit_price | numeric | 0.0% | 3 | 0 | 0 |
| revenue | numeric | 0.0% | 15 | 0 | 0 |

Likely metrics: `revenue`, `quantity`, `unit_price`. Dimensions: `region`, `channel`. Dates: `order_date`.

## Key findings

1. The table `orders` has 6,000 rows and 8 columns; data quality score 100.0/100 (completeness 100.0%, validity 100.0%, 0 duplicate rows).
2. By `region`, `West` accounts for 40.9% of total `revenue` (303,582 across 2,439 rows).
3. By `channel`, `web` accounts for 60.7% of total `revenue` (450,614 across 3,622 rows).
4. `unit_price` and `revenue` move together (r = +0.76, n = 6,000). This is an association, not evidence of cause.
5. `revenue` rose by about 87.91 per M period on a straight-line fit over 48 periods (r = +0.45). Seasonal autocorrelation at lag 12: +0.43.
6. Forecast of `revenue` for the next 6 periods totals 106,466 using seasonal naive (backtest mean absolute error 2,241 per period).

## Supporting evidence

### Data quality

Score 100.0/100 = 35% completeness (1.0000) + 25% validity (1.0000) + 15% uniqueness (1.0000) + 10% consistency (1.0000) + 15% outliers (1.0000).

No issues detected.

### Numeric columns

| column | count | mean | std | min | 25% | 50% | 75% | max | skew |
|---|---|---|---|---|---|---|---|---|---|
| quantity | 6,000 | 2.971 | 1.418 | 1 | 2 | 3 | 4 | 5 | 0.02384 |
| unit_price | 6,000 | 41.5 | 27.9 | 19 | 19 | 19 | 49 | 99 | 1.077 |
| revenue | 6,000 | 123.6 | 109.5 | 19 | 49 | 95 | 196 | 495 | 1.683 |

### `revenue` by `region`

| region | rows | total | mean | median | share_of_total |
|---|---|---|---|---|---|
| West | 2439 | 303,582 | 124.5 | 95 | 0.4093 |
| East | 1447 | 181,305 | 125.3 | 95 | 0.2444 |
| North | 1219 | 148,880 | 122.1 | 95 | 0.2007 |
| South | 895 | 107,997 | 120.7 | 95 | 0.1456 |

### `revenue` by `channel`

| channel | rows | total | mean | median | share_of_total |
|---|---|---|---|---|---|
| web | 3622 | 450,614 | 124.4 | 95 | 0.6075 |
| store | 1761 | 216,302 | 122.8 | 95 | 0.2916 |
| partner | 617 | 74,848 | 121.3 | 76 | 0.1009 |

### Correlations (strongest first)

| a | b | r | n | p_value |
|---|---|---|---|---|
| unit_price | revenue | 0.7578 | 6000 | 0 |
| quantity | revenue | 0.5491 | 6000 | 0 |
| quantity | unit_price | 0.008615 | 6000 | 0.5047 |

### Question: top 2 regions by total revenue

```sql
SELECT "region", SUM("revenue") AS "sum_revenue" FROM "orders" GROUP BY "region" ORDER BY "sum_revenue" DESC LIMIT 2
```

Reads from: orders. Aggregates: SUM("revenue"). Groups by: "region". Sorted by: "sum_revenue" DESC. Returns at most 2 row(s).

| region | sum_revenue |
|---|---|
| West | 303,582 |
| East | 181,305 |

### Forecast of `revenue`

48 historical periods, 5 rolling backtest(s) of 6 steps. Selected: **seasonal_naive**. Range = forecast +/- the 80th percentile of absolute backtest errors.

| index | backtest MAE |
|---|---|
| naive | 2,412 |
| seasonal_naive | 2,241 |
| holt_winters | 1,749 |

| period | forecast | low | high |
|---|---|---|---|
| 2025-01-01 | 14,022 | 10,120 | 17,924 |
| 2025-02-01 | 20,321 | 16,419 | 24,223 |
| 2025-03-01 | 17,492 | 13,590 | 21,394 |
| 2025-04-01 | 19,057 | 15,155 | 22,959 |
| 2025-05-01 | 17,566 | 13,664 | 21,468 |
| 2025-06-01 | 18,008 | 14,106 | 21,910 |

## Statistical limitations

- The data is observational: differences between groups can reflect who is in each group rather than the effect of any attribute.
- 3 correlation pair(s) were examined without correction for multiple comparisons; p-values are descriptive and no significance is claimed.
- The trend line ignores autocorrelation; treat it as a description, not a test.
- The statistical model did not beat the naive baselines reliably; the simplest adequate forecast is reported.
- Forecasts assume the historical pattern continues; promotions, price changes and shocks are not modelled.

## Recommended actions

1. Use the simple baseline forecast for planning and revisit once more history or explanatory drivers are available.
2. Review the largest segments with the business owner to check that the concentration is expected, then decide where a deeper analysis is worth it.
3. Treat every relationship in this report as a hypothesis; confirm the ones that matter with a controlled experiment.

## Potential business impact

- Planning range for the next 6 periods of `revenue`: 83,054 to 129,878 (range = forecast +/- the 80th percentile of absolute backtest errors.)
- **No monetary impact is estimated.** The data holds no costs, margins or measured effect of any intervention. To quantify impact, supply the value of the outcome per unit, the cost of acting and an effect size from an experiment.
