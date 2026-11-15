#!/bin/sh
# Regenerates this report. Input file sha256: 5587db71f8c5441f9ba8b0e58cd17467e141367e3f0948366152e65d181f322b
ads report examples/churn.csv --out demo/churn --sql 'SELECT plan, avg(churned) AS churn_rate, count(*) AS customers
FROM churn
GROUP BY plan
ORDER BY churn_rate DESC' --target churned --horizon 6 --seed 0 --problem 'Which customers are most at risk of churning?'
