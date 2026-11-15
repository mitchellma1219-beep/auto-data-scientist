SELECT plan, avg(churned) AS churn_rate, count(*) AS customers
FROM churn
GROUP BY plan
ORDER BY churn_rate DESC;
