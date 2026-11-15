SELECT "region", SUM("revenue") AS "sum_revenue" FROM "orders" GROUP BY "region" ORDER BY "sum_revenue" DESC LIMIT 2;
