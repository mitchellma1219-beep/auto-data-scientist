import duckdb
import pytest

from auto_data_scientist.config import Limits
from auto_data_scientist.errors import AdsError, QueryTimeoutError, UnsafeQueryError
from auto_data_scientist.ingest import load_dataset
from auto_data_scientist.query import run_query
from auto_data_scientist.sql_guard import validate_sql

ALLOWED = [
    "SELECT * FROM orders",
    "select region, sum(revenue) from orders group by 1 order by 2 desc limit 5",
    "WITH t AS (SELECT region, revenue FROM orders) SELECT region, avg(revenue) FROM t GROUP BY 1",
    "SELECT * FROM orders UNION ALL SELECT * FROM orders",
    "SELECT a.region FROM orders a JOIN orders b ON a.order_id = b.order_id",
    "SELECT region, rank() OVER (PARTITION BY region ORDER BY revenue) FROM orders",
    "SELECT CASE WHEN revenue > 100 THEN 'high' ELSE 'low' END AS tier, count(*) FROM orders "
    "GROUP BY 1",
    "SELECT * FROM (SELECT region FROM orders WHERE quantity > 2) s",
    "SELECT date_trunc('month', CAST(order_date AS TIMESTAMP)) m, sum(revenue) FROM orders "
    "GROUP BY 1",
    "SELECT * FROM range(5)",
    "SELECT 1;",
]
DENIED = [
    "", "   ", "DROP TABLE orders", "DELETE FROM orders", "UPDATE orders SET revenue = 0",
    "INSERT INTO orders SELECT * FROM orders", "CREATE TABLE x AS SELECT * FROM orders",
    "ALTER TABLE orders ADD COLUMN x INT", "TRUNCATE orders", "SELECT * INTO x FROM orders",
    "SELECT 1; DROP TABLE orders", "SELECT * FROM orders; --", "SELECT * FROM orders; SELECT 1",
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM read_csv_auto('x.csv')",
    "SELECT * FROM read_parquet('s3://bucket/x')", "SELECT * FROM read_json('x.json')",
    "SELECT * FROM read_text('/etc/hostname')", "SELECT * FROM read_blob('/etc/hostname')",
    "SELECT * FROM '/etc/passwd'", "SELECT * FROM 'data.csv'", "SELECT * FROM glob('/*')",
    "SELECT * FROM sqlite_scan('x.db', 't')", "SELECT * FROM parquet_scan('x')",
    "SELECT * FROM query('DROP TABLE orders')", "SELECT * FROM query_table('orders')",
    "SELECT getenv('HOME')", "SELECT current_setting('memory_limit')",
    "SELECT * FROM duckdb_settings()", "SELECT * FROM duckdb_secrets()",
    "SELECT * FROM pragma_database_list()", "SELECT * FROM information_schema.tables",
    "SELECT * FROM pg_catalog.pg_tables", "SELECT * FROM main.orders", "SELECT * FROM secret",
    "COPY orders TO '/tmp/x.csv'", "COPY (SELECT 1) TO '/tmp/x.csv'", "EXPORT DATABASE '/tmp/x'",
    "ATTACH '/tmp/x.db' AS x", "DETACH x", "INSTALL httpfs", "LOAD httpfs", "PRAGMA database_list",
    "SET memory_limit = '100GB'", "SET enable_external_access = true", "USE memory", "CHECKPOINT",
    "VACUUM", "CALL pragma_version()", "EXPLAIN ANALYZE SELECT * FROM orders", "DESCRIBE orders",
    "SUMMARIZE orders", "BEGIN", "COMMIT",
    "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) SELECT * FROM t",
    "SELECT * FROM orders a, orders b", "SELECT * FROM orders CROSS JOIN orders b",
    "SELECT * FROM orders a JOIN orders b", "SELECT (SELECT count(*) FROM read_csv('x'))",
    "WITH t AS (SELECT * FROM read_csv('x')) SELECT * FROM t", "SELECT * FROM orders WHERE",
    "/* hi */ DROP TABLE orders", "SELECT * FROM orders UNION SELECT * FROM '/etc/passwd'",
]


@pytest.fixture
def ds(orders, write):
    d = load_dataset(write(orders))
    yield d
    d.close()


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed_queries_validate_and_run(ds, sql):
    assert validate_sql(sql, ds.tables).sql
    run_query(ds, sql)


@pytest.mark.parametrize("sql", DENIED)
def test_denied_queries(ds, sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql, ds.tables)
    with pytest.raises(AdsError):
        run_query(ds, sql)
    assert ds.con.execute("SELECT count(*) FROM orders").fetchone() == (6000,)


def test_too_many_joins_and_long_query(ds):
    joins = " ".join(f"JOIN orders t{i} ON t{i}.order_id = o.order_id" for i in range(9))
    with pytest.raises(UnsafeQueryError, match="joins"):
        validate_sql(f"SELECT 1 FROM orders o {joins}", ds.tables)
    with pytest.raises(UnsafeQueryError, match="longer than"):
        validate_sql("SELECT " + "1," * 20000 + "1", ds.tables)


def test_results_match_pandas_and_are_explained(ds, orders):
    res = run_query(ds, "SELECT region, sum(revenue) AS total FROM orders "
                        "WHERE quantity >= 2 GROUP BY region ORDER BY total DESC")
    expected = (orders[orders.quantity >= 2].groupby("region")["revenue"].sum()
                .sort_values(ascending=False))
    assert [r[0] for r in res.rows] == list(expected.index)
    assert [r[1] for r in res.rows] == pytest.approx(list(expected.values))
    for part in ("Reads from: orders", "SUM(revenue)", "Groups by: region", "quantity >= 2",
                 "Sorted by"):
        assert part in res.explanation
    assert res.columns == ["region", "total"] and not res.truncated


def test_row_cap_and_table_case_insensitive(orders, write):
    d = load_dataset(write(orders), Limits(sql_max_rows=10))
    res = run_query(d, "SELECT * FROM ORDERS")
    assert len(res.rows) == 10 and res.truncated
    d.close()


def test_timeout_interrupts_and_connection_survives(orders, write):
    d = load_dataset(write(orders), Limits(sql_timeout_s=0.5))
    with pytest.raises(QueryTimeoutError):
        run_query(d, "SELECT count(*) FROM range(400000000) a JOIN range(400000000) b "
                     "ON a.range % 7 = b.range % 7")
    assert run_query(d, "SELECT count(*) FROM orders").rows == [(6000,)]
    d.close()


def test_database_layer_blocks_files_even_without_the_guard(ds, tmp_path):
    """Defence in depth: these bypass validate_sql on purpose and must still fail."""
    target = tmp_path / "leak.csv"
    for sql in ("SELECT * FROM read_csv('/etc/passwd')", f"COPY orders TO '{target}'",
                f"ATTACH '{tmp_path / 'x.db'}' AS x", "SET enable_external_access = true",
                "SET memory_limit = '100GB'", "INSTALL httpfs"):
        with pytest.raises(duckdb.Error):
            ds.con.execute(sql)
    assert not target.exists()
