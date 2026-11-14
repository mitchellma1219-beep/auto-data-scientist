"""Load CSV, Excel, Parquet, SQLite and DuckDB sources into a locked in-memory DuckDB database.

After loading, the connection has external access disabled and its configuration locked, so
queries run later cannot read or write files, attach databases or install extensions.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import duckdb
import pandas as pd

from auto_data_scientist.config import Limits
from auto_data_scientist.errors import DataError

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".csv", ".tsv"}
EXCEL_SUFFIXES = {".xlsx"}
PARQUET_SUFFIXES = {".parquet"}
DB_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".duckdb"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EXCEL_SUFFIXES | PARQUET_SUFFIXES | DB_SUFFIXES


def quote_ident(name: str) -> str:
    """Quote an identifier for DuckDB/SQLite."""
    return '"' + name.replace('"', '""') + '"'


def _table_name(raw: str, taken: set[str]) -> str:
    name = re.sub(r"\W+", "_", raw).strip("_").lower() or "data"
    if name[0].isdigit():
        name = f"t_{name}"
    base, i = name, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name


@dataclass
class TableInfo:
    name: str
    n_rows: int
    columns: list[tuple[str, str]]
    rejected_rows: int = 0
    reject_reasons: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    """A loaded source: a locked DuckDB connection plus table metadata."""

    con: duckdb.DuckDBPyConnection
    source: Path
    limits: Limits
    tables: dict[str, TableInfo] = field(default_factory=dict)

    def default_table(self) -> str:
        if not self.tables:
            raise DataError("the source contains no tables")
        return max(self.tables.values(), key=lambda t: t.n_rows).name

    def resolve(self, table: str | None) -> str:
        if table is None:
            return self.default_table()
        if table not in self.tables:
            raise DataError(f"unknown table {table!r}; available: {sorted(self.tables)}")
        return table

    def frame(self, table: str | None = None) -> pd.DataFrame:
        """Return a table as a pandas DataFrame."""
        name = self.resolve(table)
        return self.con.execute(f"SELECT * FROM {quote_ident(name)}").df()  # noqa: S608  # nosec B608

    def close(self) -> None:
        self.con.close()


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    row = con.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()  # noqa: S608  # nosec B608
    return int(row[0]) if row else 0


def _sniff_db(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(16)
    if head.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if head[8:12] == b"DUCK":
        return "duckdb"
    raise DataError(f"{path.name} is neither a SQLite nor a DuckDB database file")


def _register(ds: Dataset, raw_name: str, frame: pd.DataFrame, taken: set[str]) -> None:
    name = _table_name(raw_name, taken)
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise DataError(f"table {raw_name!r} has duplicate column names")
    ds.con.register("_ads_staging", frame)
    ds.con.execute(f"CREATE TABLE {quote_ident(name)} AS SELECT * FROM _ads_staging")  # noqa: S608  # nosec B608
    ds.con.unregister("_ads_staging")


def _load_text(ds: Dataset, path: Path, taken: set[str]) -> None:
    name = _table_name(path.stem, taken)
    try:
        ds.con.execute(
            f"CREATE TABLE {quote_ident(name)} AS "  # noqa: S608  # nosec B608
            "SELECT * FROM read_csv(?, store_rejects=true, sample_size=-1)",
            [str(path)],
        )
    except duckdb.Error as exc:
        raise DataError(f"could not parse {path.name} as delimited text: {exc}") from exc
    rejects = ds.con.execute(
        "SELECT count(DISTINCT line), list(DISTINCT error_type) FROM reject_errors"
    ).fetchone() or (0, [])
    ds.tables[name] = TableInfo(name, 0, [], int(rejects[0]), sorted(map(str, rejects[1] or [])))


def _load_sqlite(ds: Dataset, path: Path, taken: set[str]) -> None:
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        src = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise DataError(f"could not open {path.name} as SQLite: {exc}") from exc
    try:
        names = [r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for raw in names:
            n = src.execute(f"SELECT count(*) FROM {quote_ident(raw)}").fetchone()[0]  # noqa: S608  # nosec B608
            if n > ds.limits.max_rows:
                raise DataError(f"table {raw!r} has {n:,} rows; limit is {ds.limits.max_rows:,}")
            frame = pd.read_sql_query(f"SELECT * FROM {quote_ident(raw)}", src)  # noqa: S608  # nosec B608
            _register(ds, raw, frame, taken)
    except sqlite3.Error as exc:
        raise DataError(f"could not read {path.name}: {exc}") from exc
    finally:
        src.close()


def _load_duckdb(ds: Dataset, path: Path, taken: set[str]) -> None:
    escaped = str(path).replace("'", "''")
    try:
        ds.con.execute(f"ATTACH '{escaped}' AS _ads_src (READ_ONLY)")
        rows = ds.con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_catalog = '_ads_src' AND table_type = 'BASE TABLE'").fetchall()
        for schema, raw in rows:
            name = _table_name(raw, taken)
            ds.con.execute(
                f"CREATE TABLE {quote_ident(name)} AS SELECT * FROM "  # noqa: S608  # nosec B608
                f"_ads_src.{quote_ident(schema)}.{quote_ident(raw)}")
        ds.con.execute("DETACH _ads_src")
    except duckdb.Error as exc:
        raise DataError(f"could not read {path.name} as DuckDB: {exc}") from exc


def load_dataset(path: str | Path, limits: Limits | None = None) -> Dataset:
    """Validate and load a data source. Raises ``DataError`` with a specific reason."""
    limits = limits or Limits.from_env()
    try:
        p = Path(path).expanduser().resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise DataError(f"file not found: {path}") from exc
    if not p.is_file():
        raise DataError(f"not a regular file: {p.name}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DataError(
            f"unsupported file type {suffix or '(none)'}; supported: {sorted(SUPPORTED_SUFFIXES)}")
    size_mb = p.stat().st_size / 1_048_576
    if size_mb == 0:
        raise DataError(f"{p.name} is empty")
    if size_mb > limits.max_file_mb:
        raise DataError(f"{p.name} is {size_mb:,.0f} MB; limit is {limits.max_file_mb:,.0f} MB")

    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{int(limits.memory_limit_mb)}MB'")
    ds = Dataset(con=con, source=p, limits=limits)
    taken: set[str] = set()
    try:
        if suffix in TEXT_SUFFIXES:
            _load_text(ds, p, taken)
        elif suffix in PARQUET_SUFFIXES:
            name = _table_name(p.stem, taken)
            con.execute(f"CREATE TABLE {quote_ident(name)} AS SELECT * FROM read_parquet(?)",  # noqa: S608  # nosec B608
                        [str(p)])
        elif suffix in EXCEL_SUFFIXES:
            try:
                sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")
            except Exception as exc:  # the workbook parser raises many unrelated error types
                raise DataError(f"could not load {p.name} as an Excel workbook: {exc}") from exc
            for sheet, frame in sheets.items():
                _register(ds, str(sheet), frame, taken)
        elif _sniff_db(p) == "sqlite":
            _load_sqlite(ds, p, taken)
        else:
            _load_duckdb(ds, p, taken)
        for name in sorted(taken):
            n = _count(con, name)
            if n > limits.max_rows:
                raise DataError(f"table {name!r} has {n:,} rows; limit is {limits.max_rows:,}")
            cols = [(str(r[0]), str(r[1])) for r in
                    con.execute(f"DESCRIBE {quote_ident(name)}").fetchall()]
            info = ds.tables.get(name) or TableInfo(name, 0, [])
            info.n_rows, info.columns = n, cols
            ds.tables[name] = info
        if not ds.tables:
            raise DataError(f"{p.name} contains no tables")
        # Lock the sandbox: no file or network access and no configuration changes from here on.
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
    except DataError:
        con.close()
        raise
    except (duckdb.Error, ValueError, OSError, KeyError, ImportError) as exc:
        con.close()
        raise DataError(f"could not load {p.name}: {exc}") from exc
    log.info("loaded %s: %s", p.name, {t.name: t.n_rows for t in ds.tables.values()})
    return ds


def save_upload(data: bytes, filename: str, directory: Path, limits: Limits | None = None) -> Path:
    """Store uploaded bytes under ``directory`` with a generated name.

    Only the validated suffix of the client-supplied name is kept, so path separators,
    ``..`` segments, absolute paths and NUL bytes cannot influence where the file is written.
    """
    limits = limits or Limits.from_env()
    if "\x00" in filename:
        raise DataError("invalid file name")
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DataError(
            f"unsupported file type {suffix or '(none)'}; supported: {sorted(SUPPORTED_SUFFIXES)}")
    if len(data) > limits.max_file_mb * 1_048_576:
        raise DataError(f"upload exceeds the {limits.max_file_mb:,.0f} MB limit")
    base = directory.resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = (base / f"{uuid.uuid4().hex}{suffix}").resolve()
    if base not in target.parents:
        raise DataError("invalid upload path")
    target.write_bytes(data)
    return target
