"""Resource limits, configurable through environment variables."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass


def _env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {raw!r}")
    return value


@dataclass(frozen=True)
class Limits:
    """Hard caps that bound memory, time and output size."""

    max_file_mb: float = 500.0
    max_rows: int = 5_000_000
    sql_timeout_s: float = 30.0
    sql_max_rows: int = 10_000
    sql_max_joins: int = 8
    memory_limit_mb: int = 2048

    @classmethod
    def from_env(cls) -> Limits:
        d = cls()
        return cls(
            max_file_mb=_env("ADS_MAX_FILE_MB", d.max_file_mb),
            max_rows=int(_env("ADS_MAX_ROWS", d.max_rows)),
            sql_timeout_s=_env("ADS_SQL_TIMEOUT_S", d.sql_timeout_s),
            sql_max_rows=int(_env("ADS_SQL_MAX_ROWS", d.sql_max_rows)),
            sql_max_joins=int(_env("ADS_SQL_MAX_JOINS", d.sql_max_joins)),
            memory_limit_mb=int(_env("ADS_MEMORY_LIMIT_MB", d.memory_limit_mb)),
        )


def is_loopback(address: str | None) -> bool:
    """True if a server bound to ``address`` can only be reached from this computer.

    An empty address means "every interface" for most servers, so it counts as exposed.
    """
    if not address:
        return False
    host = address.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
