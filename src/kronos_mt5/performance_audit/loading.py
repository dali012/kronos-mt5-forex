"""Schema introspection and row loading.

Every table and column is optional: a database written by an older release of the
companion simply lacks them, and the audit must degrade to "not available"
instead of raising.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone

# Tables the audit knows how to read. Presence is reported, absence is a finding.
KNOWN_TABLES = (
    "equity",
    "fills",
    "income",
    "closed_positions",
    "basket_cycles",
    "incidents",
    "ops_events",
    "shadow_targets",
    "kv",
    "snapshot",
)


def table_names(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [r[0] for r in rows]


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
    except sqlite3.Error:
        return []


def describe_schema(con: sqlite3.Connection) -> dict:
    present = table_names(con)
    described = {}
    for table in present:
        cols = columns(con, table)
        try:
            count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        except sqlite3.Error:
            count = None
        described[table] = {"rows": count, "columns": cols}
    return {
        "tables": dict(sorted(described.items())),
        "missing_known_tables": sorted(set(KNOWN_TABLES) - set(present)),
    }


def select_all(con: sqlite3.Connection, table: str, order_by: str | None = None) -> list[dict]:
    """Read a whole table as plain dicts, or [] when it does not exist."""
    if table not in table_names(con):
        return []
    sql = f'SELECT * FROM "{table}"'
    if order_by and order_by in columns(con, table):
        sql += f' ORDER BY "{order_by}"'
    try:
        return [dict(row) for row in con.execute(sql)]
    except sqlite3.Error:
        return []


def parse_ts(value) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def as_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):  # NaN or +/- inf
        return None
    return number
