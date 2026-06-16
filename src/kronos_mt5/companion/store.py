"""SQLite store shared by the bot (writer) and the companion API (reader).

WAL mode + short-lived connections so the two processes can read/write
concurrently without locking each other. Everything is tiny and append-only.

Control model (single source of truth = the DB, so it survives bot restarts):
  - kv 'mode'  in {run, halt, kill}   run=normal, halt=freeze (keep positions,
    no new entries), kill=flatten + halt.
  - kv 'flatten_seq' / 'flatten_target' — one-shot flatten requests (target =
    'all' or a symbol like 'BTCUSDT-PERP'); strategies act once per new seq.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = "logs/companion.db"
MODES = ("run", "halt", "kill")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=10.0)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
        con.row_factory = sqlite3.Row
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY CHECK (id=1), ts TEXT, data TEXT);
            CREATE TABLE IF NOT EXISTS equity (ts TEXT, equity REAL, cash REAL, unrealized REAL, n_open INTEGER);
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT,
                qty REAL, price REAL, kind TEXT
            );
            CREATE TABLE IF NOT EXISTS kv (name TEXT PRIMARY KEY, value TEXT, ts TEXT);
            """
        )
        # migrate older DBs that predate the `kind` column
        cols = {r["name"] for r in con.execute("PRAGMA table_info(fills)")}
        if "kind" not in cols:
            con.execute("ALTER TABLE fills ADD COLUMN kind TEXT")


# --- writer side (bot) -------------------------------------------------------
def write_snapshot(data: dict, db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO snapshot (id, ts, data) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET ts=excluded.ts, data=excluded.data",
            (_now(), json.dumps(data)),
        )


def append_equity(equity: float, cash: float, unrealized: float, n_open: int,
                  db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO equity (ts, equity, cash, unrealized, n_open) VALUES (?, ?, ?, ?, ?)",
            (_now(), equity, cash, unrealized, n_open),
        )


def record_fill(fill_id: str, symbol: str, side: str, qty: float, price: float,
                kind: str = "TREND", ts: str | None = None, db_path: str = DEFAULT_DB) -> bool:
    """Insert a fill (idempotent on fill_id). kind='STOP' for stop-loss fills.
    Returns True if newly inserted."""
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO fills (fill_id, ts, symbol, side, qty, price, kind) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fill_id, ts or _now(), symbol, side, qty, price, kind),
        )
        return cur.rowcount > 0


def heartbeat(db_path: str = DEFAULT_DB) -> None:
    set_kv("bot_alive_ts", _now(), db_path)


# --- shared kv + control ----------------------------------------------------
def set_kv(name: str, value: str, db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO kv (name, value, ts) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (name, value, _now()),
        )


def get_kv(name: str, default: str | None = None, db_path: str = DEFAULT_DB) -> str | None:
    with _conn(db_path) as con:
        row = con.execute("SELECT value FROM kv WHERE name = ?", (name,)).fetchone()
        return row["value"] if row else default


def get_mode(db_path: str = DEFAULT_DB) -> str:
    m = get_kv("mode", "run", db_path)
    return m if m in MODES else "run"


def set_mode(mode: str, db_path: str = DEFAULT_DB) -> None:
    set_kv("mode", mode if mode in MODES else "run", db_path)


def is_halted(db_path: str = DEFAULT_DB) -> bool:
    return get_mode(db_path) in ("halt", "kill")


def request_flatten(target: str = "all", db_path: str = DEFAULT_DB) -> int:
    seq = int(get_kv("flatten_seq", "0", db_path) or 0) + 1
    set_kv("flatten_seq", str(seq), db_path)
    set_kv("flatten_target", target, db_path)
    return seq


def get_flatten(db_path: str = DEFAULT_DB) -> tuple[int, str]:
    return int(get_kv("flatten_seq", "0", db_path) or 0), (get_kv("flatten_target", "all", db_path) or "all")


# --- reader side (API) ------------------------------------------------------
def read_snapshot(db_path: str = DEFAULT_DB) -> dict | None:
    with _conn(db_path) as con:
        row = con.execute("SELECT ts, data FROM snapshot WHERE id = 1").fetchone()
        if not row:
            return None
        d = json.loads(row["data"])
        d.setdefault("ts", row["ts"])
        return d


def read_equity(limit: int = 5000, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT ts, equity, cash, unrealized, n_open FROM equity ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def read_fills(limit: int = 200, after_rowid: int = 0, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT rowid, ts, symbol, side, qty, price, kind FROM fills "
            "WHERE rowid > ? ORDER BY rowid DESC LIMIT ?",
            (after_rowid, limit),
        ).fetchall()
    return [dict(r) for r in rows]
