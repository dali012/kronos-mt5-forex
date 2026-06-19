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
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT, equity REAL, cash REAL, unrealized REAL, n_open INTEGER,
                realized REAL, commissions REAL, funding REAL, slippage REAL, total_pnl REAL
            );
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT,
                qty REAL, price REAL, kind TEXT, trade_id TEXT, commission REAL,
                reference_price REAL, slippage REAL, reconciliation INTEGER
            );
            CREATE TABLE IF NOT EXISTS income (
                income_id TEXT PRIMARY KEY, ts TEXT, time_ms INTEGER, income_type TEXT,
                symbol TEXT, asset TEXT, amount REAL, trade_id TEXT, tran_id TEXT, info TEXT
            );
            CREATE TABLE IF NOT EXISTS kv (name TEXT PRIMARY KEY, value TEXT, ts TEXT);
            CREATE INDEX IF NOT EXISTS idx_income_ts_type ON income(ts, income_type);
            CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts);
            """
        )
        # Additive migrations preserve VPS databases created by earlier versions.
        fill_cols = {r["name"] for r in con.execute("PRAGMA table_info(fills)")}
        for name, sql_type in {
            "kind": "TEXT",
            "trade_id": "TEXT",
            "commission": "REAL",
            "reference_price": "REAL",
            "slippage": "REAL",
            "reconciliation": "INTEGER",
        }.items():
            if name not in fill_cols:
                con.execute(f"ALTER TABLE fills ADD COLUMN {name} {sql_type}")
        equity_cols = {r["name"] for r in con.execute("PRAGMA table_info(equity)")}
        for name in ("realized", "commissions", "funding", "slippage", "total_pnl"):
            if name not in equity_cols:
                con.execute(f"ALTER TABLE equity ADD COLUMN {name} REAL")


# --- writer side (bot) -------------------------------------------------------
def write_snapshot(data: dict, db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO snapshot (id, ts, data) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET ts=excluded.ts, data=excluded.data",
            (_now(), json.dumps(data)),
        )


def append_equity(
    equity: float,
    cash: float,
    unrealized: float,
    n_open: int,
    db_path: str = DEFAULT_DB,
    *,
    realized: float | None = None,
    commissions: float | None = None,
    funding: float | None = None,
    slippage: float | None = None,
    total_pnl: float | None = None,
) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO equity "
            "(ts, equity, cash, unrealized, n_open, realized, commissions, funding, slippage, total_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(), equity, cash, unrealized, n_open, realized,
                commissions, funding, slippage, total_pnl,
            ),
        )


def record_fill(
    fill_id: str,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    kind: str = "TREND",
    ts: str | None = None,
    db_path: str = DEFAULT_DB,
    *,
    trade_id: str | None = None,
    commission: float | None = None,
    reference_price: float | None = None,
    slippage: float | None = None,
    reconciliation: bool = False,
) -> bool:
    """Insert a fill (idempotent on fill_id). kind='STOP' for stop-loss fills.
    Returns True if newly inserted."""
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO fills "
            "(fill_id, ts, symbol, side, qty, price, kind, trade_id, commission, "
            "reference_price, slippage, reconciliation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fill_id, ts or _now(), symbol, side, qty, price, kind, trade_id,
                commission, reference_price, slippage, int(reconciliation),
            ),
        )
        return cur.rowcount > 0


def record_income(
    income_id: str,
    time_ms: int,
    income_type: str,
    amount: float,
    *,
    symbol: str = "",
    asset: str = "USDT",
    trade_id: str = "",
    tran_id: str = "",
    info: str = "",
    db_path: str = DEFAULT_DB,
) -> bool:
    ts = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).isoformat()
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO income "
            "(income_id, ts, time_ms, income_type, symbol, asset, amount, trade_id, tran_id, info) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                income_id, ts, time_ms, income_type, symbol, asset, amount,
                trade_id, tran_id, info,
            ),
        )
        return cur.rowcount > 0


def ensure_accounting_baseline(
    equity: float,
    cash: float,
    unrealized: float,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Persist one immutable accounting baseline, preferring existing equity history."""
    with _conn(db_path) as con:
        existing = {
            row["name"]: row["value"]
            for row in con.execute(
                "SELECT name, value FROM kv WHERE name LIKE 'accounting_start_%'"
            )
        }
        if "accounting_start_equity" not in existing:
            row = con.execute(
                "SELECT ts, equity, cash, unrealized FROM equity ORDER BY ts ASC LIMIT 1"
            ).fetchone()
            values = {
                "accounting_start_ts": row["ts"] if row else _now(),
                "accounting_start_equity": str(row["equity"] if row else equity),
                "accounting_start_cash": str(row["cash"] if row else cash),
                "accounting_start_unrealized": str(row["unrealized"] if row else unrealized),
            }
            now = _now()
            con.executemany(
                "INSERT OR IGNORE INTO kv (name, value, ts) VALUES (?, ?, ?)",
                [(name, value, now) for name, value in values.items()],
            )
            existing.update(values)
    return {
        "ts": existing["accounting_start_ts"],
        "equity": float(existing["accounting_start_equity"]),
        "cash": float(existing["accounting_start_cash"]),
        "unrealized": float(existing["accounting_start_unrealized"]),
    }


def performance_summary(
    equity: float,
    cash: float,
    unrealized: float,
    db_path: str = DEFAULT_DB,
) -> dict:
    baseline = ensure_accounting_baseline(equity, cash, unrealized, db_path)
    start_ts = baseline["ts"]
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT income_type, COALESCE(SUM(amount), 0) AS amount "
            "FROM income WHERE ts >= ? GROUP BY income_type",
            (start_ts,),
        ).fetchall()
        by_type = {row["income_type"]: float(row["amount"]) for row in rows}
        slip = con.execute(
            "SELECT COALESCE(SUM(slippage), 0) AS total, COUNT(slippage) AS known "
            "FROM fills WHERE ts >= ?",
            (start_ts,),
        ).fetchone()

    realized = by_type.get("REALIZED_PNL", 0.0)
    commission_income = by_type.get("COMMISSION", 0.0)
    commissions = -commission_income  # positive means a cost
    funding = by_type.get("FUNDING_FEE", 0.0)  # signed: received +, paid -
    attributed = {"REALIZED_PNL", "COMMISSION", "FUNDING_FEE"}
    other_income = sum(value for key, value in by_type.items() if key not in attributed)
    total_pnl = equity - baseline["equity"]
    unrealized_change = unrealized - baseline["unrealized"]
    explained = realized - commissions + funding + other_income + unrealized_change
    return {
        "starting_ts": start_ts,
        "starting_equity": baseline["equity"],
        "starting_cash": baseline["cash"],
        "starting_unrealized": baseline["unrealized"],
        "current_equity": equity,
        "current_cash": cash,
        "total_pnl": total_pnl,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "unrealized_change": unrealized_change,
        "commissions": commissions,
        "funding_payments": funding,
        "other_income": other_income,
        "slippage": float(slip["total"]),
        "slippage_known_fills": int(slip["known"]),
        "reconciliation_residual": total_pnl - explained,
        "income_last_update": get_kv("accounting_income_last_success_ts", None, db_path),
        "income_error": get_kv("accounting_income_error", "", db_path) or None,
    }


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
            "SELECT ts, equity, cash, unrealized, n_open, realized, commissions, "
            "funding, slippage, total_pnl FROM equity ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def read_fills(limit: int = 200, after_rowid: int = 0, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT rowid, ts, symbol, side, qty, price, kind, trade_id, commission, "
            "reference_price, slippage, reconciliation FROM fills "
            "WHERE rowid > ? ORDER BY rowid DESC LIMIT ?",
            (after_rowid, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def read_income(limit: int = 500, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT income_id, ts, time_ms, income_type, symbol, asset, amount, "
            "trade_id, tran_id, info FROM income ORDER BY time_ms DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
