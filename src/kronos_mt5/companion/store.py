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
import math
import sqlite3
from collections import defaultdict
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
            CREATE TABLE IF NOT EXISTS closed_positions (
                position_key TEXT PRIMARY KEY, opened_ts TEXT, closed_ts TEXT,
                symbol TEXT, side TEXT, peak_qty REAL, entry_price REAL, exit_price REAL,
                net_pnl REAL, commissions REAL, initial_risk REAL, r_multiple REAL,
                exit_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS basket_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_ts TEXT NOT NULL,
                ended_ts TEXT, direction TEXT NOT NULL, symbols TEXT NOT NULL,
                starting_equity REAL NOT NULL, ending_equity REAL,
                peak_equity REAL NOT NULL, trough_equity REAL NOT NULL,
                initial_risk REAL, net_pnl REAL, r_multiple REAL,
                exit_reasons TEXT, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                started_ts TEXT NOT NULL, ended_ts TEXT, details TEXT
            );
            CREATE TABLE IF NOT EXISTS ops_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                kind TEXT NOT NULL, details TEXT
            );
            CREATE TABLE IF NOT EXISTS shadow_targets (
                model TEXT NOT NULL, cycle_id TEXT NOT NULL, cycle_ts TEXT NOT NULL,
                recorded_ts TEXT NOT NULL, symbol TEXT NOT NULL, price REAL NOT NULL,
                signal REAL NOT NULL, raw_weight REAL NOT NULL, target_weight REAL NOT NULL,
                portfolio_weight REAL NOT NULL, portfolio_scale REAL NOT NULL,
                funding_rate REAL, funding_scalar REAL NOT NULL, cost_bps REAL NOT NULL,
                PRIMARY KEY (model, cycle_id, symbol)
            );
            CREATE TABLE IF NOT EXISTS kv (name TEXT PRIMARY KEY, value TEXT, ts TEXT);
            CREATE INDEX IF NOT EXISTS idx_income_ts_type ON income(ts, income_type);
            CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts);
            CREATE INDEX IF NOT EXISTS idx_basket_status ON basket_cycles(status);
            CREATE INDEX IF NOT EXISTS idx_incidents_kind ON incidents(kind, started_ts);
            CREATE INDEX IF NOT EXISTS idx_shadow_model_ts ON shadow_targets(model, cycle_ts);
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
        for name, sql_type in {
            "basket_direction": "TEXT",
            "average_correlation": "REAL",
            "effective_bets": "REAL",
            "market_regime": "TEXT",
        }.items():
            if name not in equity_cols:
                con.execute(f"ALTER TABLE equity ADD COLUMN {name} {sql_type}")


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
    basket_direction: str | None = None,
    average_correlation: float | None = None,
    effective_bets: float | None = None,
    market_regime: str | None = None,
) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO equity "
            "(ts, equity, cash, unrealized, n_open, realized, commissions, funding, slippage, "
            "total_pnl, basket_direction, average_correlation, effective_bets, market_regime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(), equity, cash, unrealized, n_open, realized,
                commissions, funding, slippage, total_pnl, basket_direction,
                average_correlation, effective_bets, market_regime,
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


def record_shadow_targets(rows: list[dict], db_path: str = DEFAULT_DB) -> int:
    """Persist shadow decisions idempotently; these rows never place orders."""
    if not rows:
        return 0
    recorded_ts = _now()
    values = [
        (
            str(row["model"]),
            str(row["cycle_id"]),
            str(row["cycle_ts"]),
            recorded_ts,
            str(row["symbol"]),
            float(row["price"]),
            float(row["signal"]),
            float(row["raw_weight"]),
            float(row["target_weight"]),
            float(row["portfolio_weight"]),
            float(row["portfolio_scale"]),
            float(row["funding_rate"]) if row.get("funding_rate") is not None else None,
            float(row["funding_scalar"]),
            float(row["cost_bps"]),
        )
        for row in rows
    ]
    with _conn(db_path) as con:
        before = con.total_changes
        con.executemany(
            "INSERT OR IGNORE INTO shadow_targets "
            "(model, cycle_id, cycle_ts, recorded_ts, symbol, price, signal, raw_weight, "
            "target_weight, portfolio_weight, portfolio_scale, funding_rate, funding_scalar, "
            "cost_bps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return con.total_changes - before


def record_closed_position(
    position_key: str,
    opened_ts: str,
    closed_ts: str,
    symbol: str,
    side: str,
    peak_qty: float,
    entry_price: float,
    exit_price: float,
    net_pnl: float,
    commissions: float,
    *,
    initial_risk: float | None = None,
    exit_reason: str = "OTHER",
    db_path: str = DEFAULT_DB,
) -> bool:
    """Persist one fully closed position cycle, idempotently."""
    r_multiple = net_pnl / initial_risk if initial_risk and initial_risk > 0 else None
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO closed_positions "
            "(position_key, opened_ts, closed_ts, symbol, side, peak_qty, entry_price, "
            "exit_price, net_pnl, commissions, initial_risk, r_multiple, exit_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                position_key, opened_ts, closed_ts, symbol, side, peak_qty,
                entry_price, exit_price, net_pnl, commissions, initial_risk,
                r_multiple, exit_reason,
            ),
        )
        return cur.rowcount > 0


def update_basket_cycle(
    direction: str,
    symbols: list[str],
    equity: float,
    initial_risk: float | None,
    db_path: str = DEFAULT_DB,
    *,
    started_ts: str | None = None,
) -> dict | None:
    """Advance the one correlated crypto-basket lifecycle.

    A basket begins when the portfolio goes from flat to invested and completes
    only when it becomes flat or directly reverses direction. Individual coin
    positions remain diagnostics; they do not inflate the forward-test count.
    """
    now = _now()
    direction = direction.upper()
    with _conn(db_path) as con:
        active = con.execute(
            "SELECT * FROM basket_cycles WHERE status='active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        reverses = bool(
            active
            and direction in {"LONG", "SHORT"}
            and active["direction"] in {"LONG", "SHORT"}
            and direction != active["direction"]
        )
        if active and (direction == "FLAT" or reverses):
            pnl = equity - float(active["starting_equity"])
            risk = active["initial_risk"]
            r_multiple = pnl / risk if risk and risk > 0 else None
            reason_rows = con.execute(
                "SELECT kind, COUNT(*) AS n FROM fills WHERE ts >= ? "
                "AND kind IN ('STOP', 'TRAIL', 'TP') GROUP BY kind",
                (active["started_ts"],),
            ).fetchall()
            reasons = {row["kind"]: int(row["n"]) for row in reason_rows}
            if not reasons:
                reasons = {"TREND_OR_MANUAL": 1}
            con.execute(
                "UPDATE basket_cycles SET ended_ts=?, ending_equity=?, "
                "peak_equity=MAX(peak_equity, ?), trough_equity=MIN(trough_equity, ?), "
                "net_pnl=?, r_multiple=?, exit_reasons=?, status='completed' WHERE id=?",
                (now, equity, equity, equity, pnl, r_multiple, json.dumps(reasons), active["id"]),
            )
            active = None
        if direction != "FLAT":
            if active is None:
                inferred_start_ts = started_ts or now
                historical = con.execute(
                    "SELECT equity FROM equity WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
                    (inferred_start_ts,),
                ).fetchone()
                if historical is None:
                    historical = con.execute(
                        "SELECT equity FROM equity WHERE ts < ? ORDER BY ts DESC LIMIT 1",
                        (inferred_start_ts,),
                    ).fetchone()
                starting_equity = float(historical["equity"]) if historical else equity
                extrema = con.execute(
                    "SELECT MIN(equity) AS trough, MAX(equity) AS peak FROM equity WHERE ts >= ?",
                    (inferred_start_ts,),
                ).fetchone()
                peak_equity = max(equity, float(extrema["peak"] or equity), starting_equity)
                trough_equity = min(equity, float(extrema["trough"] or equity), starting_equity)
                cur = con.execute(
                    "INSERT INTO basket_cycles "
                    "(started_ts, direction, symbols, starting_equity, peak_equity, "
                    "trough_equity, initial_risk, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
                    (
                        inferred_start_ts, direction, json.dumps(sorted(symbols)),
                        starting_equity, peak_equity, trough_equity, initial_risk,
                    ),
                )
                basket_id = cur.lastrowid
            else:
                basket_id = int(active["id"])
                known_symbols = set(json.loads(active["symbols"])) | set(symbols)
                con.execute(
                    "UPDATE basket_cycles SET symbols=?, peak_equity=MAX(peak_equity, ?), "
                    "trough_equity=MIN(trough_equity, ?), initial_risk=COALESCE(initial_risk, ?) "
                    "WHERE id=?",
                    (json.dumps(sorted(known_symbols)), equity, equity, initial_risk, basket_id),
                )
            row = con.execute("SELECT * FROM basket_cycles WHERE id=?", (basket_id,)).fetchone()
            return dict(row)
    return None


def record_event(kind: str, details: str = "", db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO ops_events (ts, kind, details) VALUES (?, ?, ?)",
            (_now(), kind, details),
        )


def start_incident(kind: str, details: str = "", db_path: str = DEFAULT_DB) -> int:
    """Open an incident once; repeated watchdog polls remain idempotent."""
    with _conn(db_path) as con:
        row = con.execute(
            "SELECT id FROM incidents WHERE kind=? AND ended_ts IS NULL ORDER BY id DESC LIMIT 1",
            (kind,),
        ).fetchone()
        if row:
            return int(row["id"])
        cur = con.execute(
            "INSERT INTO incidents (kind, started_ts, details) VALUES (?, ?, ?)",
            (kind, _now(), details),
        )
        return int(cur.lastrowid)


def resolve_incident(kind: str, db_path: str = DEFAULT_DB) -> None:
    with _conn(db_path) as con:
        con.execute(
            "UPDATE incidents SET ended_ts=? WHERE kind=? AND ended_ts IS NULL",
            (_now(), kind),
        )


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
            "funding, slippage, total_pnl, basket_direction, average_correlation, "
            "effective_bets, market_regime FROM equity ORDER BY ts DESC LIMIT ?",
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


def read_shadow_targets(limit: int = 5000, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT model, cycle_id, cycle_ts, recorded_ts, symbol, price, signal, "
            "raw_weight, target_weight, portfolio_weight, portfolio_scale, funding_rate, "
            "funding_scalar, cost_bps FROM shadow_targets "
            "ORDER BY cycle_ts DESC, model, symbol LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def shadow_report(db_path: str = DEFAULT_DB) -> dict:
    """Score shadow targets using next-cycle price returns and modeled turnover costs.

    Funding is deliberately not estimated from a point-in-time rate. The income
    ledger remains the source of truth for live funding, while shadow reports
    disclose this limitation instead of manufacturing precision.
    """
    rows = read_shadow_targets(limit=100000, db_path=db_path)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["model"])][str(row["cycle_id"])].append(row)

    models = {}
    for model, cycles_by_id in grouped.items():
        all_cycles = sorted(
            cycles_by_id.values(),
            key=lambda cycle: datetime.fromisoformat(str(cycle[0]["cycle_ts"])),
        )
        if not all_cycles:
            continue
        # Live market-bar cycle IDs are integer nanosecond timestamps. Startup
        # observations contain a run UUID and are useful for showing current
        # targets, but restarts are not independent performance periods.
        cycles = [cycle for cycle in all_cycles if str(cycle[0]["cycle_id"]).isdigit()]
        virtual_equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        period_returns: list[float] = []
        previous: dict[str, dict] | None = None
        first_ts = (
            datetime.fromisoformat(str(cycles[0][0]["cycle_ts"])) if cycles else None
        )
        last_ts = first_ts
        for cycle in cycles:
            current = {str(row["symbol"]): row for row in cycle}
            current_ts = datetime.fromisoformat(str(cycle[0]["cycle_ts"]))
            last_ts = max(last_ts, current_ts) if last_ts is not None else current_ts
            if previous is None:
                entry_turnover = sum(abs(float(row["portfolio_weight"])) for row in current.values())
                cost_bps = max((float(row["cost_bps"]) for row in current.values()), default=0.0)
                virtual_equity *= 1.0 - entry_turnover * cost_bps / 1e4
            else:
                price_return = 0.0
                for symbol, prior in previous.items():
                    now = current.get(symbol)
                    if now is None or float(prior["price"]) <= 0:
                        continue
                    asset_return = float(now["price"]) / float(prior["price"]) - 1.0
                    price_return += float(prior["portfolio_weight"]) * asset_return
                symbols = set(previous) | set(current)
                turnover = sum(
                    abs(
                        float(current.get(symbol, {}).get("portfolio_weight", 0.0))
                        - float(previous.get(symbol, {}).get("portfolio_weight", 0.0))
                    )
                    for symbol in symbols
                )
                cost_bps = max((float(row["cost_bps"]) for row in current.values()), default=0.0)
                net_return = price_return - turnover * cost_bps / 1e4
                period_returns.append(net_return)
                virtual_equity *= max(0.0, 1.0 + net_return)
            peak = max(peak, virtual_equity)
            if peak > 0:
                max_drawdown = min(max_drawdown, virtual_equity / peak - 1.0)
            previous = current

        elapsed_days = (
            max(0.0, (last_ts - first_ts).total_seconds() / 86400.0)
            if first_ts is not None and last_ts is not None
            else 0.0
        )
        enough_history = len(period_returns) >= 7 and elapsed_days >= 7.0
        annualized_return = (
            virtual_equity ** (365.0 / elapsed_days) - 1.0
            if enough_history and virtual_equity > 0
            else None
        )
        sharpe = None
        if enough_history:
            mean_return = sum(period_returns) / len(period_returns)
            variance = sum((value - mean_return) ** 2 for value in period_returns) / (
                len(period_returns) - 1
            )
            periods_per_year = 365.0 / max(elapsed_days / len(period_returns), 1e-9)
            if variance > 0:
                sharpe = mean_return / math.sqrt(variance) * math.sqrt(periods_per_year)
        latest_cycle = all_cycles[-1]
        latest = {
            row["symbol"]: {
                "signal": row["signal"],
                "target_weight": row["target_weight"],
                "portfolio_weight": row["portfolio_weight"],
                "portfolio_scale": row["portfolio_scale"],
                "funding_rate": row["funding_rate"],
                "funding_scalar": row["funding_scalar"],
            }
            for row in latest_cycle
        }
        models[model] = {
            "cycles": len(cycles),
            "startup_observations": len(all_cycles) - len(cycles),
            "scored_periods": len(period_returns),
            "total_return": virtual_equity - 1.0,
            "annualized_return": annualized_return,
            "sharpe": sharpe,
            "maximum_drawdown": max_drawdown,
            "latest_cycle_ts": latest_cycle[0]["cycle_ts"],
            "latest_targets": latest,
        }
    return {
        "models": models,
        "method": (
            "next-market-cycle price return minus configured turnover cost; "
            "startup/restart observations excluded"
        ),
        "funding_note": "Shadow funding PnL is not simulated from point-in-time rates.",
    }


def read_closed_positions(limit: int = 500, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM closed_positions ORDER BY closed_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def read_basket_cycles(limit: int = 200, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM basket_cycles ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["symbols"] = json.loads(item["symbols"] or "[]")
        item["exit_reasons"] = json.loads(item["exit_reasons"] or "{}")
        result.append(item)
    return result


def read_incidents(limit: int = 500, db_path: str = DEFAULT_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM incidents ORDER BY started_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def forward_test_report(db_path: str = DEFAULT_DB) -> dict:
    """Return the acceptance scorecard for the 30–50 basket-cycle test."""
    snap = read_snapshot(db_path) or {}
    perf = snap.get("performance") or {}
    equity_rows = read_equity(limit=100000, db_path=db_path)
    baskets = read_basket_cycles(limit=100000, db_path=db_path)
    closed = [row for row in baskets if row["status"] == "completed"]
    active = next((row for row in baskets if row["status"] == "active"), None)
    known_r = [float(row["r_multiple"]) for row in closed if row["r_multiple"] is not None]

    peak = None
    max_drawdown = 0.0
    for row in equity_rows:
        value = float(row["equity"])
        peak = value if peak is None else max(peak, value)
        if peak:
            max_drawdown = min(max_drawdown, value / peak - 1.0)

    regimes = sorted(
        {
            str(row["basket_direction"])
            for row in equity_rows
            if row.get("basket_direction") in {"LONG", "SHORT", "MIXED"}
        }
    )
    market_regimes = sorted(
        {
            str(row["market_regime"])
            for row in equity_rows
            if row.get("market_regime") and "UNKNOWN" not in str(row["market_regime"])
        }
    )
    volatility_regimes = sorted({regime.rsplit("_", 2)[-2] + "_VOL" for regime in market_regimes})
    incidents = read_incidents(limit=100000, db_path=db_path)
    incident_stats = {}
    now = datetime.now(timezone.utc)
    for kind in ("MARK_STALE", "BOT_DOWN"):
        selected = [row for row in incidents if row["kind"] == kind]
        seconds = 0.0
        for row in selected:
            start = datetime.fromisoformat(row["started_ts"])
            end = datetime.fromisoformat(row["ended_ts"]) if row["ended_ts"] else now
            seconds += max(0.0, (end - start).total_seconds())
        incident_stats[kind] = {
            "count": len(selected),
            "open": sum(row["ended_ts"] is None for row in selected),
            "duration_secs": seconds,
        }

    with _conn(db_path) as con:
        starts = int(
            con.execute("SELECT COUNT(*) AS n FROM ops_events WHERE kind='BOT_START'").fetchone()["n"]
        )
        exits = {
            row["exit_reason"]: int(row["n"])
            for row in con.execute(
                "SELECT exit_reason, COUNT(*) AS n FROM closed_positions GROUP BY exit_reason"
            ).fetchall()
        }

    count = len(closed)
    if count < 30:
        status = "COLLECTING"
    elif len(market_regimes) < 2:
        status = "REGIME_COVERAGE_REQUIRED"
    elif len(known_r) < count:
        status = "R_DATA_INCOMPLETE"
    elif count < 50:
        status = "MINIMUM_REACHED"
    else:
        status = "REVIEW_READY"
    start_equity = float(perf.get("starting_equity") or 0.0)
    total_pnl = float(perf.get("total_pnl") or 0.0)
    return {
        "status": status,
        "completed_basket_cycles": count,
        "completed_symbol_positions": len(read_closed_positions(limit=100000, db_path=db_path)),
        "minimum_target": 30,
        "preferred_target": 50,
        "minimum_progress_pct": min(100.0, count / 30 * 100.0),
        "preferred_progress_pct": min(100.0, count / 50 * 100.0),
        "active_basket": active,
        "net_return_pct": total_pnl / start_equity * 100.0 if start_equity else None,
        "maximum_drawdown_pct": max_drawdown * 100.0,
        "average_r": sum(known_r) / len(known_r) if known_r else None,
        "r_known_cycles": len(known_r),
        "commissions": perf.get("commissions"),
        "funding_payments": perf.get("funding_payments"),
        "slippage": perf.get("slippage"),
        "exit_reason_counts": exits,
        "bot_starts": starts,
        "restart_count": max(0, starts - 1),
        "incidents": incident_stats,
        "directional_regimes_seen": regimes,
        "multiple_directional_regimes_seen": len(regimes) >= 2,
        "market_regimes_seen": market_regimes,
        "volatility_regimes_seen": volatility_regimes,
        "multiple_market_regimes_seen": len(market_regimes) >= 2,
        "qualification_note": (
            "Each simultaneous correlated crypto book counts as one basket cycle; "
            "symbol positions are diagnostic only. Passing the trade count alone is not approval."
        ),
    }
