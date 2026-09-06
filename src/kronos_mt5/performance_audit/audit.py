"""Orchestration: load a source, run every analysis, assemble the result."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kronos_mt5.performance_audit import accounting as accounting_mod
from kronos_mt5.performance_audit import equity as equity_mod
from kronos_mt5.performance_audit import fills as fills_mod
from kronos_mt5.performance_audit import operations as operations_mod
from kronos_mt5.performance_audit import positions as positions_mod
from kronos_mt5.performance_audit import shadow as shadow_mod
from kronos_mt5.performance_audit.findings import build_findings, summarize_findings
from kronos_mt5.performance_audit.loading import (
    columns,
    describe_schema,
    parse_ts,
    select_all,
)
from kronos_mt5.performance_audit.source import ExportSource, read_only_connection
from kronos_mt5.performance_audit.version import SCHEMA_VERSION, TOOL_VERSION

# Config keys worth echoing into the report. All are non-secret tunables; the
# export already stripped every sensitive value before this tool ever sees it.
CONTEXT_CONFIG_KEYS = (
    "BINANCE_ENVIRONMENT",
    "BINANCE_SYMBOLS",
    "BINANCE_TARGET_VOL",
    "BINANCE_MAX_DRAWDOWN",
    "BINANCE_USE_PATIENT_LIMIT",
    "BINANCE_USE_PORTFOLIO_ALLOCATOR",
    "BINANCE_SHADOW_ENABLED",
    "DEMO_ONLY",
)


def _truthy(value) -> bool | None:
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_context(source: ExportSource, con: sqlite3.Connection, fill_rows: list[dict]) -> dict:
    meta = source.metadata or {}
    config = meta.get("config") or {}
    symbols = sorted({str(r.get("symbol")) for r in fill_rows if r.get("symbol")})
    ohlcv_files = 0
    if source.root is not None:
        inventory = list(source.root.rglob("market_data_inventory.json"))
        if inventory:
            try:
                data = json.loads(inventory[0].read_text())
                ohlcv_files = len(
                    [f for f in data.get("files", []) if f.get("symbol") and f.get("timeframe")]
                )
            except (OSError, ValueError):
                ohlcv_files = 0

    accounting_start = None
    try:
        row = con.execute("SELECT value FROM kv WHERE name = 'accounting_start_ts'").fetchone()
        accounting_start = row[0] if row else None
    except sqlite3.Error:
        accounting_start = None

    source_block = {
        key: value
        for key, value in sorted(meta.items())
        if key.startswith(("source_", "export_")) and value
    }
    return {
        "source": source_block,
        "config": {k: v for k, v in sorted(config.items()) if k in CONTEXT_CONFIG_KEYS},
        "binance_environment": config.get("BINANCE_ENVIRONMENT"),
        "demo_only": _truthy(config.get("DEMO_ONLY")),
        "symbols": symbols,
        "ohlcv_available": ohlcv_files > 0,
        "market_data_files": ohlcv_files,
        "accounting_start_ts": accounting_start,
    }


def run_analysis(source: ExportSource) -> dict:
    """Run every analysis over `source`. Deterministic for identical input bytes."""
    with read_only_connection(source.database_path) as con:
        schema = describe_schema(con)
        equity_rows = select_all(con, "equity", order_by="ts")
        fill_rows = select_all(con, "fills", order_by="ts")
        income_rows = select_all(con, "income", order_by="ts")
        position_rows = select_all(con, "closed_positions", order_by="closed_ts")
        incident_rows = select_all(con, "incidents", order_by="started_ts")
        ops_rows = select_all(con, "ops_events", order_by="ts")
        shadow_rows = select_all(con, "shadow_targets", order_by="cycle_ts")
        fill_columns = columns(con, "fills")
        context = _build_context(source, con, fill_rows)

    equity_result = equity_mod.analyze_equity(equity_rows)
    daily = equity_result.get("daily") or []
    execution_result = fills_mod.analyze_fills(fill_rows, fill_columns)
    accounting_result = accounting_mod.reconcile(equity_rows, income_rows, fill_rows, daily)
    positions_result = positions_mod.analyze_positions(position_rows, len(fill_rows))
    shadow_result = shadow_mod.analyze_shadow(shadow_rows)
    operations_result = operations_mod.analyze_operations(
        incident_rows, ops_rows, equity_rows, equity_result.get("observation_gaps") or []
    )
    regimes_result = accounting_mod.detect_regimes(ops_rows, daily)

    # Cross-check the income window against the equity window: an income cursor
    # that starts late is a common source of an unexplained residual.
    income_first = parse_ts((accounting_result.get("income") or {}).get("first_ts"))
    equity_first = parse_ts(equity_result.get("first_observation"))
    if income_first and equity_first:
        accounting_result["income_starts_after_equity_hours"] = (
            income_first - equity_first
        ).total_seconds() / 3600.0

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "context": context,
        "schema": schema,
        "equity": equity_result,
        "accounting": accounting_result,
        "execution": execution_result,
        "positions": positions_result,
        "shadow": shadow_result,
        "operations": operations_result,
        "regimes": regimes_result,
    }
    analysis["findings"] = build_findings(analysis)
    analysis["findings_summary"] = summarize_findings(analysis["findings"])
    return analysis


def build_run_metadata(source: ExportSource, generated_at_utc: str) -> dict:
    """Everything that varies between runs, kept out of `analysis`."""
    return {
        "generated_at_utc": generated_at_utc,
        "tool_version": TOOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "input_name": Path(source.input_path).name,
        "input_kind": source.kind,
        "input_sha256": source.sha256,
        "database_name": Path(source.database_path).name,
    }
