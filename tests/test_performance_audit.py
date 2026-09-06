"""Offline performance-audit tests.

Every fixture is synthetic and generated inside the test run — no real trading
data is committed. Analysis functions are exercised directly where possible so
the expected numbers can be written by hand.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kronos_mt5.performance_audit import accounting, equity, fills, positions, shadow
from kronos_mt5.performance_audit.audit import build_run_metadata, run_analysis
from kronos_mt5.performance_audit.cli import main
from kronos_mt5.performance_audit.findings import build_findings
from kronos_mt5.performance_audit.source import (
    AuditInputError,
    UnsafeArchiveError,
    read_only_connection,
    resolve_source,
    safe_extract,
    sha256_file,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ts(day: int, hour: int = 0, minute: int = 0) -> str:
    return (START + timedelta(days=day, hours=hour, minutes=minute)).isoformat()


def _equity_row(day: int, value: float, *, hour: int = 23, n_open: int = 2, **extra) -> dict:
    row = {
        "ts": _ts(day, hour),
        "equity": value,
        "cash": value,
        "unrealized": 0.0,
        "n_open": n_open,
        "realized": None,
        "commissions": None,
        "funding": None,
        "slippage": None,
        "total_pnl": None,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# equity metrics
# --------------------------------------------------------------------------


def test_total_return_is_exact():
    rows = [_equity_row(0, 1000.0), _equity_row(1, 1100.0), _equity_row(2, 1210.0)]
    result = equity.analyze_equity(rows)
    assert result["start_equity"] == 1000.0
    assert result["end_equity"] == 1210.0
    assert result["absolute_return"] == pytest.approx(210.0)
    assert result["total_return_pct"] == pytest.approx(21.0)
    # two 10% daily steps
    assert [round(r["return"], 10) for r in result["daily"][1:]] == [0.1, 0.1]
    assert result["positive_days"] == 2
    assert result["negative_days"] == 0


def test_max_drawdown_value_and_dates_are_exact():
    # 100 -> 120 (peak) -> 90 (trough, -25%) -> 130 (recovery)
    values = [100.0, 120.0, 110.0, 90.0, 125.0, 130.0]
    rows = [_equity_row(i, v) for i, v in enumerate(values)]
    result = equity.analyze_equity(rows)
    assert result["drawdown_max_drawdown_pct"] == pytest.approx(-25.0)
    assert result["drawdown_peak_date"] == "2026-01-02"
    assert result["drawdown_trough_date"] == "2026-01-04"
    assert result["drawdown_peak_equity"] == 120.0
    assert result["drawdown_trough_equity"] == 90.0
    assert result["drawdown_recovered"] is True
    assert result["drawdown_recovery_date"] == "2026-01-05"
    assert result["drawdown_drawdown_days"] == 2


def test_unrecovered_drawdown_reports_no_recovery():
    rows = [_equity_row(i, v) for i, v in enumerate([100.0, 120.0, 80.0, 90.0])]
    result = equity.analyze_equity(rows)
    assert result["drawdown_recovered"] is False
    assert result["drawdown_recovery_date"] is None


def test_daily_resampling_keeps_the_last_observation_of_each_day():
    rows = [
        _equity_row(0, 100.0, hour=1),
        _equity_row(0, 105.0, hour=12),
        _equity_row(0, 110.0, hour=23),  # this one wins
        _equity_row(1, 90.0, hour=2),
        _equity_row(1, 120.0, hour=22),  # and this one
    ]
    result = equity.analyze_equity(rows)
    assert [d["equity"] for d in result["daily"]] == [110.0, 120.0]
    assert result["days_observed"] == 2
    assert result["raw_observations"] == 5
    # raw extremes are preserved separately for reconciliation
    raw = result["observation_series"]
    assert raw["min_equity"] == 90.0
    assert raw["max_equity"] == 120.0
    assert raw["first_equity"] == 100.0


def test_duplicate_and_invalid_equity_rows_are_counted_not_fatal():
    rows = [
        _equity_row(0, 100.0),
        _equity_row(0, 100.0),  # exact duplicate timestamp
        _equity_row(1, 0.0, hour=10),  # invalid: non-positive
        _equity_row(1, -5.0, hour=11),  # invalid: negative
        {"ts": "not-a-timestamp", "equity": 100.0},
        {"ts": _ts(2), "equity": None},
        _equity_row(3, 110.0),
    ]
    result = equity.analyze_equity(rows)
    assert result["duplicate_timestamps"] == 1
    assert result["invalid_equity_rows"] == 3  # 0.0, -5.0 and the None
    assert result["unparsable_timestamps"] == 1
    assert [d["equity"] for d in result["daily"]] == [100.0, 110.0]
    assert result["missing_day_count"] == 2  # 02 Jan and 03 Jan have no valid value


def test_missing_days_and_observation_gaps_are_reported():
    rows = [_equity_row(0, 100.0), _equity_row(5, 105.0)]
    result = equity.analyze_equity(rows)
    assert result["missing_day_count"] == 4
    assert result["observation_gap_count"] == 1
    assert result["observation_gaps"][0]["hours"] == pytest.approx(120.0)


def test_zero_variance_and_single_point_do_not_crash():
    flat = equity.analyze_equity([_equity_row(i, 100.0) for i in range(5)])
    assert flat["zero_variance"] is True
    assert flat["sharpe_ratio"] is None
    assert flat["annualized_volatility_pct"] == 0.0

    single = equity.analyze_equity([_equity_row(0, 100.0)])
    assert single["available"] is True
    assert single["sharpe_ratio"] is None

    empty = equity.analyze_equity([])
    assert empty["available"] is False


def test_short_history_is_flagged_low_confidence():
    rows = [_equity_row(i, 100.0 + i) for i in range(40)]
    result = equity.analyze_equity(rows)
    assert result["annualized_low_confidence"] is True
    assert "annualized" in result["annualized_confidence_note"]


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------


def _income(kind: str, amount: float, day: int, index: int) -> dict:
    return {
        "income_id": f"{kind}-{index}",
        "ts": _ts(day),
        "income_type": kind,
        "amount": amount,
        "symbol": "BTCUSDT-PERP",
    }


def test_accounting_residual_is_exact():
    equity_rows = [
        _equity_row(0, 1000.0, unrealized=0.0),
        _equity_row(1, 1010.0, unrealized=5.0),
    ]
    income_rows = [
        _income("REALIZED_PNL", 8.0, 0, 1),
        _income("COMMISSION", -1.0, 0, 2),
        _income("FUNDING_FEE", 0.5, 0, 3),
    ]
    daily = equity.analyze_equity(equity_rows)["daily"]
    result = accounting.reconcile(equity_rows, income_rows, [], daily)
    # explained = 8 - 1 + 0.5 + 0 + (5 - 0) = 12.5 ; equity change = 10
    assert result["equity_change"] == pytest.approx(10.0)
    assert result["explained_change"] == pytest.approx(12.5)
    assert result["residual"] == pytest.approx(-2.5)
    assert result["commissions_cost"] == pytest.approx(1.0)
    assert result["reconciled_within_tolerance"] is False


def test_accounting_reconciles_within_tolerance():
    equity_rows = [_equity_row(0, 1000.0), _equity_row(1, 1009.5)]
    income_rows = [
        _income("REALIZED_PNL", 10.0, 0, 1),
        _income("COMMISSION", -0.5, 0, 2),
    ]
    daily = equity.analyze_equity(equity_rows)["daily"]
    result = accounting.reconcile(equity_rows, income_rows, [], daily)
    assert result["residual"] == pytest.approx(0.0, abs=1e-9)
    assert result["reconciled_within_tolerance"] is True


def test_balance_discontinuity_is_detected():
    rows = [_equity_row(0, 1000.0), _equity_row(1, 2000.0), _equity_row(2, 2010.0)]
    daily = equity.analyze_equity(rows)["daily"]
    jumps = accounting.find_discontinuities(daily)
    assert len(jumps) == 1
    assert jumps[0]["change_pct"] == pytest.approx(100.0)
    assert "deposit" in jumps[0]["likely_explanation"]


def test_duplicate_income_ids_are_reported():
    rows = [_income("REALIZED_PNL", 1.0, 0, 1), _income("REALIZED_PNL", 1.0, 0, 1)]
    assert accounting.summarize_income(rows)["duplicate_income_ids"] == 1


def test_restart_regimes_are_detected():
    ops = [
        {"ts": _ts(0), "kind": "BOT_START"},
        {"ts": _ts(1), "kind": "BOT_STOP"},
        {"ts": _ts(1, 1), "kind": "BOT_START"},
    ]
    result = accounting.detect_regimes(ops, [])
    assert result["starts"] == 2
    assert result["continuous_single_run"] is False
    assert result["distinct_restart_days"] == 2


# --------------------------------------------------------------------------
# fills
# --------------------------------------------------------------------------


def _fill(index: int, **extra) -> dict:
    row = {
        "fill_id": f"f{index}",
        "ts": _ts(0, index),
        "symbol": "BTCUSDT-PERP",
        "side": "BUY",
        "qty": 2.0,
        "price": 100.0,
        "kind": "TREND",
        "trade_id": f"t{index}",
        "commission": 0.1,
        "reference_price": 99.0,
        "slippage": 2.0,
        "reconciliation": 0,
    }
    row.update(extra)
    return row


def test_fill_attribution_groups_and_totals():
    rows = [_fill(1), _fill(2, side="SELL"), _fill(3, symbol="ETHUSDT-PERP")]
    result = fills.analyze_fills(rows)
    assert result["total_fills"] == 3
    assert result["totals"]["quote_turnover"] == pytest.approx(600.0)
    assert result["totals"]["commission"] == pytest.approx(0.3)
    assert result["totals"]["slippage_quote"] == pytest.approx(6.0)
    assert len(result["groups"]) == 3
    assert result["by_symbol"]["ETHUSDT-PERP"]["fills"] == 1
    # the legacy column's unit is stated, never assumed
    assert result["slippage_unit"] == "quote_currency_signed_positive_is_worse"


def test_missing_fill_fields_are_counted():
    rows = [
        _fill(1, kind=None),
        _fill(2, reference_price=None, slippage=None),
        _fill(3, commission=None),
        _fill(4, trade_id=None, reconciliation=1),
    ]
    totals = fills.analyze_fills(rows)["totals"]
    assert totals["missing_kind"] == 1
    assert totals["missing_reference_price"] == 1
    assert totals["missing_slippage"] == 1
    assert totals["missing_commission"] == 1
    assert totals["missing_trade_id"] == 1
    assert totals["reconciliation_fills"] == 1


def test_execution_telemetry_absence_is_explicit():
    legacy = fills.analyze_fills([_fill(1)])
    assert legacy["execution_telemetry_available"] is False
    assert "impl_shortfall_bps" in legacy["telemetry_columns_missing"]

    modern_columns = list(_fill(1).keys()) + list(fills.TELEMETRY_COLUMNS)
    modern_rows = [
        _fill(1, liquidity="MAKER", exec_role="PATIENT_LIMIT", impl_shortfall_bps=-2.0),
        _fill(2, liquidity="TAKER", exec_role="PATIENT_FALLBACK", impl_shortfall_bps=20.0),
    ]
    modern = fills.analyze_fills(modern_rows, modern_columns)
    assert modern["execution_telemetry_available"] is True
    assert modern["totals"]["avg_shortfall_bps"] == pytest.approx(9.0)
    assert modern["totals"]["weighted_shortfall_bps"] == pytest.approx(9.0)


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------


def _position(index: int, pnl: float, **extra) -> dict:
    row = {
        "position_key": f"p{index}",
        "opened_ts": _ts(index),
        "closed_ts": _ts(index + 1),
        "symbol": "BTCUSDT-PERP",
        "side": "LONG",
        "net_pnl": pnl,
        "r_multiple": 1.0 if pnl > 0 else -1.0,
        "exit_reason": "TREND_OR_MANUAL",
    }
    row.update(extra)
    return row


def test_position_statistics_are_exact():
    rows = [_position(1, 10.0), _position(2, -4.0), _position(3, 6.0)]
    result = positions.analyze_positions(rows, total_fills=20)
    assert result["wins"] == 2 and result["losses"] == 1
    assert result["win_rate_pct"] == pytest.approx(200 / 3)
    assert result["gross_profit"] == pytest.approx(16.0)
    assert result["gross_loss"] == pytest.approx(4.0)
    assert result["profit_factor"] == pytest.approx(4.0)
    assert result["expectancy"] == pytest.approx(4.0)
    assert result["avg_holding_hours"] == pytest.approx(24.0)
    assert result["metrics_reliable"] is True


def test_incomplete_position_lifecycle_is_flagged():
    """4 closed positions behind 460 fills cannot support trade statistics."""
    rows = [_position(i, 1.0) for i in range(4)]
    result = positions.analyze_positions(rows, total_fills=460)
    assert result["lifecycle_coverage_incomplete"] is True
    assert result["metrics_reliable"] is False
    assert "MUST NOT" in result["reliability_note"]
    assert result["trades_are_not_reconstructed_from_fills"] is True

    findings = build_findings(
        {"positions": result, "context": {"ohlcv_available": True}, "execution": {}}
    )
    codes = {f["code"] for f in findings}
    assert "PA-DATA-002" in codes
    lifecycle = next(f for f in findings if f["code"] == "PA-DATA-002")
    assert lifecycle["prominent"] is True
    assert lifecycle["severity"] == "WARNING"


# --------------------------------------------------------------------------
# shadow models
# --------------------------------------------------------------------------


def _shadow_row(model: str, cycle: int, symbol: str, weight: float, signal: float) -> dict:
    return {
        "model": model,
        "cycle_id": f"c{cycle}",
        "cycle_ts": _ts(cycle),
        "recorded_ts": _ts(cycle),
        "symbol": symbol,
        "price": 100.0,
        "signal": signal,
        "raw_weight": weight,
        "target_weight": weight,
        "portfolio_weight": weight,
        "portfolio_scale": 1.0,
        "funding_rate": 0.0001,
        "funding_scalar": 1.0,
        "cost_bps": 8.0,
    }


def test_shadow_pnl_is_not_evaluable_without_ohlcv():
    rows = [
        _shadow_row("live", 0, "BTCUSDT-PERP", 0.5, 1.0),
        _shadow_row("challenger", 0, "BTCUSDT-PERP", -0.5, -1.0),
    ]
    result = shadow.analyze_shadow(rows)
    evaluation = result["pnl_evaluation"]
    assert evaluation["status"] == "NOT_EVALUABLE"
    assert len(evaluation["missing_data"]) >= 5
    assert any("OHLCV" in item for item in evaluation["missing_data"])
    assert "shadow Sharpe" in evaluation["explicitly_not_computed"]
    # no profitability key of any kind is produced
    assert not any(k in json.dumps(result) for k in ("shadow_return_pct", "shadow_sharpe"))


def test_shadow_positioning_metrics_and_divergence():
    rows = [
        _shadow_row("live", 0, "BTC", 0.5, 1.0),
        _shadow_row("live", 1, "BTC", 0.1, 1.0),
        _shadow_row("challenger", 0, "BTC", 0.5, 1.0),
        _shadow_row("challenger", 1, "BTC", -0.4, -1.0),
    ]
    result = shadow.analyze_shadow(rows)
    live = result["models"]["live"]
    assert live["cycles"] == 2
    assert live["gross_exposure"]["mean"] == pytest.approx(0.3)
    assert live["turnover_per_cycle"]["mean"] == pytest.approx(0.4)

    divergence = result["comparisons_vs_live"]["challenger"]
    assert divergence["shared_cycles"] == 2
    assert divergence["signal_sign_disagreements"] == 1
    assert divergence["materially_different_cycles"] == 1


# --------------------------------------------------------------------------
# archive safety
# --------------------------------------------------------------------------


def _make_db(path: Path, *, tables: dict[str, list[dict]]) -> None:
    con = sqlite3.connect(path)
    for table, rows in tables.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        con.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        con.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
    con.commit()
    con.close()


@pytest.fixture
def synthetic_db(tmp_path) -> Path:
    path = tmp_path / "companion_sanitized.db"
    _make_db(
        path,
        tables={
            "equity": [_equity_row(i, 1000.0 + i * 5) for i in range(10)],
            "fills": [_fill(i) for i in range(6)],
            "income": [
                _income("REALIZED_PNL", 50.0, 0, 1),
                _income("COMMISSION", -0.6, 0, 2),
            ],
            "closed_positions": [_position(1, 10.0)],
            "incidents": [
                {
                    "id": 1,
                    "kind": "MARK_STALE",
                    "started_ts": _ts(2),
                    "ended_ts": _ts(2, 1),
                    "details": "SYMBOL",
                }
            ],
            "ops_events": [{"id": 1, "ts": _ts(0), "kind": "BOT_START", "details": ""}],
            "shadow_targets": [_shadow_row("live", 0, "BTC", 0.5, 1.0)],
            "kv": [{"name": "accounting_start_ts", "value": _ts(0), "ts": _ts(0)}],
        },
    )
    return path


def test_safe_extract_accepts_a_normal_archive(tmp_path, synthetic_db):
    archive = tmp_path / "export.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(synthetic_db, arcname="export/database/companion_sanitized.db")
    dest = tmp_path / "out"
    safe_extract(archive, dest)
    assert (dest / "export" / "database" / "companion_sanitized.db").is_file()


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")
    with pytest.raises(UnsafeArchiveError, match="traversal"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_rejects_absolute_paths(tmp_path):
    archive = tmp_path / "abs.tar.gz"
    info = tarfile.TarInfo("/etc/passwd")
    info.size = 0
    with tarfile.open(archive, "w:gz") as tar, io.BytesIO(b"") as payload:
        tar.addfile(info, payload)
    with pytest.raises(UnsafeArchiveError, match="absolute"):
        safe_extract(archive, tmp_path / "out")


def test_safe_extract_rejects_symlinks(tmp_path):
    archive = tmp_path / "link.tar.gz"
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(info)
    with pytest.raises(UnsafeArchiveError, match="link"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "link").exists()


def test_safe_extract_rejects_hardlinks(tmp_path):
    archive = tmp_path / "hard.tar.gz"
    info = tarfile.TarInfo("hard")
    info.type = tarfile.LNKTYPE
    info.linkname = "target"
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(info)
    with pytest.raises(UnsafeArchiveError, match="link"):
        safe_extract(archive, tmp_path / "out")


def test_malformed_archive_raises_actionable_error(tmp_path):
    broken = tmp_path / "broken.tar.gz"
    broken.write_bytes(b"definitely not a tarball")
    with pytest.raises(AuditInputError, match="malformed archive"):
        safe_extract(broken, tmp_path / "out")


# --------------------------------------------------------------------------
# read-only guarantees
# --------------------------------------------------------------------------


def test_database_is_opened_read_only(synthetic_db):
    with read_only_connection(synthetic_db) as con:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE should_not_exist (x INTEGER)")
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM equity")


def test_audit_never_modifies_the_input(tmp_path, synthetic_db):
    before_hash = sha256_file(synthetic_db)
    before_mtime = os.stat(synthetic_db).st_mtime_ns
    out = tmp_path / "report"
    assert main(["--input", str(synthetic_db), "--output-dir", str(out), "--quiet"]) == 0
    assert sha256_file(synthetic_db) == before_hash
    assert os.stat(synthetic_db).st_mtime_ns == before_mtime


# --------------------------------------------------------------------------
# missing tables / degraded inputs
# --------------------------------------------------------------------------


def test_missing_tables_degrade_gracefully(tmp_path):
    path = tmp_path / "sparse.db"
    _make_db(path, tables={"equity": [_equity_row(i, 100.0 + i) for i in range(3)]})
    with tempfile.TemporaryDirectory() as tmp:
        source = resolve_source(path, Path(tmp))
        analysis = run_analysis(source)
    assert analysis["equity"]["available"] is True
    assert analysis["execution"]["total_fills"] == 0
    assert analysis["positions"]["available"] is False
    assert analysis["shadow"]["available"] is False
    codes = {f["code"] for f in analysis["findings"]}
    assert "PA-SCHEMA-001" in codes
    missing = [f for f in analysis["findings"] if f["code"] == "PA-SCHEMA-001"]
    assert any(f["severity"] == "ERROR" for f in missing)  # fills is essential


def test_empty_database_does_not_crash(tmp_path):
    path = tmp_path / "empty.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE placeholder (x INTEGER)")
    con.execute("DROP TABLE placeholder")
    con.commit()
    con.close()
    with tempfile.TemporaryDirectory() as tmp:
        source = resolve_source(path, Path(tmp))
        analysis = run_analysis(source)
    assert analysis["equity"]["available"] is False
    assert analysis["findings_summary"]["total"] > 0


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_analysis_output_is_deterministic(tmp_path, synthetic_db):
    first = tmp_path / "a"
    second = tmp_path / "b"
    assert main(["--input", str(synthetic_db), "--output-dir", str(first), "--quiet"]) == 0
    assert main(["--input", str(synthetic_db), "--output-dir", str(second), "--quiet"]) == 0

    left = json.loads((first / "summary.json").read_text())
    right = json.loads((second / "summary.json").read_text())
    # the analytical payload must be byte-identical between runs...
    assert json.dumps(left["analysis"], sort_keys=True) == json.dumps(
        right["analysis"], sort_keys=True
    )
    # ...and the only varying data lives under `run`
    assert set(left["run"]) == set(right["run"])
    assert left["run"]["input_sha256"] == right["run"]["input_sha256"]
    assert (first / "daily_equity.csv").read_text() == (second / "daily_equity.csv").read_text()
    assert (first / "fill_attribution.csv").read_text() == (
        second / "fill_attribution.csv"
    ).read_text()

    # a timestamp must never leak into the analysis block
    assert "generated_at_utc" not in json.dumps(left["analysis"])


def test_run_metadata_carries_the_input_hash(synthetic_db):
    with tempfile.TemporaryDirectory() as tmp:
        source = resolve_source(synthetic_db, Path(tmp))
        run = build_run_metadata(source, "2026-01-01T00:00:00+00:00")
    assert run["input_sha256"] == sha256_file(synthetic_db)
    assert run["input_kind"] == "database"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_writes_every_output(tmp_path, synthetic_db):
    out = tmp_path / "report"
    assert main(["--input", str(synthetic_db), "--output-dir", str(out), "--quiet"]) == 0
    for name in ("report.md", "summary.json", "daily_equity.csv", "fill_attribution.csv"):
        assert (out / name).is_file(), name
    report = (out / "report.md").read_text()
    assert "# Kronos performance & data-quality audit" in report
    assert "Read this first" in report
    assert "NOT_EVALUABLE" in report
    assert "cannot conclude" in report


def test_cli_accepts_an_archive(tmp_path, synthetic_db):
    archive = tmp_path / "export.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(
            synthetic_db,
            arcname="kronos-performance-export/database/companion_sanitized.db",
        )
    out = tmp_path / "report"
    assert main(["--input", str(archive), "--output-dir", str(out), "--quiet"]) == 0
    payload = json.loads((out / "summary.json").read_text())
    assert payload["run"]["input_kind"] == "archive"
    assert payload["run"]["input_sha256"] == sha256_file(archive)


def test_cli_reports_a_missing_input(tmp_path, capsys):
    code = main(["--input", str(tmp_path / "nope.tar.gz"), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    assert "input not found" in capsys.readouterr().err


def test_cli_rejects_an_unsupported_input(tmp_path, capsys):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")
    code = main(["--input", str(junk), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    err = capsys.readouterr().err
    assert "unsupported input" in err
    assert ".tar.gz" in err  # tells the user what IS accepted


def test_cli_rejects_a_non_sqlite_database(tmp_path, capsys):
    fake = tmp_path / "fake.db"
    fake.write_bytes(b"not a database")
    code = main(["--input", str(fake), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    assert "not a SQLite database" in capsys.readouterr().err


def test_cli_rejects_an_archive_without_a_database(tmp_path, capsys):
    archive = tmp_path / "empty.tar.gz"
    note = tmp_path / "note.txt"
    note.write_text("x")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(note, arcname="export/note.txt")
    code = main(["--input", str(archive), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    assert "no sanitized SQLite database" in capsys.readouterr().err


def test_cli_rejects_a_truncated_database(tmp_path, capsys):
    """A zero-byte .db is a failed copy, not an empty database."""
    truncated = tmp_path / "truncated.db"
    truncated.write_bytes(b"")
    code = main(["--input", str(truncated), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    assert "not a SQLite database" in capsys.readouterr().err


def test_cli_rejects_a_directory_input(tmp_path, capsys):
    code = main(["--input", str(tmp_path), "--output-dir", str(tmp_path / "o")])
    assert code == 2
    assert "is a directory" in capsys.readouterr().err
