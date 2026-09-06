"""Leakage, accounting, exchange constraints and deterministic Nautilus regression tests."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import replace

import pandas as pd
import pytest

from kronos_mt5.baseline.config import BaselineConfig
from kronos_mt5.baseline.engine import DAY_NS, friction, funding_payment, run_engine
from kronos_mt5.baseline.instruments import instrument, reject_reason, rounded
from kronos_mt5.baseline.metrics import equity_metrics, round_trips, summarize
from kronos_mt5.baseline.report import chronological_windows, freeze
from kronos_mt5.baseline.synthetic import fixture


@pytest.fixture(scope="module")
def replay():
    frames, funding, filters, config = fixture()
    origin = int(frames["BTCUSDT"].iloc[0].open_time) * 1_000_000
    result = run_engine(
        frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    return frames, funding, filters, config, origin, result


def test_repeatability_warmup_flat_boundaries(replay):
    frames, funding, filters, config, origin, result = replay
    repeat = run_engine(
        frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    assert result == repeat
    assert result["observations"][0]["equity"] == config.starting_equity
    assert result["observations"][0]["ts_ns"] == origin + 253 * DAY_NS
    assert result["observations"][-1]["n_open"] == 0
    assert min(f["ts_ns"] for f in result["fills"]) >= origin + 253 * DAY_NS
    metrics = summarize(result)
    assert metrics["elapsed_days"] == 47
    assert abs(metrics["accounting_residual"]) < 1e-5
    assert metrics["net_pnl"] == pytest.approx(
        metrics["gross_pnl"] - sum(metrics["costs"].values())
    )
    assert sum(m["net_pnl"] for m in metrics["monthly"].values()) == pytest.approx(
        metrics["net_pnl"]
    )


def test_future_perturbation_cannot_change_past_indicators_or_fills(replay):
    frames, funding, filters, config, origin, original = replay
    changed = copy.deepcopy(frames)
    shock = origin + 280 * DAY_NS
    frame = changed["BTCUSDT"]
    frame.loc[frame.open_time * 1_000_000 >= shock, ["open", "high", "low", "close"]] *= 2
    later = run_engine(
        changed, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    for key in ("decisions", "fills"):
        assert [r for r in original[key] if r["ts_ns"] < shock] == [
            r for r in later[key] if r["ts_ns"] < shock
        ]
    for history in original["signal_history"].values():
        for observation in history:
            last = (observation["ts_ns"] + 1 - DAY_NS) // 1_000_000
            available = (
                frames["BTCUSDT"].loc[frames["BTCUSDT"].open_time <= last, "close"].tail(253)
            )
            assert observation["closes"] == pytest.approx(
                available.tolist(), rel=1e-4
            )  # instrument tick rounding


def test_next_bar_open_execution_not_same_close(replay):
    frames, _, filters, config, _, result = replay
    first = result["fills"][0]
    prior = [d for d in result["decisions"] if d["ts_ns"] < first["ts_ns"]][-1]
    assert first["ts_ns"] % DAY_NS == 0
    assert prior["ts_ns"] == first["ts_ns"] - 1
    candle = (
        frames["BTCUSDT"].loc[frames["BTCUSDT"].open_time == first["ts_ns"] // 1_000_000].iloc[0]
    )
    expected = rounded(
        candle.open * (1 + (config.half_spread_bps + config.slippage_bps) / 1e4),
        filters["BTCUSDT"]["tick_size"],
        up=True,
    )
    assert first["price"] == expected
    assert first["price"] != pytest.approx(prior["reference_price"])
    assert first["commission"] == pytest.approx(
        first["qty"] * first["price"] * config.commission_bps / 1e4
    )


def test_funding_is_signed_causal_and_in_account(replay):
    frames, funding, filters, config, origin, result = replay
    assert funding_payment(2, 100, 0.001) == -0.2
    assert funding_payment(-2, 100, 0.001) == 0.2
    assert funding_payment(2, 100, -0.001) == 0.2
    assert friction(10000, 5, 1, 2) == {"commission": 5, "spread": 1, "slippage": 2}
    assert all(
        p["units"] == 0
        or p["cash_flow"] == pytest.approx(-p["units"] * p["valuation_price"] * p["funding_rate"])
        for p in result["funding"]
    )
    shock = origin + 280 * DAY_NS
    changed = copy.deepcopy(funding)
    changed["BTCUSDT"].loc[changed["BTCUSDT"].funding_time * 1_000_000 >= shock, "funding_rate"] = (
        0.1
    )
    late = run_engine(
        frames, changed, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    assert [d for d in late["decisions"] if d["ts_ns"] < shock] == [
        d for d in result["decisions"] if d["ts_ns"] < shock
    ]
    assert late["observations"][-1]["equity"] < result["observations"][-1]["equity"]


def test_precision_minimum_notional_and_perpetual():
    _, _, filters, config = fixture()
    f = filters["BTCUSDT"]
    assert reject_reason(0.0001, 100, f) == "quantity_precision"
    assert reject_reason(0.01, 100.001, f) == "price_precision"
    assert reject_reason(0.001, 100, f) == "minimum_notional"
    assert reject_reason(0.001, 100, f, reduce_only=True) is None
    assert reject_reason(100001, 100, f) == "quantity_limits"
    assert rounded(1.2349, "0.001") == 1.234
    f = {**f, "tick_size": "0.01000000", "step_size": "0.00100000"}
    inst = instrument("BTCUSDT", f, config.commission_bps, config.leverage)
    assert inst.size_precision == 3 and inst.price_precision == 2
    assert inst.is_inverse is False


def test_engine_rejects_exchange_minimum_and_warmup(replay):
    frames, funding, filters, config, origin, _ = replay
    filters = copy.deepcopy(filters)
    filters["BTCUSDT"]["min_notional"] = "10000000"
    r = run_engine(frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS)
    assert not r["fills"]  # production min-notional buffer suppresses the intent before submission
    with pytest.raises(ValueError, match="warm-up"):
        run_engine(frames, funding, filters, config, origin + 200 * DAY_NS, origin + 300 * DAY_NS)


def test_chronological_windows_and_untouched_holdout(tmp_path):
    holdout = pd.Timestamp("2026-01-01", tz="UTC").value
    windows = chronological_windows(holdout - 200 * DAY_NS, holdout, 90, holdout)
    assert len(windows) == 3
    assert all(a < b <= holdout for a, b in windows)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(windows))
    with pytest.raises(ValueError):
        chronological_windows(holdout, holdout + DAY_NS, 90, holdout)
    config = BaselineConfig()
    source = {"source_sha256": {"strategy.py": "fixed"}}
    path = tmp_path / "lock.json"
    freeze(path, config, {}, source)
    freeze(path, config, {}, source)
    with pytest.raises(ValueError, match="holdout"):
        freeze(path, replace(config, commission_bps=6), {}, source)
    with pytest.raises(ValueError, match="frozen"):
        replace(config, holdout_start="2027-01-01")
    payload = config.payload()
    payload["strategy"]["target_vol"] = 0.2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        BaselineConfig.read(path)


def test_known_equity_and_trade_metrics():
    values = [100, 110, 99, 120]
    points = [{"ts_ns": i * DAY_NS, "equity": v} for i, v in enumerate(values)]
    r = equity_metrics(points)
    assert r["total_return"] == pytest.approx(0.2)
    assert r["max_drawdown"] == pytest.approx(-0.1)
    assert r["max_drawdown_duration_days"] == 2
    assert r["daily_return_count"] == 3
    assert r["annualized_low_confidence"]
    cash = equity_metrics([{"ts_ns": 0, "equity": 100}, {"ts_ns": DAY_NS, "equity": 100}])
    assert cash["sharpe"] is None and cash["sortino"] is None
    fills = [
        {
            "symbol": "BTCUSDT",
            "ts_ns": 0,
            "side": "BUY",
            "qty": 2,
            "price": 100,
            "commission": 1,
            "spread": 0.2,
            "slippage": 0.4,
        },
        {
            "symbol": "BTCUSDT",
            "ts_ns": DAY_NS,
            "side": "SELL",
            "qty": 3,
            "price": 110,
            "commission": 1.5,
            "spread": 0.3,
            "slippage": 0.6,
        },
        {
            "symbol": "BTCUSDT",
            "ts_ns": 2 * DAY_NS,
            "side": "BUY",
            "qty": 1,
            "price": 105,
            "commission": 0.5,
            "spread": 0.1,
            "slippage": 0.2,
        },
    ]
    trades = round_trips(fills, [{"symbol": "BTCUSDT", "ts_ns": DAY_NS // 2, "cash_flow": -2}])
    assert len(trades) == 2
    assert trades[0]["net_pnl"] == 16
    assert trades[1]["net_pnl"] == 4
    assert trades[1]["direction"] == "short"


def synthetic_manifest(root, frames, funding):
    from kronos_mt5.marketdata.manifest import save_manifest
    from kronos_mt5.marketdata.pipeline import store_entry
    from kronos_mt5.marketdata.store import StoreLayout

    frame = frames["BTCUSDT"]
    start, end = int(frame.iloc[0].open_time), int(frame.iloc[-1].open_time) + 86_400_000
    entries = []
    for kind, interval, rows in (
        ("klines", "1d", frame.to_dict("records")),
        ("fundingRate", "funding", funding["BTCUSDT"].to_dict("records")),
    ):
        entries.append(
            store_entry(
                root,
                StoreLayout(root),
                "BTCUSDT",
                interval,
                kind,
                "fixture",
                rows,
                start,
                end,
                [{"url": "bundled:synthetic", "checksum_verified": False}],
            )
        )
    path = root / "manifests/dataset-fixture.json"
    save_manifest(
        path,
        {
            "schema_version": 1,
            "symbols": ["BTCUSDT"],
            "intervals": ["1d"],
            "start_ms": start,
            "end_ms": end,
            "funding": True,
            "partitions": entries,
        },
    )
    return path


def test_report_and_reproduce_without_network(tmp_path, replay, capsys):
    from kronos_mt5.baseline.__main__ import main
    from kronos_mt5.baseline.report import run

    frames, funding, filters, config, _, _ = replay
    manifest = synthetic_manifest(tmp_path / "data", frames, funding)
    constraints = {"schema_version": 1, "source": "synthetic fixture", "symbols": filters}
    output = tmp_path / "baseline"
    report = run(manifest, config, constraints, output)
    assert report["holdout"]["status"] == "UNAVAILABLE"
    assert report["development"]["trades"] > 0
    assert report["dataset_manifest"]["partitions"]
    assert "Attribution" in (output / "report.md").read_text()
    assert (
        main(
            [
                "reproduce",
                "--report",
                str(output / "report.json"),
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "reproduced"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["identical_metrics"]


def test_default_run_never_evaluates_holdout(tmp_path, monkeypatch):
    from kronos_mt5.baseline import report as module

    frames, funding, filters, config = fixture(days=410)
    # Move the same deterministic synthetic sample forward one year, crossing holdout.
    for key in ("open_time", "close_time"):
        frames["BTCUSDT"][key] += 366 * 86_400_000
    funding["BTCUSDT"]["funding_time"] += 366 * 86_400_000
    manifest = synthetic_manifest(tmp_path / "data", frames, funding)
    seen = []

    def engine(frames, funding, filters, config, start, end):
        seen.append((start, end))
        return {
            "observations": [
                {
                    "ts_ns": t,
                    "equity": 100000.0,
                    "n_open": 0,
                    "gross_exposure": 0,
                    "regime": "UNKNOWN",
                }
                for t in range(start, end + 1, DAY_NS)
            ],
            "fills": [],
            "funding": [],
            "rejections": [],
            "start_ns": start,
            "end_ns": end,
        }

    monkeypatch.setattr(module, "run_engine", engine)
    constraints = {"schema_version": 1, "source": "synthetic fixture", "symbols": filters}
    output = tmp_path / "baseline"
    report = module.run(manifest, config, constraints, output)
    holdout = pd.Timestamp("2026-01-01", tz="UTC").value
    assert report["holdout"]["status"] == "UNTOUCHED"
    assert all(end <= holdout for start, end in seen)
    seen.clear()
    report = module.run(manifest, config, constraints, output, include_holdout=True)
    assert report["holdout"]["status"] == "EVALUATED_FIXED_BASELINE_ONLY"
    assert any(start == holdout and end > holdout for start, end in seen)


def test_gap_stop_uses_executable_bid_and_current_quote_for_costs(replay):
    frames, funding, filters, config, origin, _ = replay
    frames = copy.deepcopy(frames)
    frames["BTCUSDT"].loc[270:, ["open", "high", "low", "close"]] *= 0.5
    result = run_engine(
        frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    gap = origin + 270 * DAY_NS
    stop = next(f for f in result["fills"] if f["ts_ns"] >= gap)
    new_open = float(frames["BTCUSDT"].iloc[270].open)
    assert stop["ts_ns"] == gap and stop["side"] == "SELL"
    assert stop["price"] <= new_open  # no idealized fill at the old stop trigger
    assert stop["mid"] == pytest.approx(new_open, abs=0.01)
    assert stop["spread"] + stop["slippage"] < stop["notional"] * 0.001
    assert summarize(result)["gross_pnl"] < -30000  # market gap remains price PnL


def test_portfolio_repeatability_and_leverage_rejection(replay):
    frames, funding, filters, config, origin, _ = replay
    from kronos_mt5.marketdata.spec import PRODUCTION_UNIVERSE

    universe = PRODUCTION_UNIVERSE
    f = {s: frames["BTCUSDT"].copy() for s in universe}
    rates = {s: funding["BTCUSDT"].copy() for s in universe}
    constraints = {s: dict(filters["BTCUSDT"]) for s in universe}
    portfolio = replace(config, symbols=universe)
    a = run_engine(f, rates, constraints, portfolio, origin + 253 * DAY_NS, origin + 300 * DAY_NS)
    b = run_engine(f, rates, constraints, portfolio, origin + 253 * DAY_NS, origin + 300 * DAY_NS)
    assert summarize(a) == summarize(b)
    assert summarize(a)["exposure"]["max_concurrent_positions"] == 8
    capped = run_engine(
        frames,
        funding,
        filters,
        replace(config, leverage=1),
        origin + 253 * DAY_NS,
        origin + 300 * DAY_NS,
    )
    assert any(r["reason"] == "leverage_limit" for r in capped["rejections"])
