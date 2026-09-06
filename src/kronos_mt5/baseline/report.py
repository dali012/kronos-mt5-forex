"""Fixed-parameter development, chronological windows and a separately gated holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kronos_mt5.marketdata.manifest import save_manifest
from kronos_mt5.marketdata.pipeline import load_series, safe_output, validate_dataset
from kronos_mt5.walk_forward import generate_windows

from .config import BaselineConfig
from .engine import DAY_NS, run_engine
from .metrics import equity_metrics, summarize, trade_metrics
from .provenance import canonical_sha256, collect_provenance

LIMITATIONS = [
    "This replays the deployed dc8a74c decision, sizing and protective-order logic from verified frozen source snapshots. Execution remains explicitly approximate; this is not a claim of byte-identical live execution.",
    "Historical in-sample status before this PR is unknown: repository research has already used parts of 2023+. Chronological replay is not proof these dates were unseen during original strategy selection.",
    "Patient limits are approximated as next-open market orders paying conservative 5bps/side taker commission by default. No maker fill, queue position, cancellation latency or fallback drift reconstruction.",
    "Daily bars produce an assumed OHLC path: open, high at 08:00+1ns, low at 16:00+1ns, close at day-end-2ns (OLHC sensitivity supported). Stops fill at executable quotes after gaps, not ideal trigger prices.",
    "Intraday mark-based stops, five-second drawdown checks, exchange outages, maintenance tiers and liquidation/ADL are not reconstructable. Risk checks use synthetic quotes; explicit gross leverage rejection caps new exposure.",
    "Funding uses settled historical rates at event timestamps; archived rates lack mark prices, so those payments use the last causal synthetic quote. Funding veto uses the latest settled rate, not the live predicted rate.",
    "Current exchange filter snapshot is applied throughout history; historical filter/leverage-tier changes are unknown. No VIP/BNB discounts. Spread and slippage are configured assumptions, not measured order books.",
    "The universe is today's deployed survivors. No delisted assets or universe-selection reconstruction; survivor bias remains.",
    "Each reported replay/window starts flat after 253 closed warm-up bars and liquidates at its predeclared end with costs. Windows reset capital, risk and allocator state; aggregate development replay is reported independently.",
    "Gross PnL adds measured spread/slippage back to execution-price PnL; net equity already contains these costs and never subtracts them twice. Currency rounding residual is reported.",
    "CAGR and risk ratios from less than 365 days have low confidence. Undefined metrics are null, never invented as zero.",
]


def chronological_windows(
    start_ns: int, end_ns: int, days: int, holdout_ns: int
) -> list[tuple[int, int]]:
    if start_ns >= end_ns or days < 1 or end_ns > holdout_ns:
        raise ValueError("development windows must be chronological and exclude final holdout")
    origin = pd.Timestamp(start_ns - 253 * DAY_NS, unit="ns", tz="UTC").date().isoformat()
    end = pd.Timestamp(end_ns, unit="ns", tz="UTC").date().isoformat()
    windows = generate_windows(origin, end, train_days=253, test_days=days, step_days=days)
    result = [(pd.Timestamp(w.test_start).value, pd.Timestamp(w.test_end).value) for w in windows]
    tail = result[-1][1] if result else start_ns
    if tail < end_ns:
        result.append((tail, end_ns))
    return result


def freeze(
    path: Path,
    config: BaselineConfig,
    filters: dict,
    source: dict,
    dataset_manifest_sha256: str,
) -> None:
    # A filesystem lock records the exact pre-holdout configuration. It does not
    # claim to prevent a human from creating another directory and data snooping.
    payload = {
        "configuration_sha256": config.fingerprint(),
        "exchange_filter_snapshot_sha256": canonical_sha256(filters),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "deployed_bot_commit": source["deployed_bot_commit"],
        "deployed_strategy_blob_sha256": source["deployed_strategy_blob_sha256"],
        "research_implementation_commit": source["research_implementation_commit"],
        "research_adapter_sha256": source["research_adapter_sha256"],
        "holdout_start": config.holdout_start,
    }
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "frozen baseline differs: final holdout cannot select parameters or code"
            )
    else:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")


def references(frame: pd.DataFrame | None, start_ns: int, end_ns: int, equity: float) -> dict:
    result = {
        "cash": {"start_equity": equity, "end_equity": equity, "total_return": 0.0, "net_pnl": 0.0},
        "btc_buy_and_hold": None,
    }
    if frame is None:
        result["btc_reference_suppressed"] = "BTCUSDT not available in manifest"
        return result
    sample = frame[
        (frame.open_time * 1_000_000 >= start_ns) & (frame.open_time * 1_000_000 < end_ns)
    ]
    price = float(sample.iloc[0].open)
    points = [{"ts_ns": start_ns, "equity": equity}]
    points += [
        {"ts_ns": int(r.open_time) * 1_000_000 + DAY_NS, "equity": equity * float(r.close) / price}
        for r in sample.itertuples()
    ]
    result["btc_buy_and_hold"] = {
        **equity_metrics(points),
        "label": "Informational unlevered BTC price reference from futures candles; no costs or funding, not equivalent to strategy or spot total return",
    }
    return result


def run(
    manifest_path: Path,
    config: BaselineConfig,
    filters_payload: dict,
    output: Path,
    *,
    include_holdout: bool = False,
) -> dict:
    validation = validate_dataset(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if "1d" not in manifest["intervals"] or not set(config.symbols) <= set(manifest["symbols"]):
        raise ValueError("manifest does not cover configured daily universe")
    if config.funding_mode == "historical" and not manifest["funding"]:
        raise ValueError(
            "historical funding is required; omit only for explicitly incomplete smoke runs"
        )
    if filters_payload.get("schema_version") != 1 or not filters_payload.get("source"):
        raise ValueError("filters require a versioned provenance snapshot")
    filters = filters_payload["symbols"]
    if not set(config.symbols) <= set(filters):
        raise ValueError("missing exchange constraints")
    output = safe_output(output)
    source = collect_provenance()
    filters_sha256 = canonical_sha256(filters_payload)
    freeze(
        output / "baseline-lock.json",
        config,
        filters_payload,
        source,
        validation["manifest_sha256"],
    )
    # Daily history for eight symbols is small (~11k rows). Downloader/validator
    # remain bounded per partition; higher-frequency history is never loaded here.
    frames = {s: load_series(manifest_path, s, "1d") for s in config.symbols}
    funding = (
        {s: load_series(manifest_path, s, "funding") for s in config.symbols}
        if config.funding_mode == "historical"
        else {}
    )
    lower = int(manifest["start_ms"]) * 1_000_000 + 253 * DAY_NS
    upper = int(manifest["end_ms"]) * 1_000_000
    holdout = pd.Timestamp(config.holdout_start, tz="UTC").value
    dev_end = min(upper, holdout)
    if lower >= dev_end:
        raise ValueError("need 253 warm-up days and development data before 2026-01-01")
    btc = frames.get("BTCUSDT")
    if btc is None and "BTCUSDT" in manifest["symbols"]:
        btc = load_series(manifest_path, "BTCUSDT", "1d")

    def evaluate(start, end):
        replay = run_engine(frames, funding, filters, config, start, end)
        metrics = summarize(replay)
        for symbol in config.symbols:
            metrics["per_symbol"].setdefault(
                symbol,
                {
                    **trade_metrics([]),
                    "costs": {k: 0.0 for k in ("commission", "spread", "slippage", "funding")},
                    "turnover_notional": 0.0,
                },
            )
        metrics["start_utc"] = pd.Timestamp(start, unit="ns", tz="UTC").isoformat()
        metrics["end_utc_exclusive"] = pd.Timestamp(end, unit="ns", tz="UTC").isoformat()
        metrics["references"] = references(btc, start, end, config.starting_equity)
        return metrics, replay

    development, dev_replay = evaluate(lower, dev_end)
    windows = []
    for start, end in chronological_windows(lower, dev_end, config.window_days, holdout):
        metrics, _ = evaluate(start, end)
        windows.append(
            {
                "warmup_start_utc": pd.Timestamp(
                    start - 253 * DAY_NS, unit="ns", tz="UTC"
                ).isoformat(),
                "training": "none: fixed deployed parameters",
                **metrics,
            }
        )
    holdout_result = {
        "status": "UNTOUCHED",
        "reason": "requires explicit --include-holdout after configuration freeze",
    }
    holdout_replay = None
    if include_holdout and upper > holdout:
        if int(manifest["start_ms"]) * 1_000_000 > holdout - 253 * DAY_NS:
            raise ValueError("insufficient past warm-up for the fixed final holdout")
        holdout_result, holdout_replay = evaluate(holdout, upper)
        holdout_result["status"] = "EVALUATED_FIXED_BASELINE_ONLY"
    elif upper <= holdout:
        holdout_result = {"status": "UNAVAILABLE", "reason": "dataset ends before final holdout"}
    report = {
        "schema_version": 1,
        **source,
        "provenance": source,
        "dataset_manifest": manifest,
        "dataset_manifest_path": str(manifest_path.resolve()),
        "dataset_manifest_sha256": validation["manifest_sha256"],
        "validation": validation,
        "configuration": config.payload(),
        "configuration_sha256": config.fingerprint(),
        "strategy_configuration_fingerprint": config.fingerprint(),
        "exchange_filters": filters_payload,
        "exchange_filter_snapshot_sha256": filters_sha256,
        "limitations": LIMITATIONS,
        "universe_scope": "full_deployed" if len(config.symbols) == 8 else "subset_smoke_only",
        "completeness": "APPROXIMATE_BASELINE"
        if config.funding_mode == "historical"
        else "INCOMPLETE_FUNDING_OMITTED",
        "data_gaps": [],
        "development": development,
        "walk_forward": windows,
        "holdout": holdout_result,
    }
    save_manifest(output / "configuration.json", config.payload())
    save_manifest(output / "report.json", clean(report))
    save_manifest(
        output / "development-replay.json",
        clean({k: v for k, v in dev_replay.items() if k != "signal_history"}),
    )
    if holdout_replay:
        save_manifest(
            output / "holdout-replay.json",
            clean({k: v for k, v in holdout_replay.items() if k != "signal_history"}),
        )
    (output / "report.md").write_text(render(report))
    return clean(report)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not pd.notna(value):
        return None
    return value


def render(report: dict) -> str:
    lines = [
        "# Fixed-strategy historical baseline",
        "",
        f"Status: **{report['completeness']}**.",
        "",
        f"Deployed bot commit: `{report['deployed_bot_commit']}`.",
        f"Deployed strategy SHA-256: `{report['deployed_strategy_blob_sha256']}`; snapshot verified: `{report['deployed_snapshot_verification_passed']}`.",
        f"Research implementation commit: `{report['research_implementation_commit']}`.",
        f"Manifest SHA-256: `{report['validation']['manifest_sha256']}`.",
        f"Configuration SHA-256: `{report['configuration_sha256']}`.",
        "",
        "Full manifest, configuration, source hashes, costs, attribution, references and suppressed metrics are in `report.json`.",
        "",
    ]
    for name in ("development", "holdout"):
        metrics = report[name]
        lines += [f"## {name.title()}", ""]
        if "start_equity" not in metrics:
            lines += [f"{metrics['status']}: {metrics.get('reason', '')}", ""]
            continue
        lines += [
            f"{metrics['start_utc']} to {metrics['end_utc_exclusive']} (exclusive).",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        for key in (
            "start_equity",
            "end_equity",
            "total_return",
            "cagr",
            "annualized_low_confidence",
            "max_drawdown",
            "max_drawdown_duration_days",
            "sharpe",
            "sortino",
            "calmar",
            "profit_factor",
            "win_rate",
            "expectancy_per_trade",
            "trades",
            "average_holding_hours",
            "turnover_notional",
            "gross_pnl",
            "net_pnl",
            "accounting_residual",
            "funding_mark_approximations",
        ):
            lines.append(f"| {key} | {metrics.get(key)} |")
        for key, value in metrics["costs"].items():
            lines.append(f"| {key} cost (positive paid) | {value} |")
        lines += [
            "",
            f"Suppressed: {', '.join(metrics['suppressed_metrics']) or 'none'}.",
            "",
            f"Cash return: 0%; BTC informational return: {metrics['references']['btc_buy_and_hold']['total_return'] if metrics['references']['btc_buy_and_hold'] else 'unavailable'}.",
            "",
        ]
        lines += [
            "### Attribution",
            "",
            "| Symbol | Trades | Net PnL | Commission | Spread | Slippage | Funding |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for symbol, values in metrics["per_symbol"].items():
            costs = values["costs"]
            lines.append(
                f"| {symbol} | {values['trades']} | {values['net_pnl']:.4f} | {costs['commission']:.4f} | {costs['spread']:.4f} | {costs['slippage']:.4f} | {costs['funding']:.4f} |"
            )
        lines += [
            "",
            "| Direction | Trades | Net PnL | Win rate | Expectancy |",
            "|---|---:|---:|---:|---:|",
        ]
        for side, values in metrics["long_short"].items():
            lines.append(
                f"| {side} | {values['trades']} | {values['net_pnl']:.4f} | {values['win_rate']} | {values['expectancy_per_trade']} |"
            )
        lines += ["", "| Exposure statistic | Value |", "|---|---:|"]
        lines += [f"| {key} | {value} |" for key, value in metrics["exposure"].items()]
        for category in ("monthly", "yearly", "market_regime"):
            lines += [
                "",
                f"### {category.replace('_', ' ').title()}",
                "",
                "| Period/regime | Net PnL | Return |",
                "|---|---:|---:|",
            ]
            for key, values in metrics[category].items():
                lines.append(
                    f"| {key} | {values['net_pnl']:.4f} | {values.get('return', 'not annualized')} |"
                )
        lines += [""]
    lines += [
        "## Chronological windows",
        "",
        "| Start | End exclusive | Return | Trades | Sharpe |",
        "|---|---|---:|---:|---:|",
    ]
    for w in report["walk_forward"]:
        lines.append(
            f"| {w['start_utc'][:10]} | {w['end_utc_exclusive'][:10]} | {w['total_return']:.6f} | {w['trades']} | {w['sharpe']} |"
        )
    lines += ["", "## Assumptions and limitations", ""]
    lines += [f"- {note}" for note in report["limitations"]]
    lines += [
        "",
        "The final holdout must remain unavailable to subsequent strategy/parameter selection. Once examined it is not a fresh holdout for future ideas.",
        "",
    ]
    return "\n".join(lines)
