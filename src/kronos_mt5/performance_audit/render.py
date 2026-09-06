"""Output rendering: report.md, summary.json, daily_equity.csv, fill_attribution.csv.

The `analysis` block is deterministic: given the same input bytes it renders
byte-identically. Anything that varies per run (wall-clock time, input path)
lives under `run` in summary.json and in a clearly separated section of the
report, never inside an analytical result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kronos_mt5.performance_audit.version import SCHEMA_VERSION, TOOL_VERSION

CSV_DAILY_FIELDS = (
    "date",
    "ts",
    "equity",
    "cash",
    "unrealized",
    "n_open",
    "mean_n_open",
    "max_n_open",
    "observations",
    "return",
)
CSV_FILL_FIELDS = (
    "symbol",
    "side",
    "kind",
    "fills",
    "quantity",
    "quote_turnover",
    "commission",
    "slippage_quote",
    "reconciliation_fills",
    "missing_kind",
    "missing_reference_price",
    "missing_slippage",
    "missing_commission",
    "missing_trade_id",
    "avg_shortfall_bps",
    "weighted_shortfall_bps",
    "median_shortfall_bps",
    "shortfall_sample",
)


def _num(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{suffix}"
    return str(value)


def write_summary_json(path: Path, analysis: dict, run: dict) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        # `run` holds everything non-deterministic; `analysis` is reproducible.
        "run": run,
        "analysis": analysis,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_daily_csv(path: Path, daily: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_DAILY_FIELDS))
        writer.writeheader()
        for row in daily:
            writer.writerow({key: row.get(key) for key in CSV_DAILY_FIELDS})


def write_fill_csv(path: Path, groups: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FILL_FIELDS))
        writer.writeheader()
        for row in groups:
            writer.writerow({key: row.get(key) for key in CSV_FILL_FIELDS})


def _reconciliation_verdict(accounting: dict) -> str:
    """`yes` / `NO` / `UNRELIABLE` — never a verdict built on provisional totals."""
    within = accounting.get("reconciled_within_tolerance")
    if within is None:
        return "**UNRELIABLE (validity withheld)**"
    return "yes" if within else "NO"


def _findings_table(items: list[dict]) -> list[str]:
    if not items:
        return ["No findings.", ""]
    lines = ["| Severity | Code | Finding | Scope |", "|---|---|---|---|"]
    for item in items:
        lines.append(
            f"| {item['severity']} | `{item['code']}` | {item['title']} | `{item['scope']}` |"
        )
    lines.append("")
    return lines


def render_report(analysis: dict, run: dict) -> str:
    equity = analysis.get("equity", {})
    accounting = analysis.get("accounting", {})
    execution = analysis.get("execution", {})
    positions = analysis.get("positions", {})
    shadow = analysis.get("shadow", {})
    operations = analysis.get("operations", {})
    regimes = analysis.get("regimes", {})
    context = analysis.get("context", {})
    findings = analysis.get("findings", [])
    schema = analysis.get("schema", {})

    lines: list[str] = []
    add = lines.append

    add("# Kronos performance & data-quality audit")
    add("")
    add("Offline, read-only forensics over a sanitized companion export. This audit")
    add("recomputes every figure from raw rows rather than trusting what the bot")
    add("recorded, and reports what the data cannot support as loudly as what it can.")
    add("")

    # --- read this first -------------------------------------------------
    prominent = [f for f in findings if f.get("prominent")]
    add("## Read this first")
    add("")
    if prominent:
        for item in prominent:
            add(f"- **{item['severity']} `{item['code']}` — {item['title']}.** {item['impact']}")
    else:
        add("- No blocking interpretation caveats were raised.")
    add("")
    add("> This tool does not claim, and cannot establish, that the strategy is")
    add("> profitable. It measures what happened in the recorded sample.")
    add("")

    # --- run metadata ----------------------------------------------------
    add("## Run metadata")
    add("")
    add("*(Non-deterministic; excluded from the analytical results.)*")
    add("")
    add(f"- Generated (UTC): `{run.get('generated_at_utc')}`")
    add(
        f"- Tool version: `{run.get('tool_version')}` (summary schema `{run.get('schema_version')}`)"
    )
    add(f"- Input: `{run.get('input_name')}` ({run.get('input_kind')})")
    add(f"- Input SHA-256: `{run.get('input_sha256')}`")
    if run.get("database_name"):
        add(f"- Database: `{run.get('database_name')}`")
    add("")

    # --- source ----------------------------------------------------------
    add("## Source and configuration")
    add("")
    source = context.get("source") or {}
    if source:
        for key in sorted(source):
            add(f"- {key.replace('_', ' ')}: `{source[key]}`")
    else:
        add("- No export provenance available (database supplied directly).")
    if context.get("provenance_known"):
        add(
            f"- Environment: `{context.get('binance_environment')}`, "
            f"DEMO_ONLY=`{context.get('demo_only')}`"
        )
    else:
        add(
            "- Environment: **UNKNOWN — provenance could not be established.** The "
            "audit does not guess the environment from table contents. Missing: "
            f"`{context.get('missing_provenance')}`"
        )
    add(f"- Symbols traded: {', '.join(context.get('symbols') or []) or 'n/a'}")
    add(f"- OHLCV available: **{'yes' if context.get('ohlcv_available') else 'no'}**")
    add("")
    add("### Tables")
    add("")
    add("| Table | Rows |")
    add("|---|---:|")
    for name, info in (schema.get("tables") or {}).items():
        add(f"| `{name}` | {info.get('rows')} |")
    if schema.get("missing_known_tables"):
        add("")
        add(f"Missing expected tables: {', '.join(schema['missing_known_tables'])}")
    add("")

    # --- equity ----------------------------------------------------------
    add("## Equity performance")
    add("")
    if not equity.get("available"):
        add("No usable equity observations.")
        add("")
    else:
        add(
            f"Daily series built from the **last valid observation of each UTC day** "
            f"({equity.get('raw_observations'):,} raw observations -> "
            f"{equity.get('days_observed')} daily points)."
        )
        add("")
        add("| Metric | Value |")
        add("|---|---:|")
        add(f"| Window | {equity.get('first_day')} .. {equity.get('last_day')} |")
        add(
            f"| Calendar days covered | {equity.get('calendar_days_covered')} "
            f"(inclusive dates) |"
        )
        add(f"| Elapsed days | {equity.get('elapsed_days')} (CAGR basis) |")
        add(f"| Days observed | {equity.get('days_observed')} |")
        add(
            f"| Return periods | {equity.get('return_periods')} "
            f"({equity.get('ratio_return_periods')} used for ratios) |"
        )
        add(f"| Starting equity | {_num(equity.get('start_equity'), 4)} |")
        add(f"| Ending equity | {_num(equity.get('end_equity'), 4)} |")
        add(f"| Minimum equity | {_num(equity.get('min_equity'), 4)} |")
        add(f"| Maximum equity | {_num(equity.get('max_equity'), 4)} |")
        add(f"| Absolute return | {_num(equity.get('absolute_return'), 4)} |")
        add(f"| Total return | {_num(equity.get('total_return_pct'), 4, '%')} |")
        add(
            f"| Annualized return (CAGR) | "
            f"{_num(equity.get('annualized_return_pct'), 4, '%')} |"
        )
        add(f"| Annualized volatility | {_num(equity.get('annualized_volatility_pct'), 2, '%')} |")
        add(f"| Sharpe (rf=0) | {_num(equity.get('sharpe_ratio'), 3)} |")
        add(f"| Sortino | {_num(equity.get('sortino_ratio'), 3)} |")
        add(f"| Calmar | {_num(equity.get('calmar_ratio'), 3)} |")
        add(f"| Max drawdown | {_num(equity.get('drawdown_max_drawdown_pct'), 3, '%')} |")
        add(
            f"| Positive / negative / flat days | {equity.get('positive_days')} / "
            f"{equity.get('negative_days')} / {equity.get('flat_days')} |"
        )
        add(f"| Hit rate | {_num(equity.get('hit_rate_pct'), 2, '%')} |")
        best, worst = equity.get("best_day") or {}, equity.get("worst_day") or {}
        add(f"| Best day | {best.get('date')} ({_num(best.get('return_pct'), 3, '%')}) |")
        add(f"| Worst day | {worst.get('date')} ({_num(worst.get('return_pct'), 3, '%')}) |")
        add("")
        if equity.get("ratios_note"):
            add(f"> **Ratio basis:** {equity['ratios_note']}")
            add("")
        if not equity.get("ratios_available", True):
            add("> Volatility, Sharpe and Sortino are reported as **unavailable**.")
            add("")
        if equity.get("annualized_low_confidence"):
            add(f"> **Low confidence:** {equity.get('annualized_confidence_note')}")
            add("")
        add("### Drawdown")
        add("")
        add(
            f"- Peak `{equity.get('drawdown_peak_date')}` at "
            f"{_num(equity.get('drawdown_peak_equity'), 4)}"
        )
        add(
            f"- Trough `{equity.get('drawdown_trough_date')}` at "
            f"{_num(equity.get('drawdown_trough_equity'), 4)}"
        )
        add(
            f"- Depth {_num(equity.get('drawdown_max_drawdown_pct'), 3, '%')} over "
            f"{equity.get('drawdown_drawdown_days')} days"
        )
        recovered = equity.get("drawdown_recovered")
        add(
            f"- Recovered: **{'yes, ' + str(equity.get('drawdown_recovery_date')) if recovered else 'no'}**"
        )
        add("")
        exposure = equity.get("exposure") or {}
        add("### Exposure")
        add("")
        add(f"- Mean open positions per day: {_num(exposure.get('mean_positions'), 2)}")
        add(f"- Max open positions: {_num(exposure.get('max_positions'), 0)}")
        add(
            f"- Days with a flat book: {exposure.get('days_with_flat_book')} "
            f"({_num(exposure.get('days_with_flat_book_pct'), 1, '%')})"
        )
        add("")
        raw = equity.get("observation_series") or {}
        if raw:
            add("### Raw observation series (for reconciliation)")
            add("")
            add(f"*{equity.get('daily_series_convention')}*")
            add("")
            add("| Metric | Daily series | Raw observations |")
            add("|---|---:|---:|")
            add(
                f"| Start / first | {_num(equity.get('start_equity'), 4)} | "
                f"{_num(raw.get('first_equity'), 4)} |"
            )
            add(
                f"| End / last | {_num(equity.get('end_equity'), 4)} | "
                f"{_num(raw.get('last_equity'), 4)} |"
            )
            add(
                f"| Minimum | {_num(equity.get('min_equity'), 4)} | "
                f"{_num(raw.get('min_equity'), 4)} |"
            )
            add(
                f"| Maximum | {_num(equity.get('max_equity'), 4)} | "
                f"{_num(raw.get('max_equity'), 4)} |"
            )
            add(
                f"| Return | {_num(equity.get('total_return_pct'), 4, '%')} | "
                f"{_num(raw.get('period_return_pct'), 4, '%')} |"
            )
            add(
                f"| Max drawdown | {_num(equity.get('drawdown_max_drawdown_pct'), 3, '%')} | "
                f"{_num(raw.get('max_drawdown_pct'), 3, '%')} |"
            )
            add("")

        add("### Coverage")
        add("")
        add(f"- Missing calendar days: {equity.get('missing_day_count')}")
        add(f"- Observation gaps > 1h: {equity.get('observation_gap_count')}")
        add(f"- Duplicate timestamps: {equity.get('duplicate_timestamps')}")
        add(f"- Invalid equity rows: {equity.get('invalid_equity_rows')}")
        add("")

    # --- accounting ------------------------------------------------------
    add("## PnL reconciliation")
    add("")
    if not accounting.get("available"):
        add("Not available.")
        add("")
    else:
        add("Identity checked (the companion's own):")
        add("")
        add("```")
        add("equity_change = realized - commissions + funding + other_income")
        add("                + unrealized_change + residual")
        add("```")
        add("")
        add("| Component | Value |")
        add("|---|---:|")
        add(f"| Equity change | {_num(accounting.get('equity_change'), 4)} |")
        add(f"| Cash change | {_num(accounting.get('cash_change'), 4)} |")
        add(f"| Unrealized change | {_num(accounting.get('unrealized_change'), 4)} |")
        add(f"| Realized PnL (income) | {_num(accounting.get('realized_pnl'), 4)} |")
        add(f"| Commissions (cost) | {_num(accounting.get('commissions_cost'), 4)} |")
        add(f"| Funding (signed) | {_num(accounting.get('funding'), 4)} |")
        add(f"| Other income | {_num(accounting.get('other_income'), 4)} |")
        add(f"| **Explained** | {_num(accounting.get('explained_change'), 4)} |")
        add(f"| **Residual** | {_num(accounting.get('residual'), 4)} |")
        add(f"| Tolerance | {_num(accounting.get('residual_tolerance'), 4)} |")
        add(f"| Reconciled within tolerance | " f"{_reconciliation_verdict(accounting)} |")
        add("")
        window = accounting.get("window") or {}
        income_window = accounting.get("income_window") or {}
        fills_window = accounting.get("fills_window") or {}
        if window:
            add("### Reconciliation window")
            add("")
            add(f"`{window.get('rule')}`")
            add("")
            add(f"- Start (exclusive): `{window.get('start_exclusive')}`")
            add(f"- End (inclusive): `{window.get('end_inclusive')}`")
            add("")
            add("| Ledger | Included | Before | After | Bad timestamp |")
            add("|---|---:|---:|---:|---:|")
            for label, block in (("income", income_window), ("fills", fills_window)):
                add(
                    f"| {label} | {block.get('included')} | "
                    f"{block.get('before_window')} | {block.get('after_window')} | "
                    f"{block.get('invalid_timestamp')} |"
                )
            add("")
            excluded = income_window.get("excluded_amounts_by_type") or {}
            if any(excluded.values()):
                add("Excluded income amounts by type:")
                add("")
                for bucket, amounts in excluded.items():
                    if amounts:
                        add(f"- `{bucket}`: {amounts}")
                add("")
            excluded_commission = fills_window.get("excluded_commission") or {}
            if any(excluded_commission.values()):
                add(f"- Excluded fill commission: `{excluded_commission}`")
                add(
                    f"- Excluded fill slippage (quote): "
                    f"`{fills_window.get('excluded_slippage_quote')}`"
                )
                add("")
            for label, block in (("income", income_window), ("fills", fills_window)):
                dedupe = block.get("deduplication") or {}
                if dedupe.get("exact_duplicates") or dedupe.get("conflicting_id_count"):
                    add(
                        f"- `{label}` de-duplication: "
                        f"{dedupe.get('exact_duplicates')} exact duplicate(s), "
                        f"{dedupe.get('conflicting_id_count')} conflicting id(s)"
                    )
            if not accounting.get("reconciliation_reliable", True):
                add("")
                add(
                    f"> **Reconciliation UNRELIABLE — totals above are provisional.** "
                    f"{accounting.get('reconciliation_unreliable_reason')}"
                )
            add("")
        add(
            "Recorded slippage (measurement only, **not** a cash expense): "
            f"{_num(accounting.get('fills_slippage_total_quote'), 4)} quote currency "
            f"across {accounting.get('fills_with_slippage')} fills."
        )
        add("")
        add("Likely explanations for the residual:")
        add("")
        for reason in accounting.get("likely_residual_explanations") or []:
            add(f"- {reason}")
        add("")
        if accounting.get("discontinuities"):
            add("### Balance discontinuities")
            add("")
            for item in accounting["discontinuities"]:
                add(
                    f"- `{item['from_date']}` -> `{item['to_date']}`: "
                    f"{_num(item['change'], 4)} ({_num(item['change_pct'], 2, '%')})"
                )
            add("")
        else:
            add("No day-over-day equity discontinuity exceeded the deposit/withdrawal threshold.")
            add("")

    # --- execution -------------------------------------------------------
    add("## Execution attribution")
    add("")
    totals = execution.get("totals") or {}
    add(f"- Fills: {execution.get('total_fills')}")
    add(f"- Quote turnover: {_num(totals.get('quote_turnover'), 2)}")
    add(f"- Commission (observed fills): {_num(totals.get('commission'), 4)}")
    add(
        f"- Recorded slippage: {_num(totals.get('slippage_quote'), 4)} "
        f"({execution.get('slippage_unit')})"
    )
    add(f"- Reconciliation fills: {totals.get('reconciliation_fills')}")
    add(
        f"- Execution telemetry available: "
        f"**{'yes' if execution.get('execution_telemetry_available') else 'no'}**"
    )
    if execution.get("telemetry_columns_missing"):
        add(f"  - missing: `{'`, `'.join(execution['telemetry_columns_missing'])}`")
    add("")
    add(f"> **Unit note.** `{execution.get('slippage_unit_note')}`")
    add("")
    add("Per-symbol turnover (full breakdown in `fill_attribution.csv`):")
    add("")
    add("| Symbol | Fills | Quote turnover | Commission | Slippage (quote) |")
    add("|---|---:|---:|---:|---:|")
    for symbol, stats in (execution.get("by_symbol") or {}).items():
        add(
            f"| {symbol} | {stats['fills']} | {_num(stats['quote_turnover'])} | "
            f"{_num(stats['commission'], 4)} | {_num(stats['slippage_quote'], 4)} |"
        )
    add("")

    # --- positions -------------------------------------------------------
    add("## Closed positions")
    add("")
    if positions.get("lifecycle_coverage_incomplete"):
        add(f"> **WARNING — lifecycle coverage incomplete.** {positions.get('reliability_note')}")
        add("")
    if not positions.get("available"):
        add("No closed positions recorded.")
        add("")
    else:
        add("| Metric | Value |")
        add("|---|---:|")
        add(f"| Closed positions | {positions.get('closed_positions')} |")
        add(f"| Wins / losses | {positions.get('wins')} / {positions.get('losses')} |")
        add(f"| Win rate | {_num(positions.get('win_rate_pct'), 2, '%')} |")
        add(
            f"| Gross profit / loss | {_num(positions.get('gross_profit'), 4)} / "
            f"{_num(positions.get('gross_loss'), 4)} |"
        )
        add(f"| Net PnL | {_num(positions.get('net_pnl'), 4)} |")
        add(f"| Profit factor | {_num(positions.get('profit_factor'), 3)} |")
        add(f"| Expectancy | {_num(positions.get('expectancy'), 4)} |")
        add(f"| Median PnL | {_num(positions.get('median_pnl'), 4)} |")
        add(
            f"| Average R | {_num(positions.get('avg_r_multiple'), 3)} "
            f"(n={positions.get('r_multiple_sample')}) |"
        )
        add(f"| Average holding (h) | {_num(positions.get('avg_holding_hours'), 2)} |")
        add("")
        add(f"Exit reasons: {positions.get('exit_reasons')}")
        add("")
        add("Trades are **not** reconstructed by pairing fills; fills carry no position id.")
        add("")

    # --- shadow ----------------------------------------------------------
    add("## Shadow models")
    add("")
    evaluation = shadow.get("pnl_evaluation") or {}
    add(f"**Profitability: `{evaluation.get('status')}`** — {evaluation.get('reason')}")
    add("")
    add("Required before any model can be ranked by return:")
    add("")
    for item in evaluation.get("missing_data") or []:
        add(f"1. {item}")
    add("")
    if shadow.get("available"):
        add("| Model | Cycles | Mean gross | Max gross | Mean net | Mean scale | Turnover/cycle |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for model, stats in (shadow.get("models") or {}).items():
            add(
                f"| `{model}` | {stats['cycles']} | "
                f"{_num(stats['gross_exposure']['mean'], 3)} | "
                f"{_num(stats['gross_exposure']['max'], 3)} | "
                f"{_num(stats['net_exposure']['mean'], 3)} | "
                f"{_num(stats['portfolio_scale']['mean'], 3)} | "
                f"{_num(stats['turnover_per_cycle']['mean'], 3)} |"
            )
        add("")
        comparisons = shadow.get("comparisons_vs_live") or {}
        if comparisons:
            add("Divergence from the live model:")
            add("")
            add(
                "| Model | Shared cycles | Mean abs weight delta | Signal disagreement | "
                "Materially different cycles |"
            )
            add("|---|---:|---:|---:|---:|")
            for model, stats in comparisons.items():
                add(
                    f"| `{model}` | {stats['shared_cycles']} | "
                    f"{_num(stats['mean_abs_weight_delta'], 4)} | "
                    f"{_num(stats['signal_disagreement_pct'], 2, '%')} | "
                    f"{_num(stats['materially_different_cycles_pct'], 1, '%')} |"
                )
            add("")

    # --- operations ------------------------------------------------------
    add("## Operational attribution")
    add("")
    add(f"> {operations.get('disclaimer')}")
    add("")
    add(
        f"- Incidents: {operations.get('incidents_total')} "
        f"{operations.get('incidents_by_kind')}"
    )
    add(
        f"- Service starts / stops: {operations.get('service_starts')} / "
        f"{operations.get('service_stops')}"
    )
    add(
        f"- Telemetry gaps > 1h: {operations.get('telemetry_gap_count')} "
        f"(largest {_num(operations.get('largest_telemetry_gap_hours'), 2)}h)"
    )
    add(f"- Distinct restart days: {regimes.get('distinct_restart_days')}")
    add("")
    add(
        f"- Incident windows evaluable: {operations.get('incidents_evaluable')} / "
        f"{operations.get('incidents_total')} "
        f"(freshness tolerance {_num(operations.get('freshness_tolerance_seconds'), 0)}s)"
    )
    add("")
    reasons = operations.get("not_evaluable_reasons") or {}
    if reasons:
        add("Incident windows reported as **not evaluable**, and why:")
        add("")
        for reason, count in reasons.items():
            add(f"- {count} x {reason}")
        add("")
    worst = operations.get("most_negative_incident_windows") or []
    if worst:
        add(
            "Most negative equity moves in a ±6h window around an incident "
            "(association, not causation):"
        )
        add("")
        add("| Incident | Started | Before | After | Equity change |")
        add("|---|---|---:|---:|---:|")
        for item in worst[:5]:
            add(
                f"| {item['kind']} | {item['started_ts']} | "
                f"{_num(item.get('equity_before'), 4)} | "
                f"{_num(item.get('equity_after'), 4)} | "
                f"{_num(item.get('equity_change'), 4)} |"
            )
        add("")

    # --- findings --------------------------------------------------------
    add("## Data-quality findings")
    add("")
    summary = analysis.get("findings_summary") or {}
    add(
        f"{summary.get('total', 0)} findings: "
        f"{summary.get('by_severity', {}).get('ERROR', 0)} ERROR, "
        f"{summary.get('by_severity', {}).get('WARNING', 0)} WARNING, "
        f"{summary.get('by_severity', {}).get('INFO', 0)} INFO."
    )
    add("")
    add(f"*{summary.get('no_single_health_score')}*")
    add("")
    lines.extend(_findings_table(findings))
    for item in findings:
        add(f"### `{item['code']}` — {item['title']}")
        add("")
        add(f"- **Severity:** {item['severity']}")
        add(f"- **Scope:** `{item['scope']}`")
        add(f"- **Evidence:** `{json.dumps(item['evidence'], sort_keys=True, default=str)}`")
        add(f"- **Impact:** {item['impact']}")
        add(f"- **Remediation:** {item['remediation']}")
        add("")

    add("## What this audit cannot conclude")
    add("")
    add("- It cannot say the strategy is profitable, on testnet or live.")
    add("- It cannot rank shadow models without forward price data.")
    add("- It cannot attribute PnL to a cause; incident overlap is association only.")
    add("- It cannot validate trade statistics while position lifecycle coverage is incomplete.")
    add("")
    return "\n".join(lines) + "\n"
