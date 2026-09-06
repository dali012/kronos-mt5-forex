"""Deterministic data-quality findings.

Every finding carries a stable code, the table/field it concerns, the evidence
that produced it, what it means for interpreting performance, and what to do
about it. There is deliberately NO single "health score": a scalar would hide
exactly the weaknesses this tool exists to surface.
"""

from __future__ import annotations

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

_SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


def finding(
    code: str,
    severity: str,
    title: str,
    *,
    scope: str,
    evidence: dict,
    impact: str,
    remediation: str,
    prominent: bool = False,
) -> dict:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "scope": scope,
        "evidence": evidence,
        "impact": impact,
        "remediation": remediation,
        "prominent": prominent,
    }


def build_findings(analysis: dict) -> list[dict]:
    """Derive findings from a completed analysis. Pure and deterministic."""
    out: list[dict] = []
    schema = analysis.get("schema", {})
    equity = analysis.get("equity", {})
    accounting = analysis.get("accounting", {})
    execution = analysis.get("execution", {})
    positions = analysis.get("positions", {})
    shadow = analysis.get("shadow", {})
    operations = analysis.get("operations", {})
    regimes = analysis.get("regimes", {})
    context = analysis.get("context", {})

    # --- environment -----------------------------------------------------
    environment = str(context.get("binance_environment") or "").upper()
    if environment == "TESTNET" or context.get("demo_only"):
        out.append(
            finding(
                "PA-ENV-001",
                WARNING,
                "Results come from a Binance TESTNET account, not live capital",
                scope="config/env",
                evidence={
                    "binance_environment": context.get("binance_environment"),
                    "demo_only": context.get("demo_only"),
                },
                impact=(
                    "Testnet fills, liquidity, funding and latency do not reproduce "
                    "live market impact or queue position. These figures are evidence "
                    "that the SYSTEM runs, not that the STRATEGY is profitable."
                ),
                remediation=(
                    "Do not extrapolate to live returns. Any go-live decision needs a "
                    "separate, funded, small-size live sample."
                ),
                prominent=True,
            )
        )

    if not context.get("provenance_known"):
        out.append(
            finding(
                "PA-ENV-002",
                WARNING,
                "Trading environment provenance is unknown",
                scope="context/provenance",
                evidence={
                    "missing": context.get("missing_provenance"),
                    "binance_environment": context.get("binance_environment"),
                    "demo_only": context.get("demo_only"),
                    "source_commit": (context.get("source") or {}).get("source_commit"),
                },
                impact=(
                    "The audit cannot establish whether this database represents "
                    "testnet, live trading, several runs mixed together, or a copy "
                    "of another account. Environment was NOT guessed from table "
                    "contents. Every profitability conclusion is therefore "
                    "unsupported, and these results must not be presented as live."
                ),
                remediation=(
                    "Audit the full sanitized export archive rather than a bare "
                    "database, so the environment and source commit travel with "
                    "the data."
                ),
                prominent=True,
            )
        )

    # --- sample length ---------------------------------------------------
    days = equity.get("days_covered")
    if isinstance(days, int) and days < 365:
        out.append(
            finding(
                "PA-STAT-001",
                WARNING,
                f"Sample covers {days} days — shorter than one year",
                scope="equity",
                evidence={
                    "days_covered": days,
                    "days_observed": equity.get("days_observed"),
                    "daily_return_count": equity.get("daily_return_count"),
                },
                impact=(
                    "Annualized return, volatility, Sharpe, Sortino and Calmar are "
                    "extrapolations from a short window and carry wide error bars. "
                    "A single regime (trending or ranging) can dominate the result."
                ),
                remediation=(
                    "Treat annualized figures as indicative only. Compare against "
                    "out-of-sample windows before drawing conclusions."
                ),
                prominent=True,
            )
        )

    # --- OHLCV -----------------------------------------------------------
    if not context.get("ohlcv_available"):
        out.append(
            finding(
                "PA-DATA-001",
                WARNING,
                "No OHLCV price history in the export",
                scope="export",
                evidence={"market_data_files": context.get("market_data_files", 0)},
                impact=(
                    "Shadow-model PnL, benchmark comparison, per-trade mark-to-market "
                    "and any counterfactual strategy evaluation are impossible."
                ),
                remediation=(
                    "Export timestamped OHLCV for every traded symbol over the full "
                    "window before attempting strategy comparison (see the "
                    "NOT_EVALUABLE requirements)."
                ),
                prominent=True,
            )
        )

    # --- position lifecycle ---------------------------------------------
    if positions.get("lifecycle_coverage_incomplete"):
        out.append(
            finding(
                "PA-DATA-002",
                WARNING,
                "Closed-position lifecycle coverage looks incomplete",
                scope="closed_positions",
                evidence={
                    "closed_positions": positions.get("closed_positions"),
                    "total_fills": positions.get("total_fills"),
                    "coverage_ratio": positions.get("coverage_ratio"),
                },
                impact=(
                    "Win rate, profit factor, expectancy, average R and holding period "
                    "describe only the few positions that were recorded. They are NOT "
                    "strategy performance and must not be quoted as such."
                ),
                remediation=(
                    "Fix position lifecycle recording in the companion, or rebuild "
                    "trades from venue position history. Do not pair fills naively."
                ),
                prominent=True,
            )
        )
    if positions.get("missing_r_multiple"):
        out.append(
            finding(
                "PA-DATA-003",
                INFO,
                "Some closed positions have no R multiple",
                scope="closed_positions.r_multiple",
                evidence={
                    "missing_r_multiple": positions.get("missing_r_multiple"),
                    "r_multiple_sample": positions.get("r_multiple_sample"),
                },
                impact="Average and median R are computed from a subset only.",
                remediation="Ensure initial risk is recorded when a position opens.",
            )
        )

    # --- execution telemetry ---------------------------------------------
    missing_telemetry = execution.get("telemetry_columns_missing") or []
    if missing_telemetry:
        out.append(
            finding(
                "PA-DATA-004",
                WARNING,
                "Fills predate the execution-telemetry migration",
                scope="fills",
                evidence={
                    "missing_columns": missing_telemetry,
                    "present_columns": execution.get("telemetry_columns_present"),
                    "total_fills": execution.get("total_fills"),
                },
                impact=(
                    "Maker/taker mix, execution role, decision-to-fill latency and "
                    "implementation shortfall in basis points cannot be computed, so "
                    "execution quality can only be assessed in quote currency."
                ),
                remediation=(
                    "Re-export after the companion migration has run, then re-audit "
                    "to obtain liquidity and shortfall-bps attribution."
                ),
            )
        )
    totals = execution.get("totals") or {}
    for field, code in (
        ("missing_kind", "PA-DATA-005"),
        ("missing_reference_price", "PA-DATA-006"),
        ("missing_commission", "PA-DATA-007"),
        ("missing_trade_id", "PA-DATA-008"),
    ):
        count = totals.get(field) or 0
        if count:
            out.append(
                finding(
                    code,
                    INFO,
                    f"{count} fills are missing `{field.removeprefix('missing_')}`",
                    scope=f"fills.{field.removeprefix('missing_')}",
                    evidence={field: count, "total_fills": execution.get("total_fills")},
                    impact=(
                        "Those fills are excluded from the corresponding aggregate, so "
                        "grouped totals cover fewer fills than the headline count."
                    ),
                    remediation=(
                        "Usually venue-reconciled fills that carry no decision "
                        "metadata; confirm the recorder tags every strategy order."
                    ),
                )
            )
    if execution.get("duplicate_fill_ids"):
        out.append(
            finding(
                "PA-DATA-009",
                ERROR,
                "Duplicate fill ids detected",
                scope="fills.fill_id",
                evidence={"duplicate_fill_ids": execution.get("duplicate_fill_ids")},
                impact="Turnover, commission and slippage totals are double counted.",
                remediation="Investigate the recorder's fill-id construction.",
            )
        )

    # --- equity data quality ---------------------------------------------
    if equity.get("invalid_equity_rows"):
        out.append(
            finding(
                "PA-DATA-010",
                ERROR,
                "Equity table contains non-positive or unusable values",
                scope="equity.equity",
                evidence={"invalid_equity_rows": equity.get("invalid_equity_rows")},
                impact="Those observations are dropped; the curve may have holes.",
                remediation="Investigate the snapshot writer for those timestamps.",
            )
        )
    if equity.get("duplicate_timestamps"):
        out.append(
            finding(
                "PA-DATA-011",
                WARNING,
                "Duplicate equity timestamps",
                scope="equity.ts",
                evidence={"duplicate_timestamps": equity.get("duplicate_timestamps")},
                impact=(
                    "Daily normalization keeps the last observation, so duplicates do "
                    "not bias the daily series, but they indicate a writer problem."
                ),
                remediation="Add a uniqueness constraint or de-duplicate on write.",
            )
        )
    if equity.get("missing_day_count"):
        out.append(
            finding(
                "PA-DATA-012",
                WARNING,
                f"{equity.get('missing_day_count')} calendar days have no equity observation",
                scope="equity",
                evidence={
                    "missing_day_count": equity.get("missing_day_count"),
                    "missing_days": (equity.get("missing_days") or [])[:20],
                },
                impact=(
                    "Daily returns bridge the gap, so a multi-day move is compressed "
                    "into one return and volatility is understated."
                ),
                remediation="Correlate with restarts and downtime before trusting ratios.",
            )
        )
    if equity.get("observation_gap_count"):
        out.append(
            finding(
                "PA-OPS-001",
                WARNING,
                f"{equity.get('observation_gap_count')} telemetry gaps longer than one hour",
                scope="equity.ts",
                evidence={
                    "observation_gap_count": equity.get("observation_gap_count"),
                    "largest_gaps": (equity.get("observation_gaps") or [])[:5],
                },
                impact=(
                    "The bot was not recording during those windows; equity moves "
                    "across them are unattributed."
                ),
                remediation="Cross-reference with BOT_DOWN incidents and service restarts.",
            )
        )
    raw = equity.get("observation_series") or {}
    if raw and equity.get("min_equity") is not None:
        min_gap = abs((raw.get("min_equity") or 0) - equity["min_equity"])
        max_gap = abs((raw.get("max_equity") or 0) - (equity.get("max_equity") or 0))
        if min_gap > 0.01 or max_gap > 0.01:
            out.append(
                finding(
                    "PA-STAT-003",
                    INFO,
                    "Daily-series extremes differ from raw intraday extremes",
                    scope="equity",
                    evidence={
                        "daily_min": equity.get("min_equity"),
                        "raw_min": raw.get("min_equity"),
                        "daily_max": equity.get("max_equity"),
                        "raw_max": raw.get("max_equity"),
                        "daily_start": equity.get("start_equity"),
                        "raw_first": raw.get("first_equity"),
                    },
                    impact=(
                        "Ratio statistics use the daily series (last observation of "
                        "each UTC day), so its min/max/start differ from a raw SELECT "
                        "over the equity table. Neither is wrong; they answer "
                        "different questions."
                    ),
                    remediation=(
                        "Quote daily figures for risk ratios and raw figures for "
                        "intraday extremes; the report shows both."
                    ),
                )
            )

    if equity.get("zero_variance"):
        out.append(
            finding(
                "PA-STAT-002",
                INFO,
                "Daily returns have zero variance",
                scope="equity",
                evidence={"daily_return_count": equity.get("daily_return_count")},
                impact="Sharpe, Sortino and volatility are undefined and reported as null.",
                remediation="Confirm the account was actually trading over the window.",
            )
        )

    # --- accounting -------------------------------------------------------
    if accounting.get("conflicting_duplicate_ids", {}).get("total"):
        out.append(
            finding(
                "PA-ACC-005",
                ERROR,
                "Record ids appear more than once with conflicting content",
                scope="income.income_id / fills.fill_id",
                evidence={
                    "income_ids": accounting["conflicting_duplicate_ids"]["income"][:20],
                    "fill_ids": accounting["conflicting_duplicate_ids"]["fills"][:20],
                    "total": accounting["conflicting_duplicate_ids"]["total"],
                },
                impact=(
                    "No row was silently chosen, so the affected totals — and the "
                    "reconciliation residual derived from them — are unreliable."
                ),
                remediation=(
                    "Fix the writer's id construction, then re-export. Until then "
                    "treat the accounting section as indicative only."
                ),
                prominent=True,
            )
        )
    income_window = accounting.get("income_window") or {}
    fills_window = accounting.get("fills_window") or {}
    outside = sum(
        (window.get(key) or 0)
        for window in (income_window, fills_window)
        for key in ("before_window", "after_window", "invalid_timestamp")
    )
    if outside:
        out.append(
            finding(
                "PA-ACC-006",
                INFO,
                f"{outside} ledger record(s) fall outside the equity window",
                scope="income + fills vs equity window",
                evidence={
                    "window": accounting.get("window"),
                    "income": {
                        k: income_window.get(k)
                        for k in ("before_window", "after_window", "invalid_timestamp")
                    },
                    "fills": {
                        k: fills_window.get(k)
                        for k in ("before_window", "after_window", "invalid_timestamp")
                    },
                    "excluded_income_amounts": income_window.get("excluded_amounts_by_type"),
                },
                impact=(
                    "Those records are excluded so the identity is evaluated over "
                    "exactly the equity interval. Totals here will not match a naive "
                    "SUM over the whole table."
                ),
                remediation=(
                    "Expected when the income cursor starts before the first equity "
                    "snapshot; investigate only if the excluded amounts are large."
                ),
            )
        )
    if accounting.get("available") and not accounting.get("reconciled_within_tolerance"):
        out.append(
            finding(
                "PA-ACC-001",
                WARNING,
                "Accounting identity does not reconcile within tolerance",
                scope="equity + income",
                evidence={
                    "equity_change": accounting.get("equity_change"),
                    "explained_change": accounting.get("explained_change"),
                    "residual": accounting.get("residual"),
                    "tolerance": accounting.get("residual_tolerance"),
                },
                impact=(
                    "Part of the equity change is unexplained by the ledger, so PnL "
                    "attribution is incomplete."
                ),
                remediation=(
                    "Check for transfers, an account reset, or an income cursor that "
                    "started after the first equity observation."
                ),
            )
        )
    delta = accounting.get("fills_commission_vs_income")
    if delta is not None and abs(delta) > 0.01:
        out.append(
            finding(
                "PA-ACC-002",
                WARNING,
                "Per-fill commissions disagree with the income ledger",
                scope="fills.commission vs income.COMMISSION",
                evidence={
                    "fills_commission_total": accounting.get("fills_commission_total"),
                    "income_commission_cost": accounting.get("commissions_cost"),
                    "difference": delta,
                    "fills_with_commission": accounting.get("fills_with_commission"),
                },
                impact=(
                    "Per-fill cost attribution understates or overstates true fees. "
                    "The income ledger is authoritative for total cost."
                ),
                remediation=(
                    "Use the income ledger for cost accounting; treat fills.commission "
                    "as observed-fills-only."
                ),
            )
        )
    if accounting.get("discontinuities"):
        out.append(
            finding(
                "PA-ACC-003",
                WARNING,
                "Large day-over-day equity discontinuities detected",
                scope="equity.equity",
                evidence={"discontinuities": accounting["discontinuities"][:5]},
                impact=(
                    "A deposit, withdrawal or account reset inside the window makes "
                    "total return and drawdown meaningless across that boundary."
                ),
                remediation="Split the sample at the discontinuity and audit each part.",
            )
        )
    out.append(
        finding(
            "PA-ACC-004",
            INFO,
            "Legacy `slippage` column is quote currency, not basis points",
            scope="fills.slippage",
            evidence={
                "unit": execution.get("slippage_unit"),
                "verified_against": (
                    "companion/recorder.py -> execution.implementation_shortfall_quote"
                ),
                "total_quote": (execution.get("totals") or {}).get("slippage_quote"),
            },
            impact=(
                "Summing it as basis points, or adding it to cash PnL, would be wrong: "
                "it is already embedded in the execution price."
            ),
            remediation="Report it as a measurement only, as this audit does.",
        )
    )

    # --- regimes / operations ---------------------------------------------
    starts = regimes.get("starts") or 0
    if starts > 1:
        out.append(
            finding(
                "PA-OPS-002",
                WARNING,
                f"{starts} service starts split the window into possible regimes",
                scope="ops_events",
                evidence={
                    "starts": starts,
                    "stops": regimes.get("stops"),
                    "distinct_restart_days": regimes.get("distinct_restart_days"),
                },
                impact=(
                    "Configuration may have changed at a restart, so the sample is not "
                    "guaranteed to be one continuous strategy. Pooled statistics can "
                    "mix different systems."
                ),
                remediation=(
                    "Reconstruct the config history per segment and audit segments "
                    "separately before comparing strategies."
                ),
                prominent=True,
            )
        )
    incidents = operations.get("incidents_by_kind") or {}
    if operations.get("incidents_not_evaluable"):
        out.append(
            finding(
                "PA-OPS-005",
                INFO,
                f"{operations['incidents_not_evaluable']} incident window(s) not evaluable",
                scope="incidents vs equity",
                evidence={
                    "not_evaluable": operations.get("incidents_not_evaluable"),
                    "evaluable": operations.get("incidents_evaluable"),
                    "reasons": operations.get("not_evaluable_reasons"),
                    "freshness_tolerance_seconds": operations.get("freshness_tolerance_seconds"),
                },
                impact=(
                    "No equity change is reported for those incidents. A stale or "
                    "missing endpoint would otherwise produce a number that looks "
                    "like measurement but describes a different moment."
                ),
                remediation=(
                    "Usually a telemetry gap around the incident; correlate with "
                    "BOT_DOWN and restart events."
                ),
            )
        )
    if incidents.get("MARK_STALE"):
        out.append(
            finding(
                "PA-OPS-003",
                WARNING,
                f"{incidents['MARK_STALE']} MARK_STALE incidents",
                scope="incidents",
                evidence={"mark_stale": incidents["MARK_STALE"]},
                impact=(
                    "While marks were stale the bot blocked new entries and its "
                    "mark-to-market equity was not updating, so both trading behaviour "
                    "and the recorded curve are affected in those windows."
                ),
                remediation="Investigate mark-stream stability before trusting fine-grained equity.",
            )
        )
    if incidents.get("BOT_DOWN"):
        out.append(
            finding(
                "PA-OPS-004",
                WARNING,
                f"{incidents['BOT_DOWN']} BOT_DOWN incident(s)",
                scope="incidents",
                evidence={"bot_down": incidents["BOT_DOWN"]},
                impact=(
                    "Positions were unmanaged while the bot was down; protective orders "
                    "held exposure but no rebalancing occurred."
                ),
                remediation="Correlate downtime windows with equity moves before attributing PnL.",
            )
        )

    # --- shadow -----------------------------------------------------------
    if (shadow.get("pnl_evaluation") or {}).get("status") == "NOT_EVALUABLE":
        out.append(
            finding(
                "PA-SHADOW-001",
                WARNING,
                "Shadow-model profitability is NOT_EVALUABLE",
                scope="shadow_targets",
                evidence={
                    "models": sorted((shadow.get("models") or {}).keys()),
                    "missing_data": shadow["pnl_evaluation"]["missing_data"],
                },
                impact=(
                    "Shadow models can be compared on positioning only. Ranking them by "
                    "return, Sharpe or drawdown is impossible with this export."
                ),
                remediation="Supply the OHLCV and cost inputs listed in the evaluation block.",
                prominent=True,
            )
        )

    # --- schema -----------------------------------------------------------
    for table in schema.get("missing_known_tables") or []:
        out.append(
            finding(
                "PA-SCHEMA-001",
                ERROR if table in ("equity", "fills") else WARNING,
                f"Expected table `{table}` is missing",
                scope=f"schema.{table}",
                evidence={"missing_table": table},
                impact="Every metric derived from that table is unavailable.",
                remediation="Re-export from a companion database that contains it.",
            )
        )

    out.sort(key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 9), f["code"]))
    return out


def summarize_findings(items: list[dict]) -> dict:
    counts = {ERROR: 0, WARNING: 0, INFO: 0}
    for item in items:
        counts[item["severity"]] = counts.get(item["severity"], 0) + 1
    return {
        "total": len(items),
        "by_severity": counts,
        "prominent_codes": [f["code"] for f in items if f.get("prominent")],
        "no_single_health_score": (
            "Findings are reported individually on purpose. A scalar health score "
            "would average away the specific weaknesses that limit interpretation."
        ),
    }
