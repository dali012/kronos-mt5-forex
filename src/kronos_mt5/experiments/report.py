"""JSON and Markdown reporting for Phase 4A experiments."""

from __future__ import annotations

import math

from .gates import FAILED, INELIGIBLE, PASSED
from .registry import CONTROL

HONESTY = [
    "The final 2026 holdout was never evaluated; no run passed --include-holdout.",
    (
        "No moving-average length, volatility threshold, entry scale, stop or symbol "
        "was optimised in this PR. Every such value is a fixed predeclared constant."
    ),
    (
        "Each experiment was declared with its hypothesis before it was run, and the "
        "acceptance gate was fixed before any result was seen."
    ),
    (
        "Experiments are few and were evaluated on overlapping history from a survivor "
        "universe. Nothing here establishes statistical significance, and no p-value or "
        "significance claim is made."
    ),
    (
        "Chronological windows overlap the same regimes; a single exceptional window is "
        "not evidence and is never used to promote a candidate."
    ),
    (
        "The control and every failed experiment stay in this report. Failures are not "
        "removed, and the gate was not weakened after results were seen."
    ),
    "Post-hoc experiments are labelled and are ineligible for strategy selection.",
]


def _fmt(value, places=6):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def ranking(results: dict) -> list[dict]:
    """Rank eligible candidates on several axes, never on return alone.

    The composite rank is the mean of the individual ranks for median window
    return, Sharpe, profit factor, profitable-window count and drawdown. It is a
    presentation aid: a candidate that fails the acceptance gate is still FAILED
    however it ranks.
    """

    rows = []
    for experiment_id, result in results.items():
        if experiment_id == CONTROL.experiment_id:
            continue
        if not result["acceptance_gate"]["eligible_for_selection"]:
            continue
        dev = result["development"]
        stats = result["acceptance_gate"]["window_statistics"]
        rows.append(
            {
                "experiment_id": experiment_id,
                "status": result["acceptance_gate"]["status"],
                "total_return": dev.get("total_return"),
                "median_window_return": stats.get("median_window_return"),
                "median_window_sharpe": stats.get("median_window_sharpe"),
                "sharpe": dev.get("sharpe"),
                "profit_factor": dev.get("profit_factor"),
                "profitable_windows": stats.get("profitable_windows"),
                "worst_window_return": stats.get("worst_window_return"),
                "max_drawdown": dev.get("max_drawdown"),
            }
        )
    if not rows:
        return rows
    axes = (
        ("median_window_return", True),
        ("sharpe", True),
        ("profit_factor", True),
        ("profitable_windows", True),
        ("max_drawdown", True),  # less negative is better
    )
    for axis, higher_is_better in axes:
        finite = [
            row
            for row in rows
            if isinstance(row.get(axis), (int, float))
            and not isinstance(row.get(axis), bool)
            and math.isfinite(row[axis])
        ]
        finite.sort(
            key=lambda row: (
                -row[axis] if higher_is_better else row[axis],
                row["experiment_id"],
            )
        )
        previous = object()
        rank = 0
        for position, row in enumerate(finite, start=1):
            if row[axis] != previous:
                rank = position
                previous = row[axis]
            row.setdefault("_ranks", {})[axis] = rank
        # Missing and non-finite values receive a fixed penalty after every
        # valid value. Their ordering can therefore never improve composite
        # rank, and final ties remain deterministic by experiment id.
        for row in rows:
            if axis not in row.get("_ranks", {}):
                row.setdefault("_ranks", {})[axis] = len(rows) + 1
    for row in rows:
        ranks = row.pop("_ranks")
        row["axis_ranks"] = ranks
        row["composite_rank_score"] = sum(ranks.values()) / len(ranks)
    rows.sort(
        key=lambda r: (
            r["status"] != PASSED,
            r["composite_rank_score"],
            r["experiment_id"],
        )
    )
    for position, row in enumerate(rows, start=1):
        row["composite_rank"] = position
    return rows


def render(bundle: dict) -> str:
    results = bundle["results"]
    control = results[CONTROL.experiment_id]
    control_dev = control["development"]
    lines = [
        "# Phase 4A strategy experiments",
        "",
        (
            "Controlled research harness. **No production file, deployed parameter or "
            "live service is changed by this report, and the final 2026 holdout is "
            "untouched.**"
        ),
        "",
        f"- Research implementation commit: `{control['provenance']['research_implementation_commit']}`",
        f"- Deployed bot commit (frozen): `{control['provenance']['deployed_bot_commit']}`",
        f"- Deployed strategy snapshot SHA-256: `{control['provenance']['deployed_strategy_snapshot_sha256']}`",
        f"- Deployed risk snapshot SHA-256: `{control['provenance']['deployed_risk_snapshot_sha256']}`",
        f"- Dataset manifest SHA-256: `{control['provenance']['dataset_manifest_sha256']}`",
        f"- Exchange filter snapshot SHA-256: `{control['provenance']['exchange_filter_snapshot_sha256']}`",
        f"- Hypotheses tested in this PR: **{bundle['hypotheses_tested']}**",
        f"- Holdout status: **{control['holdout']['status']}**",
        "",
        "## Statistical honesty",
        "",
    ]
    lines += [f"- {note}" for note in HONESTY]
    lines += [
        "",
        "## Control: corrected baseline",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "start_utc",
        "end_utc_exclusive",
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "max_drawdown_duration_days",
        "profit_factor",
        "expectancy_per_trade",
        "trades",
        "fills",
        "net_pnl",
        "policy_vetoes",
        "engine_exchange_rejections",
        "pre_submission_rejections",
        "invalid_precision_rejections",
        "rejected_orders",  # backward-compatible raw event count
    ):
        label = (
            "raw_suppression_and_rejection_records" if key == "rejected_orders" else key
        )
        lines.append(f"| {label} | {_fmt(control_dev.get(key))} |")

    lines += [
        "",
        "## Experiment comparison versus control",
        "",
        (
            "| Experiment | Status | Eligibility | Return | Δ Return | Sharpe | Δ Sharpe | "
            "Profit factor | Max DD | Trades | Net PnL | Policy vetoes | Engine/exchange "
            "rejections | Invalid precision | Profitable windows |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment_id, result in results.items():
        dev = result["development"]
        gate = result["acceptance_gate"]
        stats = gate["window_statistics"]
        delta = result.get("delta_vs_control", {})
        lines.append(
            f"| {experiment_id} | {gate['status']} | {result['eligibility']} | "
            f"{_fmt(dev.get('total_return'))} | {_fmt((delta.get('total_return') or {}).get('delta'))} | "
            f"{_fmt(dev.get('sharpe'))} | {_fmt((delta.get('sharpe') or {}).get('delta'))} | "
            f"{_fmt(dev.get('profit_factor'))} | {_fmt(dev.get('max_drawdown'))} | "
            f"{_fmt(dev.get('trades'))} | {_fmt(dev.get('net_pnl'), 2)} | "
            f"{_fmt(dev.get('policy_vetoes'))} | "
            f"{_fmt(dev.get('engine_exchange_rejections'))} | "
            f"{_fmt(dev.get('invalid_precision_rejections'))} | "
            f"{stats.get('profitable_windows')}/{stats.get('window_count')} |"
        )

    rows = ranking(results)
    lines += [
        "",
        "## Ranking of eligible candidates",
        "",
        (
            "Composite rank is the mean rank across median window return, Sharpe, "
            "profit factor, profitable-window count and max drawdown. Ranking is a "
            "presentation aid only: **a FAILED candidate is not selectable at any "
            "rank**, and selection is never made on aggregate return alone."
        ),
        "",
    ]
    if rows:
        lines += [
            (
                "| Rank | Experiment | Status | Composite score | Median window return "
                "| Median window Sharpe | Sharpe | Profit factor | Profitable windows "
                "| Worst window |"
            ),
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['composite_rank']} | {row['experiment_id']} | {row['status']} | "
                f"{_fmt(row['composite_rank_score'], 2)} | {_fmt(row['median_window_return'])} | "
                f"{_fmt(row['median_window_sharpe'])} | {_fmt(row['sharpe'])} | "
                f"{_fmt(row['profit_factor'])} | {row['profitable_windows']} | "
                f"{_fmt(row['worst_window_return'])} |"
            )
    else:
        lines.append("No eligible candidate produced a rankable result.")

    for experiment_id, result in results.items():
        gate = result["acceptance_gate"]
        dev = result["development"]
        lines += [
            "",
            f"## {experiment_id}",
            "",
            (
                f"**Status: {gate['status']}** — eligibility "
                f"`{result['eligibility']}`, fingerprint "
                f"`{result['experiment_fingerprint'][:16]}…`"
            ),
            "",
        ]
        lines.append(f"*Hypothesis*: {result['experiment']['hypothesis']}")
        lines += ["", "*Rules*:", ""]
        lines += [f"- {rule}" for rule in result["experiment"]["rules"]]
        if result["post_hoc"]:
            lines += ["", f"> **POST-HOC / {INELIGIBLE}.** {result['post_hoc_note']}"]
        overlay = result["experiment"].get("overlay")
        if overlay:
            lines += [
                "",
                f"*Changed behaviour*: {overlay['changed_behavior']}",
                f"*Overlay parameters*: `{overlay['parameters']}`",
            ]
        lines += ["", "### Acceptance gate", "", "| Check | Result | Detail |", "|---|---|---|"]
        for check in gate["checks"]:
            lines.append(
                f"| {check['check']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
            )
        if gate["status"] == FAILED:
            lines += ["", f"Failed checks: {', '.join(gate['failed_checks'])}."]
        if gate["status"] == INELIGIBLE:
            lines += ["", f"Not selectable: {gate['ineligible_reason']}"]

        deterministic = result.get("deterministic_verification", {})
        if deterministic:
            lines += [
                "",
                "### Measured deterministic verification",
                "",
                (
                    f"Comparison: **{deterministic['comparison_result']}**. "
                    f"Primary hash `{deterministic['first_result_sha256']}`; "
                    f"independent repeat hash `{deterministic['second_result_sha256']}`."
                ),
            ]

        lines += ["", "### Development metrics", "", "| Metric | Value |", "|---|---:|"]
        for key in (
            "total_return",
            "cagr",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "max_drawdown_duration_days",
            "profit_factor",
            "expectancy_per_trade",
            "win_rate",
            "trades",
            "fills",
            "average_holding_hours",
            "gross_pnl",
            "net_pnl",
            "turnover_notional",
            "accounting_residual",
            "policy_vetoes",
            "engine_exchange_rejections",
            "pre_submission_rejections",
            "invalid_precision_rejections",
            "rejected_orders",  # backward-compatible raw event count
        ):
            label = (
                "raw_suppression_and_rejection_records" if key == "rejected_orders" else key
            )
            lines.append(f"| {label} | {_fmt(dev.get(key))} |")
        lines += ["", "| Cost | Value |", "|---|---:|"]
        for key, value in dev.get("costs", {}).items():
            lines.append(f"| {key} | {_fmt(value, 4)} |")
        lines += ["", "| Exposure | Value |", "|---|---:|"]
        for key, value in dev.get("exposure", {}).items():
            lines.append(f"| {key} | {_fmt(value)} |")

        lines += [
            "",
            "| Direction | Trades | Net PnL | Win rate | Profit factor |",
            "|---|---:|---:|---:|---:|",
        ]
        for side, values in dev.get("long_short", {}).items():
            lines.append(
                f"| {side} | {_fmt(values.get('trades'))} | {_fmt(values.get('net_pnl'), 2)} | "
                f"{_fmt(values.get('win_rate'))} | {_fmt(values.get('profit_factor'))} |"
            )
        lines += ["", "| Symbol | Trades | Net PnL | Profit factor |", "|---|---:|---:|---:|"]
        for symbol, values in sorted(dev.get("per_symbol", {}).items()):
            lines.append(
                f"| {symbol} | {_fmt(values.get('trades'))} | {_fmt(values.get('net_pnl'), 2)} | "
                f"{_fmt(values.get('profit_factor'))} |"
            )
        lines += ["", "| Regime | Net PnL |", "|---|---:|"]
        for regime, value in dev.get("market_regime", {}).items():
            lines.append(f"| {regime} | {_fmt(value, 2)} |")

        lines += ["", "### Decision suppression and execution rejections", ""]
        by_reason = dev.get("rejections_by_reason") or {}
        by_symbol_reason = dev.get("rejections_by_symbol_reason") or {}
        lines.append(
            f"Policy vetoes / suppressed decisions: {dev.get('policy_vetoes')}; "
            f"actual matching-engine/exchange rejections: "
            f"{dev.get('engine_exchange_rejections')}; pre-submission validation "
            f"rejections: {dev.get('pre_submission_rejections')}; invalid precision "
            f"rejections: {dev.get('invalid_precision_rejections')}. The backward-compatible "
            f"raw event count is {dev.get('rejected_orders')}. Overlay vetoes were never "
            "submitted to Binance."
        )
        lines.append(f"Reasons across all categories: {by_reason or 'none'}.")
        if by_symbol_reason:
            lines += ["", "| Symbol | Reason | Count |", "|---|---|---:|"]
            for key, count in by_symbol_reason.items():
                symbol, _, reason = key.partition("/")
                lines.append(f"| {symbol} | {reason} | {count} |")

        stats = gate["window_statistics"]
        lines += [
            "",
            "### Chronological windows",
            "",
            (
                f"Profitable: **{stats['profitable_windows']}/"
                f"{stats['window_count']}** "
                f"({_fmt(stats['profitable_window_pct'], 1)}%). "
                f"Median return {_fmt(stats['median_window_return'])}, "
                f"median Sharpe {_fmt(stats['median_window_sharpe'])}, "
                f"worst {_fmt(stats['worst_window_return'])}."
            ),
            "",
            (
                "| Start | End exclusive | Return | Sharpe | Max DD | Trades | Fills | "
                "Policy vetoes | Engine/exchange | Invalid precision |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for w in result["walk_forward"]:
            lines.append(
                f"| {str(w['start_utc'])[:10]} | {str(w['end_utc_exclusive'])[:10]} | "
                f"{_fmt(w['total_return'])} | {_fmt(w['sharpe'], 4)} | {_fmt(w['max_drawdown'])} | "
                f"{w['trades']} | {w['fills']} | {w['policy_vetoes']} | "
                f"{w['engine_exchange_rejections']} | {w['invalid_precision_rejections']} |"
            )
        lines += [
            "",
            (
                "*All chronological windows are under one year, so their "
                "annualised figures are low confidence by construction.*"
            ),
        ]

        lines += [
            "",
            "### Path sensitivity (1.0x costs)",
            "",
            "| Path | Return | Sharpe | Max DD | Trades | Net PnL |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for path, values in result["path_sensitivity"].items():
            lines.append(
                f"| {path} | {_fmt(values.get('total_return'))} | {_fmt(values.get('sharpe'))} | "
                f"{_fmt(values.get('max_drawdown'))} | {_fmt(values.get('trades'))} | "
                f"{_fmt(values.get('net_pnl'), 2)} |"
            )
        lines += [
            "",
            "### Cost stress (OHLC path)",
            "",
            (
                "Commission, half-spread and slippage are scaled together. "
                "Historical funding is never reduced or removed."
            ),
            "",
            (
                "| Multiplier | Return | Sharpe | Net PnL | Commission | Spread "
                "| Slippage | Funding |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for multiplier, values in result["cost_stress"].items():
            costs = values.get("costs", {})
            lines.append(
                f"| {multiplier}x | {_fmt(values.get('total_return'))} | {_fmt(values.get('sharpe'))} | "
                f"{_fmt(values.get('net_pnl'), 2)} | {_fmt(costs.get('commission'), 2)} | "
                f"{_fmt(costs.get('spread'), 2)} | {_fmt(costs.get('slippage'), 2)} | "
                f"{_fmt(costs.get('funding'), 2)} |"
            )
        if result.get("delta_vs_control"):
            lines += [
                "",
                "### Delta versus corrected control",
                "",
                "| Metric | Control | Experiment | Delta |",
                "|---|---:|---:|---:|",
            ]
            for key, values in result["delta_vs_control"].items():
                lines.append(
                    f"| {key} | {_fmt(values['control'])} | {_fmt(values['experiment'])} | "
                    f"{_fmt(values['delta'])} |"
                )

    lines += [
        "",
        "## Conclusion scope",
        "",
        (
            "This report does not select, recommend or deploy a strategy. It "
            "records which predeclared hypotheses survived a fixed gate on "
            "development data only. Any candidate would still require untouched "
            "future observations before deployment."
        ),
        "",
    ]
    return "\n".join(lines)
