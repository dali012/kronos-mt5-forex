"""Run Phase 4A experiments against the pinned dataset and corrected control.

Each experiment is evaluated through the existing baseline reporting pipeline,
so it inherits the same warm-up rule, chronological windows, provenance capture
and holdout protection. The final holdout is never requested: ``include_holdout``
is not exposed anywhere in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from kronos_mt5.baseline.provenance import collect_provenance
from kronos_mt5.baseline.report import run as run_baseline
from kronos_mt5.marketdata.pipeline import safe_output

from . import gates
from .registry import BAR_PATHS, CONTROL, COST_MULTIPLIERS, EXPERIMENTS, Experiment

#: Metrics carried into every comparison and sensitivity table.
SUMMARY_KEYS = (
    "start_utc",
    "end_utc_exclusive",
    "start_equity",
    "end_equity",
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
    "turnover_over_start_equity",
    "accounting_residual",
    "rejected_orders",
    "annualized_low_confidence",
    "suppressed_metrics",
)

DELTA_KEYS = (
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
    "net_pnl",
    "gross_pnl",
    "turnover_notional",
    "rejected_orders",
)


def summary(metrics: dict) -> dict:
    """Reduce a full metrics block to the comparable summary fields."""

    out = {k: metrics.get(k) for k in SUMMARY_KEYS}
    out["costs"] = dict(metrics.get("costs", {}))
    out["exposure"] = dict(metrics.get("exposure", {}))
    out["long_short"] = {
        side: {
            k: values.get(k)
            for k in ("trades", "net_pnl", "win_rate", "profit_factor", "expectancy_per_trade")
        }
        for side, values in metrics.get("long_short", {}).items()
    }
    out["per_symbol"] = {
        symbol: {k: values.get(k) for k in ("trades", "net_pnl", "profit_factor", "win_rate")}
        for symbol, values in metrics.get("per_symbol", {}).items()
    }
    out["market_regime"] = {
        # Full metrics carry a dict per regime; an already-summarised block
        # carries the net PnL scalar directly, so this stays idempotent.
        regime: values.get("net_pnl") if isinstance(values, dict) else values
        for regime, values in metrics.get("market_regime", {}).items()
    }
    out["rejections_by_reason"] = dict(metrics.get("rejections_by_reason", {}))
    out["rejections_by_symbol"] = dict(metrics.get("rejections_by_symbol", {}))
    out["rejections_by_symbol_reason"] = dict(metrics.get("rejections_by_symbol_reason", {}))
    return out


def window_rows(windows: list[dict]) -> list[dict]:
    return [
        {
            "start_utc": w.get("start_utc"),
            "end_utc_exclusive": w.get("end_utc_exclusive"),
            "total_return": w.get("total_return"),
            "sharpe": w.get("sharpe"),
            "sortino": w.get("sortino"),
            "max_drawdown": w.get("max_drawdown"),
            "trades": w.get("trades"),
            "fills": w.get("fills"),
            "profit_factor": w.get("profit_factor"),
            "net_pnl": w.get("net_pnl"),
            "rejected_orders": w.get("rejected_orders"),
            "annualized_low_confidence": w.get("annualized_low_confidence"),
        }
        for w in windows
    ]


def deltas(candidate: dict, control: dict) -> dict:
    out = {}
    for key in DELTA_KEYS:
        a, b = control.get(key), candidate.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[key] = {"control": a, "experiment": b, "delta": b - a}
        else:
            out[key] = {"control": a, "experiment": b, "delta": None}
    return out


def _variant(
    experiment: Experiment,
    manifest: Path,
    filters_payload: dict,
    output: Path,
    *,
    bar_path: str,
    cost_multiplier: str,
) -> dict:
    """Evaluate one (path, cost) variant. ``include_holdout`` is never passed."""

    config = experiment.config(bar_path=bar_path, cost_multiplier=cost_multiplier)
    report = run_baseline(manifest, config, filters_payload, output, overlay=experiment.overlay())
    if report["holdout"]["status"] != "UNTOUCHED":
        raise ValueError(f"holdout must remain UNTOUCHED, got {report['holdout']['status']}")
    return report


def run_experiment(
    experiment: Experiment,
    manifest: Path,
    filters_payload: dict,
    output: Path,
    *,
    control_development: dict | None = None,
    control_windows: list[dict] | None = None,
) -> dict:
    """Run every predeclared variant of one experiment and score the gate.

    ``control_development`` is an already-summarised control metrics block.
    """

    output = safe_output(output)
    variants: dict[str, dict] = {}
    path_sensitivity: dict[str, dict] = {}
    cost_stress: dict[str, dict] = {}

    primary = None
    for bar_path in BAR_PATHS:
        for cost_multiplier in COST_MULTIPLIERS:
            # Cost stress is measured on the assumed OHLC path; the OLHC path is a
            # separate 1.0x sensitivity. Other pairs are not evaluated.
            if bar_path == "OLHC" and cost_multiplier != "1.0":
                continue
            key = f"{bar_path}-{cost_multiplier}x"
            report = _variant(
                experiment,
                manifest,
                filters_payload,
                output / key,
                bar_path=bar_path,
                cost_multiplier=cost_multiplier,
            )
            development = report["development"]
            variants[key] = {
                "bar_path": bar_path,
                "cost_multiplier": cost_multiplier,
                "development": summary(development),
                "walk_forward": window_rows(report["walk_forward"]),
                "holdout": report["holdout"],
                "configuration_sha256": report["configuration_sha256"],
                "report_path": str((output / key / "report.json").resolve()),
            }
            if cost_multiplier == "1.0":
                path_sensitivity[bar_path] = summary(development)
            if bar_path == "OHLC":
                cost_stress[cost_multiplier] = summary(development)
            if bar_path == "OHLC" and cost_multiplier == "1.0":
                primary = report

    if primary is None:
        raise ValueError("primary OHLC 1.0x variant did not run")

    development = primary["development"]
    windows = primary["walk_forward"]
    baseline_reference = control_development if control_development is not None else development
    gate = gates.evaluate(
        experiment=experiment,
        development=development,
        windows=windows,
        baseline=baseline_reference,
        cost_stress=cost_stress,
        path_sensitivity=path_sensitivity,
        deterministic=True,  # replaced by the reproduce step for passing candidates
        holdout_status=primary["holdout"]["status"],
    )
    result = {
        "schema_version": 1,
        "experiment": experiment.identity(),
        "experiment_fingerprint": experiment.fingerprint(),
        "eligibility": experiment.eligibility,
        "post_hoc": experiment.post_hoc,
        "post_hoc_note": experiment.post_hoc_note,
        "provenance": {
            "research_implementation_commit": primary["research_implementation_commit"],
            "deployed_bot_commit": primary["deployed_bot_commit"],
            "deployed_strategy_snapshot_sha256": primary["deployed_strategy_snapshot_sha256"],
            "deployed_risk_snapshot_sha256": primary["deployed_risk_snapshot_sha256"],
            "deployed_snapshot_verification_passed": primary[
                "deployed_snapshot_verification_passed"
            ],
            "dataset_manifest_sha256": primary["dataset_manifest_sha256"],
            "exchange_filter_snapshot_sha256": primary["exchange_filter_snapshot_sha256"],
            "configuration_sha256": primary["configuration_sha256"],
            "relevant_source_sha256": primary["relevant_source_sha256"],
            "dependency_versions": primary["dependency_versions"],
        },
        "development": summary(development),
        "walk_forward": window_rows(windows),
        "path_sensitivity": path_sensitivity,
        "cost_stress": cost_stress,
        "variants": variants,
        "holdout": primary["holdout"],
        "acceptance_gate": gate,
    }
    if control_development is not None:
        result["delta_vs_control"] = deltas(summary(development), control_development)
    if control_windows is not None:
        result["control_window_statistics"] = gates.window_statistics(control_windows)
    return result


def run_all(
    manifest: Path,
    filters_payload: dict,
    output: Path,
    *,
    only: tuple[str, ...] | None = None,
) -> dict:
    """Run the control first, then score every experiment against it."""

    output = safe_output(output)
    control = run_experiment(CONTROL, manifest, filters_payload, output / CONTROL.experiment_id)
    control_dev = control["variants"]["OHLC-1.0x"]["development"]
    control_windows = control["walk_forward"]

    results = {CONTROL.experiment_id: control}
    for experiment in EXPERIMENTS:
        if experiment.experiment_id == CONTROL.experiment_id:
            continue
        if only and experiment.experiment_id not in only:
            continue
        results[experiment.experiment_id] = run_experiment(
            experiment,
            manifest,
            filters_payload,
            output / experiment.experiment_id,
            control_development=control_dev,
            control_windows=control_windows,
        )
    return {
        "schema_version": 1,
        "provenance": collect_provenance(),
        "control_experiment_id": CONTROL.experiment_id,
        "hypotheses_tested": sum(1 for k in results if k != CONTROL.experiment_id),
        "results": results,
    }


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
