"""Acceptance gate for Phase 4A candidates.

The gate is declared before any experiment runs. A missing metric is a FAILURE,
never a pass: an absent value cannot be treated as satisfying a threshold, so a
candidate cannot slip through by simply not reporting something.
"""

from __future__ import annotations

import math
import statistics

#: Minimum number of profitable chronological windows out of the ten evaluated.
MIN_PROFITABLE_WINDOWS = 6
#: Cost multiplier at which a candidate must still be profitable.
COST_STRESS_KEY = "1.5"

PASSED = "PASSED"
FAILED = "FAILED"
INELIGIBLE = "INELIGIBLE"


class GateError(ValueError):
    """Raised when gate inputs are structurally unusable."""


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def window_statistics(windows: list[dict]) -> dict:
    """Summarise the chronological windows a candidate is judged on."""

    returns = [w.get("total_return") for w in windows]
    sharpes = [w.get("sharpe") for w in windows]
    if not windows or not all(_finite(r) for r in returns):
        raise GateError("every chronological window must report a finite total_return")
    profitable = [r for r in returns if r > 0]
    finite_sharpes = [s for s in sharpes if _finite(s)]
    return {
        "window_count": len(windows),
        "profitable_windows": len(profitable),
        "profitable_window_pct": 100.0 * len(profitable) / len(windows),
        "median_window_return": statistics.median(returns),
        "median_window_sharpe": statistics.median(finite_sharpes) if finite_sharpes else None,
        "worst_window_return": min(returns),
        "best_window_return": max(returns),
    }


def evaluate(
    *,
    experiment,
    development: dict,
    windows: list[dict],
    baseline: dict,
    cost_stress: dict,
    path_sensitivity: dict,
    deterministic: bool,
    holdout_status: str,
) -> dict:
    """Score one experiment against the predeclared acceptance gate."""

    stats = window_statistics(windows)
    checks: list[dict] = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- integrity gates -----------------------------------------------------
    check(
        "no_holdout_access",
        holdout_status == "UNTOUCHED",
        f"holdout.status={holdout_status}",
    )
    check("deterministic_reproduction", deterministic, f"identical_metrics={deterministic}")

    residual = development.get("accounting_residual")
    tolerance = 0.02 * (development.get("fills", 0) + 1)
    check(
        "accounting_reconciles",
        _finite(residual) and abs(residual) <= tolerance,
        f"residual={residual} tolerance={tolerance}",
    )

    rejections = development.get("rejections_by_reason")
    if rejections is None:
        check("no_invalid_precision_rejections", False, "rejections_by_reason missing")
    else:
        invalid = {
            k: v for k, v in rejections.items() if k in ("price_precision", "quantity_precision")
        }
        check("no_invalid_precision_rejections", not invalid, f"invalid={invalid or 'none'}")

    # --- performance gates ---------------------------------------------------
    for name, value, threshold, comparison, detail in (
        (
            "positive_development_return",
            development.get("total_return"),
            0.0,
            "gt",
            "aggregate development return must be positive",
        ),
        (
            "positive_median_window_return",
            stats["median_window_return"],
            0.0,
            "gt",
            "median chronological window return must be positive",
        ),
        (
            "profit_factor_beats_baseline",
            development.get("profit_factor"),
            baseline.get("profit_factor"),
            "gt",
            "profit factor must exceed the corrected baseline",
        ),
        (
            "sharpe_beats_baseline",
            development.get("sharpe"),
            baseline.get("sharpe"),
            "gt",
            "Sharpe must exceed the corrected baseline",
        ),
        (
            "max_drawdown_not_worse",
            development.get("max_drawdown"),
            baseline.get("max_drawdown"),
            "gte",
            "max drawdown (negative) must not be worse than the corrected baseline",
        ),
    ):
        if not _finite(value) or not _finite(threshold):
            check(name, False, f"missing metric: value={value} threshold={threshold}")
            continue
        passed = value > threshold if comparison == "gt" else value >= threshold
        check(name, passed, f"{detail}: {value} vs {threshold}")

    check(
        "enough_profitable_windows",
        stats["profitable_windows"] >= MIN_PROFITABLE_WINDOWS,
        f"{stats['profitable_windows']}/{stats['window_count']} profitable, "
        f"need >= {MIN_PROFITABLE_WINDOWS}",
    )

    stressed = (cost_stress or {}).get(COST_STRESS_KEY, {}).get("total_return")
    if not _finite(stressed):
        check("profitable_at_1_5x_costs", False, "1.5x cost-stress return missing")
    else:
        check("profitable_at_1_5x_costs", stressed > 0, f"1.5x cost return={stressed}")

    olhc = (path_sensitivity or {}).get("OLHC", {}).get("total_return")
    ohlc = (path_sensitivity or {}).get("OHLC", {}).get("total_return")
    if not _finite(olhc) or not _finite(ohlc):
        check("survives_olhc_sensitivity", False, "OHLC/OLHC sensitivity return missing")
    else:
        # The qualitative conclusion must survive: both paths agree on sign.
        check(
            "survives_olhc_sensitivity",
            (olhc > 0) == (ohlc > 0),
            f"OHLC={ohlc} OLHC={olhc} (signs must agree)",
        )

    eligible = experiment.eligibility == "candidate"
    failed = [c["check"] for c in checks if not c["passed"]]
    if not eligible:
        status = INELIGIBLE
    else:
        status = PASSED if not failed else FAILED
    return {
        "status": status,
        "eligible_for_selection": eligible,
        "ineligible_reason": None if eligible else experiment.post_hoc_note or "diagnostic only",
        "checks": checks,
        "failed_checks": failed,
        "window_statistics": stats,
        "gate_definition": {
            "min_profitable_windows": MIN_PROFITABLE_WINDOWS,
            "cost_stress_multiplier": COST_STRESS_KEY,
            "baseline_profit_factor": baseline.get("profit_factor"),
            "baseline_sharpe": baseline.get("sharpe"),
            "baseline_max_drawdown": baseline.get("max_drawdown"),
            "missing_metric_policy": "a missing or non-finite metric fails its check",
        },
    }
