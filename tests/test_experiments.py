"""Phase 4A experiment harness: rules, gates, determinism and holdout safety."""

from __future__ import annotations

import copy
import json
import pathlib
from decimal import Decimal
from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderSide

from kronos_mt5.baseline.engine import DAY_NS, ReplayStrategy, run_engine
from kronos_mt5.baseline.instruments import instrument, reject_reason
from kronos_mt5.baseline.metrics import summarize
from kronos_mt5.baseline.synthetic import fixture
from kronos_mt5.experiments import gates, registry, report, runner
from kronos_mt5.experiments.overlays import (
    BEARISH_MA_WINDOW,
    HIGH_VOL,
    CompositeOverlay,
    LongOnlyOverlay,
    NormalVolEntryGateOverlay,
    Overlay,
    ShortBearishRegimeOverlay,
)

BUY, SELL = OrderSide.BUY, OrderSide.SELL


def _strategy(*, closes=None, regime="NORMAL_VOL"):
    return SimpleNamespace(_closes=list(closes or []), current_regime=lambda: regime)


# --- long-only ---------------------------------------------------------------


def test_long_only_never_opens_or_increases_short_exposure():
    overlay = LongOnlyOverlay()
    strategy = _strategy()
    # Opening a short from flat.
    assert overlay.adjust(strategy, SELL, Decimal(5), Decimal(0))[2] == overlay.veto_reason
    # Increasing an existing short.
    assert overlay.adjust(strategy, SELL, Decimal(3), Decimal(-5))[2] == overlay.veto_reason


def test_long_only_allows_long_entries_and_increases_unchanged():
    overlay = LongOnlyOverlay()
    strategy = _strategy()
    for current, qty in ((Decimal(0), Decimal(5)), (Decimal(5), Decimal(3))):
        side, adjusted, veto = overlay.adjust(strategy, BUY, qty, current)
        assert (side, adjusted, veto) == (BUY, qty, None)


def test_long_only_allows_short_reductions_and_exits():
    overlay = LongOnlyOverlay()
    strategy = _strategy()
    # Partial reduction of an existing short: still short, but less so.
    assert overlay.adjust(strategy, BUY, Decimal(3), Decimal(-5)) == (BUY, Decimal(3), None)
    # Full close of an existing short.
    assert overlay.adjust(strategy, BUY, Decimal(5), Decimal(-5)) == (BUY, Decimal(5), None)
    # Reducing a long is untouched.
    assert overlay.adjust(strategy, SELL, Decimal(2), Decimal(5)) == (SELL, Decimal(2), None)


def test_long_only_reversal_is_capped_at_flat_not_reinterpreted():
    """A long-to-short reversal closes the long exactly and never flips short."""

    overlay = LongOnlyOverlay()
    side, qty, veto = overlay.adjust(_strategy(), SELL, Decimal(8), Decimal(5))
    assert (side, qty, veto) == (SELL, Decimal(5), None)
    # The side is never flipped, so a negative signal is not read as a long.
    assert side is SELL


# --- short bearish regime ----------------------------------------------------


def test_bearish_filter_uses_only_closed_historical_candles():
    closes = [100.0] * BEARISH_MA_WINDOW
    strategy = _strategy(closes=closes)
    assert ShortBearishRegimeOverlay.is_bearish(strategy) is False
    # Last closed candle below the causal mean of the last 200 closed candles.
    bearish = _strategy(closes=[100.0] * (BEARISH_MA_WINDOW - 1) + [50.0])
    assert ShortBearishRegimeOverlay.is_bearish(bearish) is True
    # The filter reads exactly the trailing 200 closes, nothing else.
    long_history = _strategy(closes=[1000.0] * 50 + [100.0] * (BEARISH_MA_WINDOW - 1) + [50.0])
    assert ShortBearishRegimeOverlay.is_bearish(long_history) is True


def test_bearish_filter_requires_a_confirmed_window():
    # 199 closes: one short of the fixed window, so the trend is unconfirmed.
    short_history = _strategy(closes=[100.0] * (BEARISH_MA_WINDOW - 2) + [1.0])
    assert len(short_history._closes) == BEARISH_MA_WINDOW - 1
    assert ShortBearishRegimeOverlay.is_bearish(short_history) is False
    overlay = ShortBearishRegimeOverlay()
    assert overlay.adjust(short_history, SELL, Decimal(5), Decimal(0))[2] == overlay.veto_reason


def test_short_permitted_only_in_confirmed_bearish_trend():
    overlay = ShortBearishRegimeOverlay()
    bearish = _strategy(closes=[100.0] * (BEARISH_MA_WINDOW - 1) + [50.0])
    bullish = _strategy(closes=[100.0] * (BEARISH_MA_WINDOW - 1) + [150.0])
    assert overlay.adjust(bearish, SELL, Decimal(5), Decimal(0)) == (SELL, Decimal(5), None)
    assert overlay.adjust(bullish, SELL, Decimal(5), Decimal(0))[2] == overlay.veto_reason
    # Long behaviour is unchanged in both regimes.
    for strategy in (bearish, bullish):
        assert overlay.adjust(strategy, BUY, Decimal(5), Decimal(0)) == (
            BUY,
            Decimal(5),
            None,
        )
    # Short reductions remain allowed even when the trend is not bearish.
    assert overlay.adjust(bullish, BUY, Decimal(3), Decimal(-5)) == (BUY, Decimal(3), None)


def test_bearish_overlay_is_fixed_and_not_optimised():
    assert ShortBearishRegimeOverlay.parameters["moving_average_days"] == 200
    assert ShortBearishRegimeOverlay.parameters["optimised"] is False


# --- HIGH_VOL entry gate -----------------------------------------------------


@pytest.mark.parametrize(
    ("scale", "side", "quantity", "current", "expected_quantity", "vetoed"),
    [
        # Same-side reductions and exact flattening are always unchanged.
        ("0", SELL, "2", "5", "2", False),
        ("0.5", SELL, "2", "5", "2", False),
        ("0", BUY, "2", "-5", "2", False),
        ("0.5", BUY, "2", "-5", "2", False),
        ("0", SELL, "5", "5", "5", False),
        ("0.5", SELL, "5", "5", "5", False),
        ("0", BUY, "5", "-5", "5", False),
        ("0.5", BUY, "5", "-5", "5", False),
        # Same-side increases are blocked or halved.
        ("0", BUY, "3", "5", "3", True),
        ("0.5", BUY, "3", "5", "1.5", False),
        ("0", SELL, "3", "-5", "3", True),
        ("0.5", SELL, "3", "-5", "1.5", False),
        # Smaller and larger cross-zero targets preserve the close and scale
        # only the requested opposite-side exposure.
        ("0", SELL, "8", "5", "5", False),
        ("0.5", SELL, "8", "5", "6.5", False),
        ("0", SELL, "12", "5", "5", False),
        ("0.5", SELL, "12", "5", "8.5", False),
        ("0", BUY, "8", "-5", "5", False),
        ("0.5", BUY, "8", "-5", "6.5", False),
        ("0", BUY, "12", "-5", "5", False),
        ("0.5", BUY, "12", "-5", "8.5", False),
        # From flat, the whole target is new exposure.
        ("0", BUY, "8", "0", "8", True),
        ("0.5", BUY, "8", "0", "4.0", False),
        ("0", SELL, "8", "0", "8", True),
        ("0.5", SELL, "8", "0", "4.0", False),
    ],
)
def test_high_vol_gate_handles_signed_position_cases(
    scale, side, quantity, current, expected_quantity, vetoed
):
    overlay = NormalVolEntryGateOverlay(scale)
    adjusted_side, adjusted_quantity, veto = overlay.adjust(
        _strategy(regime=HIGH_VOL), side, Decimal(quantity), Decimal(current)
    )
    assert adjusted_side is side
    assert adjusted_quantity == Decimal(expected_quantity)
    assert (veto == overlay.veto_reason) is vetoed


@pytest.mark.parametrize(
    ("current", "side", "quantity", "scale", "expected_quantity", "expected_target"),
    [
        ("5", SELL, "8", "0", "5", "0"),
        ("-5", BUY, "8", "0", "5", "0"),
        ("5", SELL, "8", "0.5", "6.5", "-1.5"),
        ("-5", BUY, "8", "0.5", "6.5", "1.5"),
    ],
)
def test_high_vol_gate_explicit_reversal_examples(
    current, side, quantity, scale, expected_quantity, expected_target
):
    adjusted_side, adjusted_quantity, veto = NormalVolEntryGateOverlay(scale).adjust(
        _strategy(regime=HIGH_VOL), side, Decimal(quantity), Decimal(current)
    )
    assert veto is None
    assert adjusted_quantity == Decimal(expected_quantity)
    assert Decimal(current) + (
        adjusted_quantity if adjusted_side == BUY else -adjusted_quantity
    ) == Decimal(expected_target)


def test_high_vol_gate_is_inert_outside_high_vol():
    cases = (
        (BUY, Decimal(3), Decimal(5)),
        (SELL, Decimal(8), Decimal(5)),
        (BUY, Decimal(8), Decimal(-5)),
        (SELL, Decimal(5), Decimal(0)),
    )
    for scale in ("0", "0.5"):
        overlay = NormalVolEntryGateOverlay(scale)
        for regime in ("NORMAL_VOL", "LOW_VOL", "UNKNOWN"):
            strategy = _strategy(regime=regime)
            for side, quantity, current in cases:
                assert overlay.adjust(strategy, side, quantity, current) == (
                    side,
                    quantity,
                    None,
                )


def test_high_vol_scaled_gate_halves_only_the_increase():
    overlay = NormalVolEntryGateOverlay("0.5")
    high = _strategy(regime=HIGH_VOL)
    # From flat, half of the requested new exposure.
    assert overlay.adjust(high, BUY, Decimal(10), Decimal(0)) == (BUY, Decimal(5), None)
    # From a long, only the *increase* is halved: 5 -> 5 + (10-5)*0.5 = 7.5.
    side, qty, veto = overlay.adjust(high, BUY, Decimal(5), Decimal(5))
    assert (side, veto) == (BUY, None)
    assert qty == Decimal("2.5")
    # Reductions are still untouched.
    assert overlay.adjust(high, SELL, Decimal(2), Decimal(5)) == (SELL, Decimal(2), None)


def test_scaled_gate_never_reverses_the_pinned_decision_direction():
    overlay = NormalVolEntryGateOverlay("0.5")
    high = _strategy(regime=HIGH_VOL)
    for side, qty, current in ((SELL, Decimal(10), Decimal(0)), (SELL, Decimal(4), Decimal(-2))):
        adjusted_side, adjusted_qty, veto = overlay.adjust(high, side, qty, current)
        assert adjusted_side is side
        if veto is None:
            assert adjusted_qty > 0


def test_entry_scale_must_be_a_fixed_fraction():
    with pytest.raises(ValueError):
        NormalVolEntryGateOverlay("1")
    with pytest.raises(ValueError):
        NormalVolEntryGateOverlay("-0.1")


# --- combination -------------------------------------------------------------


def test_combined_experiment_applies_both_rules():
    overlay = registry.get("long-only-normal-vol").overlay()
    assert isinstance(overlay, CompositeOverlay)
    high, normal = _strategy(regime=HIGH_VOL), _strategy(regime="NORMAL_VOL")
    # Long-only rule still blocks shorts, in either regime.
    assert overlay.adjust(normal, SELL, Decimal(5), Decimal(0))[2] == "long_only_short_blocked"
    # HIGH_VOL rule still blocks new long exposure.
    assert overlay.adjust(high, BUY, Decimal(5), Decimal(0))[2] == "high_vol_entry_blocked"
    # A long entry in normal volatility passes both rules untouched.
    assert overlay.adjust(normal, BUY, Decimal(5), Decimal(0)) == (BUY, Decimal(5), None)
    # Reductions pass both rules.
    assert overlay.adjust(high, SELL, Decimal(2), Decimal(5)) == (SELL, Decimal(2), None)


@pytest.mark.parametrize(
    ("regime", "side", "quantity", "current", "expected_quantity", "expected_veto"),
    [
        (HIGH_VOL, SELL, "2", "5", "2", None),
        (HIGH_VOL, SELL, "5", "5", "5", None),
        (HIGH_VOL, BUY, "3", "5", "3", "high_vol_entry_blocked"),
        (HIGH_VOL, SELL, "3", "-5", "3", "long_only_short_blocked"),
        (HIGH_VOL, SELL, "8", "5", "5", None),
        (HIGH_VOL, SELL, "12", "5", "5", None),
        (HIGH_VOL, BUY, "8", "-5", "5", None),
        (HIGH_VOL, BUY, "12", "-5", "5", None),
        (HIGH_VOL, BUY, "8", "0", "8", "high_vol_entry_blocked"),
        (HIGH_VOL, SELL, "8", "0", "8", "long_only_short_blocked"),
        ("NORMAL_VOL", BUY, "8", "0", "8", None),
        ("NORMAL_VOL", SELL, "2", "5", "2", None),
    ],
)
def test_long_only_normal_vol_composite_signed_cases(
    regime, side, quantity, current, expected_quantity, expected_veto
):
    adjusted_side, adjusted_quantity, veto = (
        registry.get("long-only-normal-vol")
        .overlay()
        .adjust(_strategy(regime=regime), side, Decimal(quantity), Decimal(current))
    )
    assert adjusted_side is side
    assert adjusted_quantity == Decimal(expected_quantity)
    assert veto == expected_veto


def test_composite_veto_stops_the_chain():
    overlay = CompositeOverlay("x", "y", LongOnlyOverlay(), NormalVolEntryGateOverlay("0"))
    assert overlay.adjust(_strategy(regime=HIGH_VOL), SELL, Decimal(5), Decimal(0))[2] == (
        "long_only_short_blocked"
    )


# --- registry ----------------------------------------------------------------


def test_ltc_diagnostic_changes_universe_and_is_ineligible():
    experiment = registry.get("exclude-ltc-diagnostic")
    assert "LTCUSDT" not in experiment.symbols
    assert len(experiment.symbols) == 7
    assert experiment.eligibility == registry.DIAGNOSTIC
    assert experiment.post_hoc is True
    assert "POST-HOC" in experiment.post_hoc_note
    assert "INELIGIBLE" in experiment.post_hoc_note
    assert experiment.overlay() is None


def test_experiment_fingerprints_are_fixed_and_distinct():
    """Fingerprints pin each experiment's declared identity."""

    fingerprints = {e.experiment_id: e.fingerprint() for e in registry.EXPERIMENTS}
    assert len(set(fingerprints.values())) == len(fingerprints)
    for experiment in registry.EXPERIMENTS:
        assert experiment.fingerprint() == experiment.fingerprint()
        assert len(experiment.fingerprint()) == 64
    assert fingerprints == {
        "control-corrected-baseline": "d9ac170af88246c2920b8fdff826f0c2e30c34f1daba6441275b2c21e4e999bc",
        "long-only": "30fe4df4f6d3c44bf4aca65aff01b9b16e9a91fb23ca1bfdb12face8c57e2d47",
        "short-bearish-regime": "5636b2a2d5f9f9b28d81a7c7ce972d0455ebe16b659ac66294b917e055575c3a",
        "normal-vol-entry-gate": "69b9c712fd181b0a676618f5601e93d8132b00bbd09d9683416ec24d343605fb",
        "normal-vol-entry-gate-scaled": fingerprints["normal-vol-entry-gate-scaled"],
        "long-only-normal-vol": fingerprints["long-only-normal-vol"],
        "exclude-ltc-diagnostic": fingerprints["exclude-ltc-diagnostic"],
    }


def test_registry_declares_candidates_and_diagnostics():
    assert registry.HYPOTHESIS_COUNT == 6
    candidates = [e for e in registry.EXPERIMENTS if e.eligibility == registry.CANDIDATE]
    diagnostics = [e for e in registry.EXPERIMENTS if e.eligibility == registry.DIAGNOSTIC]
    assert {e.experiment_id for e in candidates} == {
        "long-only",
        "short-bearish-regime",
        "normal-vol-entry-gate",
        "normal-vol-entry-gate-scaled",
        "long-only-normal-vol",
    }
    assert {e.experiment_id for e in diagnostics} == {
        "control-corrected-baseline",
        "exclude-ltc-diagnostic",
    }
    for experiment in registry.EXPERIMENTS:
        assert experiment.hypothesis and experiment.description and experiment.rules


def test_cost_multipliers_scale_the_intended_costs_only():
    experiment = registry.get("long-only")
    base = experiment.config(cost_multiplier="1.0")
    for multiplier in ("1.5", "2.0"):
        stressed = experiment.config(cost_multiplier=multiplier)
        factor = float(Decimal(multiplier))
        assert stressed.commission_bps == pytest.approx(base.commission_bps * factor)
        assert stressed.half_spread_bps == pytest.approx(base.half_spread_bps * factor)
        assert stressed.slippage_bps == pytest.approx(base.slippage_bps * factor)
        # Funding is historical and must never be scaled away or omitted.
        assert stressed.funding_mode == "historical" == base.funding_mode
        # Risk caps and the holdout boundary stay frozen.
        assert stressed.leverage == base.leverage
        assert stressed.max_drawdown == base.max_drawdown
        assert stressed.holdout_start == "2026-01-01"


def test_bar_path_variants_are_configured_separately():
    experiment = registry.get("long-only")
    assert experiment.config(bar_path="OHLC").bar_path == "OHLC"
    assert experiment.config(bar_path="OLHC").bar_path == "OLHC"
    assert set(registry.BAR_PATHS) == {"OHLC", "OLHC"}
    assert registry.COST_MULTIPLIERS == ("1.0", "1.5", "2.0")


def test_no_experiment_can_move_the_holdout_boundary():
    for experiment in registry.EXPERIMENTS:
        for path in registry.BAR_PATHS:
            for multiplier in registry.COST_MULTIPLIERS:
                config = experiment.config(bar_path=path, cost_multiplier=multiplier)
                assert config.holdout_start == "2026-01-01"


# --- acceptance gate ---------------------------------------------------------


def _passing_inputs():
    windows = [{"total_return": 0.01 + i * 0.001, "sharpe": 0.5} for i in range(10)]
    development = {
        "total_return": 0.05,
        "sharpe": 0.30,
        "profit_factor": 1.20,
        "max_drawdown": -0.10,
        "accounting_residual": 1e-7,
        "fills": 100,
        "rejections_by_reason": {"long_only_short_blocked": 12},
    }
    return {
        "experiment": registry.get("long-only"),
        "development": development,
        "windows": windows,
        "baseline": {"profit_factor": 1.092818, "sharpe": 0.219903, "max_drawdown": -0.150964},
        "cost_stress": {
            "1.0": {"total_return": 0.05},
            "1.5": {"total_return": 0.03},
            "2.0": {"total_return": 0.01},
        },
        "path_sensitivity": {"OHLC": {"total_return": 0.05}, "OLHC": {"total_return": 0.04}},
        "deterministic": True,
        "holdout_status": "UNTOUCHED",
    }


def test_acceptance_gate_passes_a_fully_qualifying_candidate():
    result = gates.evaluate(**_passing_inputs())
    assert result["status"] == gates.PASSED
    assert result["failed_checks"] == []
    assert result["window_statistics"]["profitable_windows"] == 10


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ({"holdout_status": "EVALUATED_FIXED_BASELINE_ONLY"}, "no_holdout_access"),
        ({"deterministic": False}, "deterministic_reproduction"),
        ({"development": {"accounting_residual": 99.0}}, "accounting_reconciles"),
        (
            {"development": {"rejections_by_reason": {"price_precision": 3}}},
            "no_invalid_precision_rejections",
        ),
        ({"development": {"total_return": -0.01}}, "positive_development_return"),
        ({"development": {"profit_factor": 1.0}}, "profit_factor_beats_baseline"),
        ({"development": {"sharpe": 0.1}}, "sharpe_beats_baseline"),
        ({"development": {"max_drawdown": -0.30}}, "max_drawdown_not_worse"),
        ({"cost_stress": {"1.5": {"total_return": -0.01}}}, "profitable_at_1_5x_costs"),
        (
            {"path_sensitivity": {"OHLC": {"total_return": 0.05}, "OLHC": {"total_return": -0.01}}},
            "survives_olhc_sensitivity",
        ),
    ],
)
def test_acceptance_gate_fails_each_individual_requirement(mutation, expected_failure):
    inputs = _passing_inputs()
    for key, value in mutation.items():
        if isinstance(value, dict) and isinstance(inputs.get(key), dict):
            inputs[key] = {**inputs[key], **value}
        else:
            inputs[key] = value
    result = gates.evaluate(**inputs)
    assert result["status"] == gates.FAILED
    assert expected_failure in result["failed_checks"]


def test_acceptance_gate_fails_on_too_few_profitable_windows():
    inputs = _passing_inputs()
    inputs["windows"] = [{"total_return": 0.01, "sharpe": 0.5}] * 5 + [
        {"total_return": -0.01, "sharpe": -0.5}
    ] * 5
    result = gates.evaluate(**inputs)
    assert result["status"] == gates.FAILED
    assert "enough_profitable_windows" in result["failed_checks"]
    assert "positive_median_window_return" in result["failed_checks"]


@pytest.mark.parametrize(
    "missing",
    [
        "total_return",
        "sharpe",
        "profit_factor",
        "max_drawdown",
        "accounting_residual",
        "rejections_by_reason",
    ],
)
def test_gates_cannot_be_bypassed_by_missing_metrics(missing):
    """A missing metric must fail its check, never silently satisfy it."""

    inputs = _passing_inputs()
    development = dict(inputs["development"])
    development.pop(missing)
    inputs["development"] = development
    result = gates.evaluate(**inputs)
    assert result["status"] == gates.FAILED
    assert result["failed_checks"]


def test_missing_cost_stress_or_path_sensitivity_fails():
    for key in ("cost_stress", "path_sensitivity"):
        inputs = _passing_inputs()
        inputs[key] = {}
        result = gates.evaluate(**inputs)
        assert result["status"] == gates.FAILED


def test_window_statistics_reject_unusable_windows():
    with pytest.raises(gates.GateError):
        gates.window_statistics([])
    with pytest.raises(gates.GateError):
        gates.window_statistics([{"total_return": None, "sharpe": 0.1}])


def test_diagnostic_experiment_is_ineligible_however_it_scores():
    inputs = _passing_inputs()
    inputs["experiment"] = registry.get("exclude-ltc-diagnostic")
    result = gates.evaluate(**inputs)
    assert result["status"] == gates.INELIGIBLE
    assert result["eligible_for_selection"] is False
    assert "POST-HOC" in result["ineligible_reason"]


def test_ranking_excludes_ineligible_experiments_and_never_uses_return_alone():
    results = {
        registry.CONTROL.experiment_id: {
            "development": {"total_return": 0.03},
            "acceptance_gate": {
                "eligible_for_selection": False,
                "status": gates.INELIGIBLE,
                "window_statistics": {},
            },
            "eligibility": "diagnostic",
        },
        "long-only": {
            "development": {
                "total_return": 0.10,
                "sharpe": 0.4,
                "profit_factor": 1.3,
                "max_drawdown": -0.09,
            },
            "acceptance_gate": {
                "eligible_for_selection": True,
                "status": gates.PASSED,
                "window_statistics": {
                    "median_window_return": 0.02,
                    "median_window_sharpe": 0.5,
                    "profitable_windows": 8,
                    "worst_window_return": -0.01,
                },
            },
            "eligibility": "candidate",
        },
        "normal-vol-entry-gate": {
            "development": {
                "total_return": 0.20,
                "sharpe": 0.1,
                "profit_factor": 1.05,
                "max_drawdown": -0.30,
            },
            "acceptance_gate": {
                "eligible_for_selection": True,
                "status": gates.FAILED,
                "window_statistics": {
                    "median_window_return": -0.01,
                    "median_window_sharpe": -0.2,
                    "profitable_windows": 3,
                    "worst_window_return": -0.08,
                },
            },
            "eligibility": "candidate",
        },
    }
    rows = report.ranking(results)
    assert {r["experiment_id"] for r in rows} == {"long-only", "normal-vol-entry-gate"}
    # The highest aggregate return is FAILED and must never rank first.
    assert rows[0]["experiment_id"] == "long-only"
    assert rows[0]["status"] == gates.PASSED
    assert rows[-1]["status"] == gates.FAILED
    assert set(rows[0]["axis_ranks"]) == {
        "median_window_return",
        "sharpe",
        "profit_factor",
        "profitable_windows",
        "max_drawdown",
    }


def _rankable(experiment_id, *, status=gates.FAILED, metric=1.0):
    return {
        "development": {
            "total_return": metric,
            "sharpe": metric,
            "profit_factor": metric,
            "max_drawdown": metric,
        },
        "acceptance_gate": {
            "eligible_for_selection": True,
            "status": status,
            "window_statistics": {
                "median_window_return": metric,
                "median_window_sharpe": metric,
                "profitable_windows": metric,
                "worst_window_return": metric,
            },
        },
        "eligibility": "candidate",
    }


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), float("-inf")])
def test_ranking_puts_missing_and_nonfinite_metrics_last(missing):
    results = {
        "valid": _rankable("valid", metric=1.0),
        "missing": _rankable("missing", metric=missing),
    }
    rows = report.ranking(results)
    assert [row["experiment_id"] for row in rows] == ["valid", "missing"]
    assert set(rows[1]["axis_ranks"].values()) == {len(rows) + 1}


def test_ranking_keeps_passed_first_and_breaks_exact_ties_by_id():
    results = {
        "zeta": _rankable("zeta", status=gates.PASSED, metric=1.0),
        "alpha": _rankable("alpha", status=gates.PASSED, metric=1.0),
        "failed-with-better-metrics": _rankable(
            "failed-with-better-metrics", status=gates.FAILED, metric=99.0
        ),
    }
    rows = report.ranking(results)
    assert [row["experiment_id"] for row in rows] == [
        "alpha",
        "zeta",
        "failed-with-better-metrics",
    ]
    assert rows[0]["axis_ranks"] == rows[1]["axis_ranks"]


def test_measured_determinism_failure_prevents_candidate_from_passing(monkeypatch, tmp_path):
    """A mutating evaluation double cannot be blessed by assumed evidence."""

    primary_runs = 0

    def fake_run(manifest, config, filters_payload, output, *, overlay=None):
        nonlocal primary_runs
        is_primary = config.bar_path == "OHLC" and config.commission_bps == 5.0
        primary_runs += int(is_primary)
        total_return = 0.05 + (0.001 if is_primary and primary_runs == 2 else 0.0)
        development = {
            "total_return": total_return,
            "sharpe": 0.5,
            "profit_factor": 1.5,
            "max_drawdown": -0.05,
            "accounting_residual": 0.0,
            "fills": 100,
            "rejections_by_reason": {},
        }
        windows = [{"total_return": 0.01, "sharpe": 0.5, "max_drawdown": -0.01} for _ in range(10)]
        return {
            "development": development,
            "walk_forward": windows,
            "holdout": {"status": "UNTOUCHED"},
            "configuration": config.payload(),
            "configuration_sha256": config.fingerprint(),
            "dataset_manifest_sha256": "dataset",
            "relevant_source_sha256": {"source.py": "source"},
            "research_adapter_sha256": {"adapter.py": "adapter"},
            "research_implementation_commit": "commit",
            "deployed_bot_commit": "deployed",
            "deployed_strategy_snapshot_sha256": "strategy",
            "deployed_risk_snapshot_sha256": "risk",
            "deployed_snapshot_verification_passed": True,
            "exchange_filter_snapshot_sha256": "filters",
            "dependency_versions": {"python": "test"},
        }

    monkeypatch.setattr(runner, "run_baseline", fake_run)
    result = runner.run_experiment(
        registry.get("long-only"),
        tmp_path / "manifest.json",
        {},
        tmp_path / "output",
        control_development={
            "profit_factor": 1.0,
            "sharpe": 0.1,
            "max_drawdown": -0.10,
        },
    )
    assert result["deterministic_verification"]["comparison_result"] is False
    assert (
        result["deterministic_verification"]["first_result_sha256"]
        != result["deterministic_verification"]["second_result_sha256"]
    )
    assert result["acceptance_gate"]["status"] == gates.FAILED
    assert result["acceptance_gate"]["failed_checks"] == ["deterministic_reproduction"]


# --- engine integration ------------------------------------------------------


@pytest.fixture(scope="module")
def overlay_fixture():
    frames, funding, filters, config = fixture()
    origin = int(frames["BTCUSDT"].iloc[0].open_time) * 1_000_000
    return frames, funding, filters, config, origin


def _run(overlay_fixture, overlay):
    frames, funding, filters, config, origin = overlay_fixture
    return run_engine(
        frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS, overlay
    )


def test_control_run_is_unchanged_by_the_overlay_hook(overlay_fixture):
    """No overlay and a pass-through overlay must produce identical results."""

    without = _run(overlay_fixture, None)
    passthrough = _run(overlay_fixture, Overlay())
    assert without == passthrough


def test_overlay_runs_are_deterministic_and_reconcile(overlay_fixture):
    for overlay in (LongOnlyOverlay(), NormalVolEntryGateOverlay("0"), ShortBearishRegimeOverlay()):
        first = _run(overlay_fixture, overlay)
        second = _run(overlay_fixture, overlay)
        assert first == second, overlay.overlay_id
        metrics = summarize(first)
        assert summarize(second) == metrics
        tolerance = 0.02 * (len(first["fills"]) + len(first["funding"]) + 1)
        assert abs(metrics["accounting_residual"]) <= tolerance


def test_long_only_replay_holds_no_short_exposure(overlay_fixture):
    result = _run(overlay_fixture, LongOnlyOverlay())
    for observation in result["observations"]:
        assert observation["n_open"] >= 0
    directions = {t for t in (f.get("side") for f in result["fills"]) if t}
    assert directions <= {"BUY", "SELL"}
    metrics = summarize(result)
    assert metrics["long_short"]["short"]["trades"] == 0
    assert metrics["long_short"]["short"]["net_pnl"] == 0


def test_overlay_vetoes_are_attributed_suppressed_decisions(overlay_fixture):
    result = _run(overlay_fixture, LongOnlyOverlay())
    for rejection in result["rejections"]:
        assert rejection["symbol"]
        assert rejection["reason"]
        assert rejection["category"]
        assert rejection["source"]
        assert "attempted_qty" in rejection and "attempted_price" in rejection
    vetoes = [r for r in result["rejections"] if r["reason"] == "long_only_short_blocked"]
    assert all(r["category"] == "policy_veto" for r in vetoes)
    assert all(r["source"] == "research_overlay" for r in vetoes)
    metrics = summarize(result)
    assert metrics["policy_vetoes"] == metrics["suppressed_decisions"] == len(vetoes)
    assert metrics["invalid_precision_rejections"] == 0


def test_scaled_quantity_below_step_is_a_policy_veto(overlay_fixture):
    class BelowStepOverlay(Overlay):
        def adjust(self, strategy, side, quantity, current):
            return side, Decimal("0.0005"), None

    result = _run(overlay_fixture, BelowStepOverlay())
    assert result["rejections"]
    assert {r["reason"] for r in result["rejections"]} == {"policy_scaled_quantity_below_step"}
    assert {r["category"] for r in result["rejections"]} == {"policy_veto"}
    assert result["fills"] == []
    metrics = summarize(result)
    assert metrics["policy_vetoes"] == metrics["suppressed_decisions"]
    assert metrics["policy_vetoes"] == len(result["rejections"])
    assert metrics["engine_exchange_rejections"] == 0
    assert metrics["invalid_precision_rejections"] == 0


def test_experiments_retain_exact_price_and_quantity_validation(overlay_fixture):
    """Overlay-adjusted orders still go through exact decimal exchange checks."""

    _frames, _funding, filters, config, _origin = overlay_fixture
    inst = instrument("BTCUSDT", filters["BTCUSDT"], config.commission_bps, config.leverage)
    price = inst.make_price(88187.9)
    qty = inst.make_qty(0.113)
    assert reject_reason(qty, price, filters["BTCUSDT"]) != "price_precision"
    for overlay in (LongOnlyOverlay(), NormalVolEntryGateOverlay("0.5")):
        result = _run(overlay_fixture, overlay)
        reasons = {r["reason"] for r in result["rejections"]}
        assert "price_precision" not in reasons, overlay.overlay_id
        assert "quantity_precision" not in reasons, overlay.overlay_id


def test_overlay_adjusted_quantity_is_snapped_to_step(overlay_fixture):
    """A scaled quantity is re-snapped onto the exchange step before submission."""

    result = _run(overlay_fixture, NormalVolEntryGateOverlay("0.5"))
    step = Decimal("0.001")
    for fill in result["fills"]:
        assert Decimal(str(fill["qty"])) % step == 0, fill


def test_replay_strategy_hook_is_pass_through_without_an_overlay():
    strategy = SimpleNamespace(overlay=None)
    assert ReplayStrategy._experiment_adjust(strategy, BUY, Decimal(5), 0.0) == (
        BUY,
        Decimal(5),
        None,
    )


# --- reporting and holdout safety --------------------------------------------


def test_runner_summary_and_deltas_are_json_serialisable():
    metrics = {
        "total_return": 0.05,
        "sharpe": 0.3,
        "costs": {"commission": 1.0},
        "exposure": {"max_gross_exposure": 0.4},
        "long_short": {"long": {"trades": 5, "net_pnl": 10.0}},
        "per_symbol": {"BTCUSDT": {"trades": 5, "net_pnl": 10.0}},
        "market_regime": {"HIGH_VOL": {"net_pnl": -1.0}},
        "rejections_by_reason": {"x": 1},
    }
    summary = runner.summary(metrics)
    assert json.loads(json.dumps(summary)) == summary
    assert summary["market_regime"] == {"HIGH_VOL": -1.0}
    deltas = runner.deltas(summary, {**summary, "total_return": 0.04})
    assert deltas["total_return"]["delta"] == pytest.approx(0.01)


def test_summary_is_idempotent_so_a_control_summary_can_be_reused():
    """run_all passes an already-summarised control into every experiment."""

    metrics = {
        "total_return": 0.05,
        "sharpe": 0.3,
        "costs": {"commission": 1.0},
        "exposure": {"max_gross_exposure": 0.4},
        "long_short": {"long": {"trades": 5, "net_pnl": 10.0}},
        "per_symbol": {"BTCUSDT": {"trades": 5, "net_pnl": 10.0}},
        "market_regime": {"HIGH_VOL": {"net_pnl": -1.0, "start_equity": 1.0}},
        "rejections_by_reason": {},
    }
    once = runner.summary(metrics)
    twice = runner.summary(once)
    assert once["market_regime"] == {"HIGH_VOL": -1.0}
    assert twice["market_regime"] == once["market_regime"]
    # A control summary must be usable directly as the delta reference.
    assert runner.deltas(once, once)["total_return"]["delta"] == 0


def test_ohlc_and_olhc_are_reported_separately():
    experiment = registry.get("long-only")
    keys = []
    for path in registry.BAR_PATHS:
        for multiplier in registry.COST_MULTIPLIERS:
            if path == "OLHC" and multiplier != "1.0":
                continue
            keys.append(f"{path}-{multiplier}x")
    assert keys == ["OHLC-1.0x", "OHLC-1.5x", "OHLC-2.0x", "OLHC-1.0x"]
    assert experiment.config(bar_path="OHLC") != experiment.config(bar_path="OLHC")


def test_experiment_cli_never_exposes_include_holdout():
    """No experiment code path can request the final holdout."""

    from kronos_mt5.experiments import __main__ as cli

    source = (
        cli.__file__,
        runner.__file__,
        registry.__file__,
    )
    for path in source:
        text = pathlib.Path(path).read_text()
        # No experiment path may request the holdout or register the flag.
        assert "include_holdout=True" not in text
        assert 'add_argument("--include-holdout"' not in text
        assert "include_holdout=include_holdout" not in text


def test_runner_rejects_a_report_whose_holdout_was_touched(monkeypatch):
    experiment = registry.get("long-only")

    def fake_run(manifest, config, filters_payload, output, *, overlay=None):
        return {"holdout": {"status": "EVALUATED_FIXED_BASELINE_ONLY"}, "development": {}}

    monkeypatch.setattr(runner, "run_baseline", fake_run)
    with pytest.raises(ValueError, match="UNTOUCHED"):
        runner._variant(
            experiment,
            None,
            {},
            None,
            bar_path="OHLC",
            cost_multiplier="1.0",
        )


def test_report_render_keeps_control_and_failures_visible():
    bundle = {
        "hypotheses_tested": 2,
        "results": {
            registry.CONTROL.experiment_id: {
                "experiment": registry.CONTROL.identity(),
                "experiment_fingerprint": "0" * 64,
                "eligibility": "diagnostic",
                "post_hoc": False,
                "post_hoc_note": "",
                "provenance": {
                    "research_implementation_commit": "abc",
                    "deployed_bot_commit": "dc8a74c",
                    "deployed_strategy_snapshot_sha256": "s",
                    "deployed_risk_snapshot_sha256": "r",
                    "dataset_manifest_sha256": "m",
                    "exchange_filter_snapshot_sha256": "f",
                },
                "development": {
                    "total_return": 0.033,
                    "costs": {},
                    "exposure": {},
                    "long_short": {},
                    "per_symbol": {},
                    "market_regime": {},
                    "rejections_by_reason": {},
                    "rejections_by_symbol_reason": {},
                },
                "walk_forward": [],
                "path_sensitivity": {},
                "cost_stress": {},
                "holdout": {"status": "UNTOUCHED"},
                "acceptance_gate": {
                    "status": gates.INELIGIBLE,
                    "eligible_for_selection": False,
                    "ineligible_reason": "control",
                    "checks": [],
                    "failed_checks": [],
                    "window_statistics": {
                        "window_count": 0,
                        "profitable_windows": 0,
                        "profitable_window_pct": 0.0,
                        "median_window_return": 0.0,
                        "median_window_sharpe": None,
                        "worst_window_return": 0.0,
                    },
                },
            },
        },
    }
    failed = copy.deepcopy(bundle["results"][registry.CONTROL.experiment_id])
    failed["experiment"] = registry.get("long-only").identity()
    failed["eligibility"] = "candidate"
    failed["acceptance_gate"]["status"] = gates.FAILED
    failed["acceptance_gate"]["eligible_for_selection"] = True
    failed["acceptance_gate"]["failed_checks"] = ["sharpe_beats_baseline"]
    bundle["results"]["long-only"] = failed

    text = report.render(bundle)
    assert registry.CONTROL.experiment_id in text
    assert "long-only" in text
    assert gates.FAILED in text
    assert "sharpe_beats_baseline" in text
    assert "UNTOUCHED" in text
    for note in report.HONESTY:
        assert note in text
