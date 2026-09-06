"""Shadow-model comparison, restricted to what is computable without prices.

A shadow model records the target portfolio it *would* have held. Turning that
into PnL needs forward price returns, which the export does not contain, so
profitability is reported as an explicit `NOT_EVALUABLE` result listing exactly
what data is missing. Everything computed here is a positioning statistic, not a
performance statistic.
"""

from __future__ import annotations

from kronos_mt5.performance_audit.loading import as_float, parse_ts

LIVE_MODEL = "live"
# A target-weight difference below this is noise, not a different portfolio.
MATERIAL_WEIGHT_DELTA = 0.05

NOT_EVALUABLE_REQUIREMENTS = [
    (
        "timestamped OHLCV history for every traded symbol, covering the full "
        "shadow-target window at or finer than the rebalance interval"
    ),
    (
        "the execution/rebalance timing convention (which bar close a target is "
        "formed on, and at which subsequent price it would have been filled)"
    ),
    (
        "a transaction-cost model (maker/taker fees, expected spread and slippage "
        "per symbol and order size)"
    ),
    "funding timestamps and realized funding rates per symbol over the window",
    "an explicit policy for delistings, halted symbols and missing candles",
]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "max": None, "min": None, "count": 0}
    return {
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
        "count": len(values),
    }


def analyze_shadow(rows: list[dict]) -> dict:
    if not rows:
        return {
            "available": False,
            "models": {},
            "pnl_evaluation": _not_evaluable(),
            "note": "no shadow targets recorded",
        }

    # model -> cycle_id -> symbol -> row
    by_model: dict[str, dict[str, dict[str, dict]]] = {}
    cycle_ts: dict[str, str] = {}
    for row in rows:
        model = str(row.get("model") or "UNKNOWN")
        cycle = str(row.get("cycle_id") or "")
        symbol = str(row.get("symbol") or "UNKNOWN")
        by_model.setdefault(model, {}).setdefault(cycle, {})[symbol] = row
        moment = parse_ts(row.get("cycle_ts"))
        if moment is not None:
            cycle_ts.setdefault(cycle, moment.isoformat())

    models: dict[str, dict] = {}
    for model, cycles in sorted(by_model.items()):
        gross_values: list[float] = []
        net_values: list[float] = []
        scale_values: list[float] = []
        funding_scalars: list[float] = []
        turnover_values: list[float] = []
        previous: dict[str, float] | None = None
        row_count = 0
        for cycle in sorted(cycles, key=lambda c: (cycle_ts.get(c, ""), c)):
            symbols = cycles[cycle]
            row_count += len(symbols)
            weights = {
                symbol: (as_float(r.get("target_weight")) or 0.0) for symbol, r in symbols.items()
            }
            gross_values.append(sum(abs(w) for w in weights.values()))
            net_values.append(sum(weights.values()))
            for r in symbols.values():
                scale = as_float(r.get("portfolio_scale"))
                if scale is not None:
                    scale_values.append(scale)
                scalar = as_float(r.get("funding_scalar"))
                if scalar is not None:
                    funding_scalars.append(scalar)
            if previous is not None:
                names = set(previous) | set(weights)
                turnover_values.append(
                    sum(abs(weights.get(n, 0.0) - previous.get(n, 0.0)) for n in names)
                )
            previous = weights

        timestamps = sorted(cycle_ts[c] for c in cycles if c in cycle_ts)
        models[model] = {
            "rows": row_count,
            "cycles": len(cycles),
            "symbols": sorted({s for c in cycles.values() for s in c}),
            "first_cycle_ts": timestamps[0] if timestamps else None,
            "last_cycle_ts": timestamps[-1] if timestamps else None,
            "gross_exposure": _stats(gross_values),
            "net_exposure": _stats(net_values),
            "portfolio_scale": _stats(scale_values),
            "funding_scalar": _stats(funding_scalars),
            "funding_scalar_below_one_pct": (
                sum(1 for s in funding_scalars if s < 1.0) / len(funding_scalars) * 100.0
                if funding_scalars
                else None
            ),
            "turnover_per_cycle": _stats(turnover_values),
            "total_turnover": sum(turnover_values),
        }

    comparisons = _compare_to_live(by_model, cycle_ts)
    return {
        "available": True,
        "models": models,
        "live_model": LIVE_MODEL,
        "comparisons_vs_live": comparisons,
        "material_weight_delta": MATERIAL_WEIGHT_DELTA,
        "pnl_evaluation": _not_evaluable(),
    }


def _compare_to_live(by_model: dict, cycle_ts: dict) -> dict:
    live = by_model.get(LIVE_MODEL)
    if not live:
        return {}
    out: dict = {}
    for model, cycles in sorted(by_model.items()):
        if model == LIVE_MODEL:
            continue
        shared = sorted(set(cycles) & set(live), key=lambda c: (cycle_ts.get(c, ""), c))
        weight_deltas: list[float] = []
        signal_disagreements = 0
        signal_comparisons = 0
        material_cycles = 0
        for cycle in shared:
            theirs = cycles[cycle]
            ours = live[cycle]
            names = set(theirs) | set(ours)
            cycle_material = False
            for symbol in names:
                mine = as_float((ours.get(symbol) or {}).get("target_weight")) or 0.0
                other = as_float((theirs.get(symbol) or {}).get("target_weight")) or 0.0
                delta = abs(other - mine)
                weight_deltas.append(delta)
                if delta >= MATERIAL_WEIGHT_DELTA:
                    cycle_material = True
                my_signal = as_float((ours.get(symbol) or {}).get("signal"))
                their_signal = as_float((theirs.get(symbol) or {}).get("signal"))
                if my_signal is not None and their_signal is not None:
                    signal_comparisons += 1
                    if (my_signal > 0) != (their_signal > 0) or (my_signal < 0) != (
                        their_signal < 0
                    ):
                        signal_disagreements += 1
            if cycle_material:
                material_cycles += 1
        out[model] = {
            "shared_cycles": len(shared),
            "mean_abs_weight_delta": (
                sum(weight_deltas) / len(weight_deltas) if weight_deltas else None
            ),
            "max_abs_weight_delta": max(weight_deltas) if weight_deltas else None,
            "signal_comparisons": signal_comparisons,
            "signal_sign_disagreements": signal_disagreements,
            "signal_disagreement_pct": (
                signal_disagreements / signal_comparisons * 100.0 if signal_comparisons else None
            ),
            "materially_different_cycles": material_cycles,
            "materially_different_cycles_pct": (
                material_cycles / len(shared) * 100.0 if shared else None
            ),
        }
    return out


def _not_evaluable() -> dict:
    return {
        "status": "NOT_EVALUABLE",
        "reason": (
            "shadow models record target portfolios only. Converting targets into "
            "PnL requires forward price returns, which this export does not contain."
        ),
        "missing_data": list(NOT_EVALUABLE_REQUIREMENTS),
        "explicitly_not_computed": [
            "shadow return",
            "shadow Sharpe",
            "shadow drawdown",
            "any ranking of models by profitability",
        ],
    }
