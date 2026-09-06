"""Explicit accounting and daily, 365-day risk ratios; undefined values are null."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .engine import DAY_NS


def rejection_counts(rejections: list[dict], *fields: str) -> dict:
    """Count rejections grouped by the given record fields, sorted by key.

    A rejection must always carry its symbol; an unattributed record is counted
    under ``UNATTRIBUTED`` rather than silently dropped.
    """

    counts: dict[str, int] = {}
    for rejection in rejections:
        key = "/".join(str(rejection.get(field) or "UNATTRIBUTED") for field in fields)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def equity_metrics(points: list[dict]) -> dict:
    if len(points) < 2:
        raise ValueError("need starting equity and at least one endpoint")
    points = sorted(points, key=lambda p: p["ts_ns"])
    curve = pd.Series(
        [p["equity"] for p in points], index=[p["ts_ns"] for p in points], dtype=float
    )
    if not np.isfinite(curve).all() or curve.iloc[0] <= 0:
        raise ValueError("nonfinite curve or invalid starting equity")
    start, end = float(curve.iloc[0]), float(curve.iloc[-1])
    days = (points[-1]["ts_ns"] - points[0]["ts_ns"]) / DAY_NS
    total = end / start - 1
    cagr = math.expm1(math.log(end / start) * 365 / days) if days > 0 and end > 0 else None
    dd = curve / curve.cummax() - 1
    last_peak, duration = points[0]["ts_ns"], 0.0
    underwater = False
    for ts, drawdown in dd.items():
        if drawdown < -1e-12 or underwater:
            duration = max(duration, (ts - last_peak) / DAY_NS)
        if drawdown >= -1e-12:
            last_peak = ts
        underwater = drawdown < -1e-12
    # Include the initial capital before first day's costs. Days are assigned by
    # endpoint: exactly-midnight points belong to the preceding completed day.
    daily = {}
    for point in points[1:]:
        daily[(point["ts_ns"] - 1) // DAY_NS] = point["equity"]
    daily_curve = pd.Series([start, *daily.values()], dtype=float)
    returns = daily_curve.pct_change().dropna().to_numpy()
    contiguous = all(b - a == 1 for a, b in zip(daily, list(daily)[1:]))
    sharpe = sortino = None
    if len(returns) >= 2 and contiguous and np.isfinite(returns).all():
        std = float(np.std(returns, ddof=1))
        downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
        sharpe = float(np.mean(returns) / std * np.sqrt(365)) if std > 0 else None
        sortino = float(np.mean(returns) / downside * np.sqrt(365)) if downside > 0 else None
    max_dd = float(dd.min())
    return {
        "start_equity": start,
        "end_equity": end,
        "total_return": total,
        "elapsed_days": days,
        "cagr": cagr,
        "annualized_low_confidence": days < 365,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": duration,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": cagr / abs(max_dd) if cagr is not None and max_dd < 0 else None,
        "net_pnl": end - start,
        "daily_return_count": len(returns),
        "suppressed_metrics": [
            k
            for k, v in {
                "cagr": cagr,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": cagr / abs(max_dd) if cagr is not None and max_dd < 0 else None,
            }.items()
            if v is None
        ],
        "ratio_basis": "365 days/year, zero risk-free rate; >=2 consecutive daily returns",
    }


def round_trips(fills: list[dict], funding: list[dict]) -> list[dict]:
    """Average-cost, flat-to-flat lifecycles, splitting reversal fills proportionally."""
    states, trades = {}, []
    events = [(f["ts_ns"], 1, f) for f in fills] + [(f["ts_ns"], 0, f) for f in funding]
    for _, kind, event in sorted(events, key=lambda e: (e[0], e[1], e[2]["symbol"])):
        symbol = event["symbol"]
        if kind == 0:
            if symbol in states:
                states[symbol]["funding"] -= event["cash_flow"]
            continue
        signed = event["qty"] * (1 if event["side"] == "BUY" else -1)
        remaining = abs(signed)
        direction = 1 if signed > 0 else -1
        while remaining > 1e-10:
            state = states.get(symbol)
            if state is None:
                state = {
                    "symbol": symbol,
                    "direction": "long" if direction > 0 else "short",
                    "sign": direction,
                    "units": 0.0,
                    "average_price": 0.0,
                    "opened_ns": event["ts_ns"],
                    "gross_pnl_at_fills": 0.0,
                    "commission": 0.0,
                    "spread": 0.0,
                    "slippage": 0.0,
                    "funding": 0.0,
                }
                states[symbol] = state
            adding = state["sign"] == direction
            units = remaining if adding else min(remaining, state["units"])
            for cost in ("commission", "spread", "slippage"):
                state[cost] += event[cost] * units / event["qty"]
            if adding:
                state["average_price"] = (
                    state["average_price"] * state["units"] + event["price"] * units
                ) / (state["units"] + units)
                state["units"] += units
            else:
                state["gross_pnl_at_fills"] += (
                    state["sign"] * units * (event["price"] - state["average_price"])
                )
                state["units"] -= units
            remaining -= units
            if state["units"] < 1e-10:
                state["closed_ns"] = event["ts_ns"]
                state["holding_hours"] = (event["ts_ns"] - state["opened_ns"]) / 3.6e12
                state["net_pnl"] = (
                    state["gross_pnl_at_fills"] - state["commission"] - state["funding"]
                )
                state["gross_pnl"] = (
                    state["gross_pnl_at_fills"] + state["spread"] + state["slippage"]
                )
                trades.append(state)
                del states[symbol]
    if states:
        raise ValueError("incomplete trade lifecycles at window boundary")
    return trades


def trade_metrics(trades: list[dict]) -> dict:
    pnl = [t["net_pnl"] for t in trades]
    wins = sum(p > 0 for p in pnl)
    losses = -sum(p for p in pnl if p < 0)
    return {
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else None,
        "profit_factor": sum(p for p in pnl if p > 0) / losses if losses else None,
        "expectancy_per_trade": sum(pnl) / len(trades) if trades else None,
        "average_holding_hours": float(np.mean([t["holding_hours"] for t in trades]))
        if trades
        else None,
        "net_pnl": sum(pnl),
    }


def summarize(result: dict) -> dict:
    points, fills, funding = result["observations"], result["fills"], result["funding"]
    metrics = equity_metrics(points)
    trades = round_trips(fills, funding)
    metrics.update({k: v for k, v in trade_metrics(trades).items() if k != "net_pnl"})
    costs = {key: sum(f[key] for f in fills) for key in ("commission", "spread", "slippage")}
    costs["funding"] = -sum(f["cash_flow"] for f in funding)
    metrics["costs"] = costs
    metrics["gross_pnl"] = metrics["net_pnl"] + sum(costs.values())
    metrics["turnover_notional"] = sum(f["notional"] for f in fills)
    metrics["turnover_over_start_equity"] = metrics["turnover_notional"] / metrics["start_equity"]
    metrics["fills"] = len(fills)
    metrics["rejected_orders"] = len(result["rejections"])
    metrics["rejections_by_reason"] = rejection_counts(result["rejections"], "reason")
    metrics["rejections_by_symbol"] = rejection_counts(result["rejections"], "symbol")
    metrics["rejections_by_symbol_reason"] = rejection_counts(
        result["rejections"], "symbol", "reason"
    )
    metrics["accounting_residual"] = metrics["net_pnl"] - sum(t["net_pnl"] for t in trades)
    # Nautilus cash is rounded to settlement currency precision; retain/report residual.
    tolerance = 0.02 * (len(fills) + len(funding) + 1)
    if abs(metrics["accounting_residual"]) > tolerance:
        raise ValueError(f"unreconciled backtest accounting: {metrics['accounting_residual']}")
    metrics["long_short"] = {
        side: trade_metrics([t for t in trades if t["direction"] == side])
        for side in ("long", "short")
    }
    metrics["per_symbol"] = {}
    for symbol in sorted({f["symbol"] for f in fills}):
        sub = [f for f in fills if f["symbol"] == symbol]
        symbol_costs = {k: sum(f[k] for f in sub) for k in ("commission", "spread", "slippage")}
        symbol_costs["funding"] = -sum(f["cash_flow"] for f in funding if f["symbol"] == symbol)
        metrics["per_symbol"][symbol] = {
            **trade_metrics([t for t in trades if t["symbol"] == symbol]),
            "costs": symbol_costs,
            "turnover_notional": sum(f["notional"] for f in sub),
        }
    # MTM period attribution includes open-trade PnL; do not bucket entire trades by exit month.
    buckets = {"monthly": {}, "yearly": {}, "market_regime": {}}
    durations = []
    for a, b in itertools.pairwise(points):
        duration = (b["ts_ns"] - a["ts_ns"]) / 1e9
        durations.append((duration, a))
        moment = pd.Timestamp(b["ts_ns"] - 1, unit="ns", tz="UTC")
        for category, key in (
            ("monthly", moment.strftime("%Y-%m")),
            ("yearly", str(moment.year)),
            ("market_regime", a["regime"]),
        ):
            bucket = buckets[category].setdefault(
                key, {"net_pnl": 0.0, "start_equity": a["equity"], "end_equity": b["equity"]}
            )
            bucket["net_pnl"] += b["equity"] - a["equity"]
            bucket["end_equity"] = b["equity"]
    for category in ("monthly", "yearly"):
        for bucket in buckets[category].values():
            bucket["return"] = bucket["end_equity"] / bucket["start_equity"] - 1
    metrics.update(buckets)
    elapsed = sum(d for d, _ in durations)
    metrics["exposure"] = {
        "fraction_time_in_market": sum(d for d, p in durations if p["n_open"] > 0) / elapsed,
        "average_concurrent_positions": sum(d * p["n_open"] for d, p in durations) / elapsed,
        "max_concurrent_positions": max(p["n_open"] for p in points),
        "average_gross_exposure": sum(d * (p["gross_exposure"] or 0) for d, p in durations)
        / elapsed,
        "max_gross_exposure": max(p["gross_exposure"] or 0 for p in points),
    }
    metrics["funding_mark_approximations"] = sum(
        f["approximate_mark"] and f["units"] != 0 for f in funding
    )
    return metrics
