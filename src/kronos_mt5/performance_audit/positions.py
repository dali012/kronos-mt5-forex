"""Closed-position (trade) analysis, with an explicit lifecycle-coverage check.

Trade statistics are only meaningful if the `closed_positions` table actually
captured the position lifecycle. When a handful of closed positions sit behind
hundreds of fills, win rate / profit factor / expectancy describe a biased
sub-sample, not the strategy — so coverage is measured first and the metrics are
flagged unreliable rather than quietly reported.

Trades are NEVER reconstructed by pairing fills: fills carry no position id, and
naive pairing would invent trades that never existed.
"""

from __future__ import annotations

from kronos_mt5.performance_audit.loading import as_float, parse_ts

# Below this ratio of closed positions to fills, lifecycle coverage is treated as
# incomplete. Even a pure always-in-the-market strategy closes far more often.
COVERAGE_RATIO_FLOOR = 0.05


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def analyze_positions(rows: list[dict], total_fills: int) -> dict:
    count = len(rows)
    coverage_ratio = (count / total_fills) if total_fills else None
    incomplete = bool(
        total_fills > 0 and (coverage_ratio is None or coverage_ratio < COVERAGE_RATIO_FLOOR)
    )
    out: dict = {
        "available": count > 0,
        "closed_positions": count,
        "total_fills": total_fills,
        "coverage_ratio": coverage_ratio,
        "lifecycle_coverage_incomplete": incomplete,
        "metrics_reliable": bool(count > 0 and not incomplete),
        "reliability_note": (
            f"{count} closed positions recorded against {total_fills} fills "
            f"(ratio {coverage_ratio:.4f}). Win rate, profit factor and expectancy "
            f"below describe only the recorded sub-sample and MUST NOT be read as "
            f"strategy performance."
            if incomplete
            else None
        ),
    }
    if count == 0:
        out["note"] = "no closed positions recorded"
        return out

    pnls: list[float] = []
    r_multiples: list[float] = []
    holding_hours: list[float] = []
    exit_reasons: dict[str, int] = {}
    per_symbol: dict[str, dict] = {}
    missing_r = 0
    unparsable = 0

    for row in rows:
        pnl = as_float(row.get("net_pnl"))
        if pnl is not None:
            pnls.append(pnl)
        r_value = as_float(row.get("r_multiple"))
        if r_value is None:
            missing_r += 1
        else:
            r_multiples.append(r_value)
        opened = parse_ts(row.get("opened_ts"))
        closed = parse_ts(row.get("closed_ts"))
        if opened and closed:
            holding_hours.append((closed - opened).total_seconds() / 3600.0)
        elif opened is None or closed is None:
            unparsable += 1
        reason = str(row.get("exit_reason") or "UNKNOWN")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        symbol = str(row.get("symbol") or "UNKNOWN")
        bucket = per_symbol.setdefault(symbol, {"positions": 0, "net_pnl": 0.0, "wins": 0})
        bucket["positions"] += 1
        if pnl is not None:
            bucket["net_pnl"] += pnl
            if pnl > 0:
                bucket["wins"] += 1

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    out.update(
        {
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(pnls) - len(wins) - len(losses),
            "win_rate_pct": (len(wins) / len(pnls) * 100.0) if pnls else None,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": sum(pnls),
            "profit_factor": profit_factor,
            "expectancy": (sum(pnls) / len(pnls)) if pnls else None,
            "avg_pnl": (sum(pnls) / len(pnls)) if pnls else None,
            "median_pnl": _median(pnls),
            "avg_r_multiple": (sum(r_multiples) / len(r_multiples)) if r_multiples else None,
            "median_r_multiple": _median(r_multiples),
            "r_multiple_sample": len(r_multiples),
            "missing_r_multiple": missing_r,
            "avg_holding_hours": (
                sum(holding_hours) / len(holding_hours) if holding_hours else None
            ),
            "median_holding_hours": _median(holding_hours),
            "unparsable_timestamps": unparsable,
            "exit_reasons": dict(sorted(exit_reasons.items())),
            "per_symbol": {
                symbol: {
                    **stats,
                    "win_rate_pct": (
                        stats["wins"] / stats["positions"] * 100.0 if stats["positions"] else None
                    ),
                }
                for symbol, stats in sorted(per_symbol.items())
            },
            "trades_are_not_reconstructed_from_fills": True,
        }
    )
    return out
