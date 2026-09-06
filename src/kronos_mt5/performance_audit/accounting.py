"""Independent PnL reconciliation.

Sign conventions used throughout (documented because the sources disagree):

  * `income.amount` is Binance's own signed ledger. REALIZED_PNL is signed,
    COMMISSION is NEGATIVE (a cost), FUNDING_FEE is signed (received +, paid -).
  * `equity.commissions` is the companion's snapshot field and stores commission
    as a POSITIVE cost (it is `-COMMISSION income`).
  * `fills.commission` is the per-fill commission as reported on the fill event,
    POSITIVE, and only covers fills this process observed.
  * `fills.slippage` is implementation shortfall in QUOTE CURRENCY (USDT), where
    POSITIVE means the fill was worse than the reference price. It is a
    measurement embedded in the execution price, never a separate cash expense,
    so it must NOT be added into the accounting identity.

The identity checked here is the companion's own:

    equity_change = realized - commissions + funding + other_income
                    + unrealized_change + residual
"""

from __future__ import annotations

from itertools import pairwise

from kronos_mt5.performance_audit.loading import as_float, parse_ts

# Absolute tolerance in quote currency. Below this a residual is noise from
# float accumulation over ~2k ledger rows; above it something is unexplained.
RESIDUAL_ABS_TOLERANCE = 1.0
# ... or this fraction of starting equity, whichever is larger.
RESIDUAL_REL_TOLERANCE = 0.001
# A single-observation equity jump larger than this fraction of equity looks like
# a deposit, withdrawal or account reset rather than trading PnL.
DISCONTINUITY_PCT = 5.0

ATTRIBUTED_INCOME = ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE")


def summarize_income(rows: list[dict]) -> dict:
    by_type: dict[str, dict] = {}
    duplicates = 0
    seen: set = set()
    unparsable = 0
    first_ts = None
    last_ts = None
    for row in rows:
        income_id = row.get("income_id")
        if income_id is not None:
            if income_id in seen:
                duplicates += 1
            seen.add(income_id)
        moment = parse_ts(row.get("ts"))
        if moment is None:
            unparsable += 1
        else:
            first_ts = moment if first_ts is None or moment < first_ts else first_ts
            last_ts = moment if last_ts is None or moment > last_ts else last_ts
        kind = str(row.get("income_type") or "UNKNOWN")
        amount = as_float(row.get("amount")) or 0.0
        entry = by_type.setdefault(kind, {"count": 0, "total": 0.0})
        entry["count"] += 1
        entry["total"] += amount
    return {
        "rows": len(rows),
        "by_type": {
            k: {"count": v["count"], "total": v["total"]} for k, v in sorted(by_type.items())
        },
        "duplicate_income_ids": duplicates,
        "unparsable_timestamps": unparsable,
        "first_ts": first_ts.isoformat() if first_ts else None,
        "last_ts": last_ts.isoformat() if last_ts else None,
    }


def find_discontinuities(daily: list[dict], threshold_pct: float = DISCONTINUITY_PCT) -> list[dict]:
    """Day-over-day equity jumps too large to be ordinary trading PnL."""
    out = []
    for previous, current in pairwise(daily):
        before = previous["equity"]
        after = current["equity"]
        if before <= 0:
            continue
        change_pct = (after / before - 1.0) * 100.0
        if abs(change_pct) >= threshold_pct:
            out.append(
                {
                    "from_date": previous["date"],
                    "to_date": current["date"],
                    "equity_before": before,
                    "equity_after": after,
                    "change": after - before,
                    "change_pct": change_pct,
                    "likely_explanation": (
                        "deposit / withdrawal / account reset, or a genuine large "
                        "trading day — inspect the income ledger for that date"
                    ),
                }
            )
    return out


def reconcile(
    equity_rows: list[dict],
    income_rows: list[dict],
    fills_rows: list[dict],
    daily: list[dict],
) -> dict:
    """Recompute the accounting identity from raw rows and report the residual.

    The values are never forced to agree: the residual, the tolerance used and
    the likely explanations are reported instead.
    """
    income = summarize_income(income_rows)
    by_type = income["by_type"]
    realized = by_type.get("REALIZED_PNL", {}).get("total", 0.0)
    commission_income = by_type.get("COMMISSION", {}).get("total", 0.0)
    commissions_cost = -commission_income  # positive == cost
    funding = by_type.get("FUNDING_FEE", {}).get("total", 0.0)
    other_income = sum(v["total"] for k, v in by_type.items() if k not in ATTRIBUTED_INCOME)

    usable = [
        r
        for r in equity_rows
        if parse_ts(r.get("ts")) is not None and (as_float(r.get("equity")) or 0) > 0
    ]
    usable.sort(key=lambda r: parse_ts(r["ts"]))
    result: dict = {
        "available": bool(usable),
        "income": income,
        "sign_conventions": {
            "income.amount": "Binance ledger; COMMISSION negative, FUNDING_FEE signed",
            "equity.commissions": "positive == cost",
            "fills.commission": "positive == cost, observed fills only",
            "fills.slippage": (
                "implementation shortfall in QUOTE CURRENCY; positive == worse than "
                "reference; embedded in the execution price, NOT a separate expense"
            ),
        },
    }
    if not usable:
        result["note"] = "no usable equity rows to reconcile against"
        return result

    first, last = usable[0], usable[-1]
    start_equity = as_float(first.get("equity")) or 0.0
    end_equity = as_float(last.get("equity")) or 0.0
    start_cash = as_float(first.get("cash"))
    end_cash = as_float(last.get("cash"))
    start_unrealized = as_float(first.get("unrealized")) or 0.0
    end_unrealized = as_float(last.get("unrealized")) or 0.0

    equity_change = end_equity - start_equity
    unrealized_change = end_unrealized - start_unrealized
    explained = realized - commissions_cost + funding + other_income + unrealized_change
    residual = equity_change - explained
    tolerance = max(RESIDUAL_ABS_TOLERANCE, abs(start_equity) * RESIDUAL_REL_TOLERANCE)

    # What the bot itself recorded, for comparison with our independent figures.
    recorded = {
        "realized": as_float(last.get("realized")),
        "commissions": as_float(last.get("commissions")),
        "funding": as_float(last.get("funding")),
        "slippage": as_float(last.get("slippage")),
        "total_pnl": as_float(last.get("total_pnl")),
    }

    fills_commission = sum(
        as_float(r.get("commission")) or 0.0
        for r in fills_rows
        if as_float(r.get("commission")) is not None
    )
    fills_with_commission = sum(1 for r in fills_rows if as_float(r.get("commission")) is not None)
    fills_slippage = sum(
        as_float(r.get("slippage")) or 0.0
        for r in fills_rows
        if as_float(r.get("slippage")) is not None
    )
    fills_with_slippage = sum(1 for r in fills_rows if as_float(r.get("slippage")) is not None)

    result.update(
        {
            "start_ts": parse_ts(first["ts"]).isoformat(),
            "end_ts": parse_ts(last["ts"]).isoformat(),
            "start_equity": start_equity,
            "end_equity": end_equity,
            "equity_change": equity_change,
            "start_cash": start_cash,
            "end_cash": end_cash,
            "cash_change": (
                end_cash - start_cash if (start_cash is not None and end_cash is not None) else None
            ),
            "start_unrealized": start_unrealized,
            "end_unrealized": end_unrealized,
            "unrealized_change": unrealized_change,
            "realized_pnl": realized,
            "commissions_cost": commissions_cost,
            "funding": funding,
            "other_income": other_income,
            "explained_change": explained,
            "residual": residual,
            "residual_tolerance": tolerance,
            "reconciled_within_tolerance": abs(residual) <= tolerance,
            "recorded_by_bot": recorded,
            "recorded_vs_recomputed": {
                "realized_delta": (
                    recorded["realized"] - realized if recorded["realized"] is not None else None
                ),
                "commissions_delta": (
                    recorded["commissions"] - commissions_cost
                    if recorded["commissions"] is not None
                    else None
                ),
                "funding_delta": (
                    recorded["funding"] - funding if recorded["funding"] is not None else None
                ),
                "total_pnl_delta": (
                    recorded["total_pnl"] - equity_change
                    if recorded["total_pnl"] is not None
                    else None
                ),
            },
            "fills_commission_total": fills_commission,
            "fills_with_commission": fills_with_commission,
            "fills_commission_vs_income": fills_commission - commissions_cost,
            "fills_slippage_total_quote": fills_slippage,
            "fills_with_slippage": fills_with_slippage,
            "discontinuities": find_discontinuities(daily),
            "likely_residual_explanations": _residual_explanations(
                residual, tolerance, income, fills_rows
            ),
        }
    )
    return result


def _residual_explanations(
    residual: float, tolerance: float, income: dict, fills_rows: list[dict]
) -> list[str]:
    if abs(residual) <= tolerance:
        return [
            "residual is within tolerance and consistent with float accumulation across the ledger"
        ]
    reasons = [
        (
            "income ledger may not cover the full equity window (check the first "
            "and last income timestamps against the equity range)"
        ),
        (
            "transfers, deposits or withdrawals are not part of trading PnL and "
            "appear as unexplained equity change"
        ),
        "an account reset would restart equity without a matching ledger entry",
    ]
    if income.get("duplicate_income_ids"):
        reasons.append("duplicate income ids were found and may be double counted")
    if any(r.get("reconciliation") for r in fills_rows):
        reasons.append(
            "venue-reconciled fills exist; their PnL is present in the ledger but "
            "their decision metadata is not, which can skew per-fill attribution"
        )
    return reasons


def detect_regimes(ops_rows: list[dict], daily: list[dict]) -> dict:
    """Service starts/stops that split the sample into separate runs.

    A restart can accompany a configuration change, so the audit must not silently
    treat the whole window as one continuous strategy.
    """
    starts = []
    stops = []
    for row in ops_rows:
        moment = parse_ts(row.get("ts"))
        if moment is None:
            continue
        kind = str(row.get("kind") or "")
        if kind == "BOT_START":
            starts.append(moment)
        elif kind == "BOT_STOP":
            stops.append(moment)
    starts.sort()
    stops.sort()
    boundary_days = sorted({m.date().isoformat() for m in starts})
    return {
        "starts": len(starts),
        "stops": len(stops),
        "first_start": starts[0].isoformat() if starts else None,
        "last_start": starts[-1].isoformat() if starts else None,
        "restart_days": boundary_days,
        "distinct_restart_days": len(boundary_days),
        "continuous_single_run": len(starts) <= 1,
        "note": (
            "each restart is a potential configuration change; treat segments "
            "between restarts as possibly different strategy regimes"
        ),
    }
