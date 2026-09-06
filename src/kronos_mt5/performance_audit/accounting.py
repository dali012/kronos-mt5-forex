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


# Reconciliation window. A record is in scope when
#     first_equity_ts < record_ts <= last_equity_ts
# The start is EXCLUSIVE because the first equity observation is the opening
# balance: anything stamped at or before it is already inside that balance and
# would be double counted. The end is INCLUSIVE because the last observation is
# the closing balance and must contain everything up to it.
WINDOW_RULE = "first_equity_ts < record_ts <= last_equity_ts"

INCOME_COMPARE_FIELDS = ("ts", "income_type", "amount", "symbol", "asset", "trade_id")
FILL_COMPARE_FIELDS = ("ts", "symbol", "side", "qty", "price", "commission", "slippage")


def _canonical(row: dict, fields: tuple) -> tuple:
    """A hashable, order-independent signature of the fields that matter."""
    return tuple(("" if row.get(f) is None else str(row.get(f))) for f in fields)


def dedupe_rows(rows: list[dict], id_field: str, compare_fields: tuple) -> dict:
    """Collapse exact duplicates by id; flag ids whose content conflicts.

    Deterministic regardless of database row order: groups are keyed by id and
    the surviving row is the one with the lexicographically smallest signature.
    A conflicting id is NOT silently resolved — the caller marks the affected
    reconciliation unreliable.
    """
    groups: dict[str, dict] = {}
    unkeyed: list[dict] = []
    for row in rows:
        key = row.get(id_field)
        if key is None or key == "":
            unkeyed.append(row)
            continue
        signature = _canonical(row, compare_fields)
        bucket = groups.setdefault(str(key), {"signatures": {}, "count": 0})
        bucket["count"] += 1
        bucket["signatures"].setdefault(signature, row)

    unique: list[dict] = []
    exact_duplicates = 0
    conflicting: list[str] = []
    for key in sorted(groups):
        bucket = groups[key]
        signatures = bucket["signatures"]
        exact_duplicates += bucket["count"] - len(signatures)
        if len(signatures) > 1:
            conflicting.append(key)
        chosen = signatures[min(signatures)]
        unique.append(chosen)
    unique.extend(unkeyed)
    return {
        "unique": unique,
        "input_rows": len(rows),
        "unique_rows": len(unique),
        "rows_without_id": len(unkeyed),
        "exact_duplicates": exact_duplicates,
        "conflicting_ids": sorted(conflicting),
        "conflicting_id_count": len(conflicting),
    }


def partition_by_window(rows: list[dict], ts_field: str, start, end) -> dict:
    """Split rows into included / before / after / invalid for the window."""
    included, before, after, invalid = [], [], [], []
    for row in rows:
        moment = parse_ts(row.get(ts_field))
        if moment is None:
            invalid.append(row)
        elif moment <= start:
            before.append(row)
        elif moment > end:
            after.append(row)
        else:
            included.append(row)
    return {"included": included, "before": before, "after": after, "invalid": invalid}


def _amounts_by_type(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    for row in rows:
        kind = str(row.get("income_type") or "UNKNOWN")
        out[kind] = out.get(kind, 0.0) + (as_float(row.get("amount")) or 0.0)
    return dict(sorted(out.items()))


def _sum_field(rows: list[dict], field: str) -> tuple[float, int]:
    total = 0.0
    counted = 0
    for row in rows:
        value = as_float(row.get(field))
        if value is not None:
            total += value
            counted += 1
    return total, counted


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
    usable = [
        r
        for r in equity_rows
        if parse_ts(r.get("ts")) is not None and (as_float(r.get("equity")) or 0) > 0
    ]
    usable.sort(key=lambda r: parse_ts(r["ts"]))
    result: dict = {
        "available": bool(usable),
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
        result["income"] = summarize_income(income_rows)
        result["note"] = "no usable equity rows to reconcile against"
        return result

    first, last = usable[0], usable[-1]
    window_start = parse_ts(first["ts"])
    window_end = parse_ts(last["ts"])

    # --- scope every ledger to the equity window, de-duplicated ------------
    income_dedupe = dedupe_rows(income_rows, "income_id", INCOME_COMPARE_FIELDS)
    income_parts = partition_by_window(income_dedupe["unique"], "ts", window_start, window_end)
    fills_dedupe = dedupe_rows(fills_rows, "fill_id", FILL_COMPARE_FIELDS)
    fills_parts = partition_by_window(fills_dedupe["unique"], "ts", window_start, window_end)
    windowed_income = income_parts["included"]
    windowed_fills = fills_parts["included"]

    income = summarize_income(windowed_income)
    by_type = income["by_type"]
    realized = by_type.get("REALIZED_PNL", {}).get("total", 0.0)
    commission_income = by_type.get("COMMISSION", {}).get("total", 0.0)
    commissions_cost = -commission_income  # positive == cost
    funding = by_type.get("FUNDING_FEE", {}).get("total", 0.0)
    other_income = sum(v["total"] for k, v in by_type.items() if k not in ATTRIBUTED_INCOME)
    result["income"] = income

    conflicting = income_dedupe["conflicting_id_count"] + fills_dedupe["conflicting_id_count"]
    result["window"] = {
        "rule": WINDOW_RULE,
        "start_exclusive": window_start.isoformat(),
        "end_inclusive": window_end.isoformat(),
    }
    result["income_window"] = {
        "included": len(windowed_income),
        "before_window": len(income_parts["before"]),
        "after_window": len(income_parts["after"]),
        "invalid_timestamp": len(income_parts["invalid"]),
        "excluded_amounts_by_type": {
            "before_window": _amounts_by_type(income_parts["before"]),
            "after_window": _amounts_by_type(income_parts["after"]),
            "invalid_timestamp": _amounts_by_type(income_parts["invalid"]),
        },
        "deduplication": {k: v for k, v in income_dedupe.items() if k != "unique"},
    }
    before_commission, _ = _sum_field(fills_parts["before"], "commission")
    after_commission, _ = _sum_field(fills_parts["after"], "commission")
    invalid_commission, _ = _sum_field(fills_parts["invalid"], "commission")
    before_slippage, _ = _sum_field(fills_parts["before"], "slippage")
    after_slippage, _ = _sum_field(fills_parts["after"], "slippage")
    invalid_slippage, _ = _sum_field(fills_parts["invalid"], "slippage")
    result["fills_window"] = {
        "included": len(windowed_fills),
        "before_window": len(fills_parts["before"]),
        "after_window": len(fills_parts["after"]),
        "invalid_timestamp": len(fills_parts["invalid"]),
        "excluded_commission": {
            "before_window": before_commission,
            "after_window": after_commission,
            "invalid_timestamp": invalid_commission,
        },
        "excluded_slippage_quote": {
            "before_window": before_slippage,
            "after_window": after_slippage,
            "invalid_timestamp": invalid_slippage,
        },
        "deduplication": {k: v for k, v in fills_dedupe.items() if k != "unique"},
    }
    result["conflicting_duplicate_ids"] = {
        "income": income_dedupe["conflicting_ids"],
        "fills": fills_dedupe["conflicting_ids"],
        "total": conflicting,
    }
    result["reconciliation_reliable"] = conflicting == 0
    result["totals_provisional"] = conflicting > 0
    if conflicting:
        result["reconciliation_unreliable_reason"] = (
            f"{conflicting} record id(s) appear more than once with CONFLICTING "
            f"content. One row per id was selected deterministically so diagnostic "
            f"totals could still be produced, but which row is correct is unknown, "
            f"so reconciliation validity is WITHHELD: every total below is "
            f"provisional and must not be trusted."
        )
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

    fills_commission, fills_with_commission = _sum_field(windowed_fills, "commission")
    fills_slippage, fills_with_slippage = _sum_field(windowed_fills, "slippage")

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
            # Withheld (None), never True/False, when a conflicting id means the
            # residual was computed from a provisionally selected row.
            "reconciled_within_tolerance": (None if conflicting else abs(residual) <= tolerance),
            "reconciliation_status": (
                "UNRELIABLE"
                if conflicting
                else ("RECONCILED" if abs(residual) <= tolerance else "RESIDUAL_EXCEEDS_TOLERANCE")
            ),
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
                residual, tolerance, income, windowed_fills, result
            ),
        }
    )
    return result


def _residual_explanations(
    residual: float,
    tolerance: float,
    income: dict,
    fills_rows: list[dict],
    context: dict | None = None,
) -> list[str]:
    context = context or {}
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
    excluded = (context.get("income_window") or {}).get("before_window", 0)
    if excluded:
        reasons.append(
            f"{excluded} income record(s) predate the first equity observation and "
            f"are excluded by the reconciliation window; their PnL is already inside "
            f"the opening balance"
        )
    if not context.get("reconciliation_reliable", True):
        reasons.append(context.get("reconciliation_unreliable_reason", ""))
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
