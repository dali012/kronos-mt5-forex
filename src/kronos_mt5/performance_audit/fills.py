"""Fill and execution attribution.

`fills.slippage` semantics were verified against the producer
(`companion/recorder.py` -> `execution.implementation_shortfall_quote`) and
empirically against exported rows:

    BUY : (fill_price - reference_price) * qty
    SELL: (reference_price - fill_price) * qty

so it is signed **quote currency** (USDT), POSITIVE meaning the fill was worse
than the reference price. It is NOT basis points and NOT per-unit. It is already
embedded in the execution price, so it is reported as a measurement and never
added to the cash accounting.

Execution telemetry columns (`liquidity`, `exec_role`, `impl_shortfall_bps`, ...)
were added later; a database written before that migration simply lacks them and
every derived metric is reported as unavailable.
"""

from __future__ import annotations

from kronos_mt5.performance_audit.loading import as_float, parse_ts

TELEMETRY_COLUMNS = (
    "order_type",
    "liquidity",
    "exec_role",
    "decision_price",
    "limit_price",
    "decision_to_fill_ms",
    "fallback_reason",
    "adverse_drift_bps",
    "impl_shortfall_quote",
    "impl_shortfall_bps",
)

SLIPPAGE_UNIT = "quote_currency_signed_positive_is_worse"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _blank_group() -> dict:
    return {
        "fills": 0,
        "quantity": 0.0,
        "quote_turnover": 0.0,
        "commission": 0.0,
        "slippage_quote": 0.0,
        "reconciliation_fills": 0,
        "missing_kind": 0,
        "missing_reference_price": 0,
        "missing_slippage": 0,
        "missing_commission": 0,
        "missing_trade_id": 0,
        "liquidity": {},
        "exec_role": {},
        "_shortfall_bps": [],
        "_shortfall_weights": [],
    }


def analyze_fills(rows: list[dict], available_columns: list[str] | None = None) -> dict:
    """Group execution metrics by symbol, side and kind."""
    available = set(available_columns or (rows[0].keys() if rows else []))
    telemetry_present = sorted(c for c in TELEMETRY_COLUMNS if c in available)
    telemetry_missing = sorted(c for c in TELEMETRY_COLUMNS if c not in available)

    groups: dict[tuple, dict] = {}
    totals = _blank_group()
    first_ts = None
    last_ts = None
    duplicate_ids = 0
    seen_ids: set = set()
    unparsable_ts = 0

    for row in rows:
        fill_id = row.get("fill_id")
        if fill_id is not None:
            if fill_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(fill_id)
        moment = parse_ts(row.get("ts"))
        if moment is None:
            unparsable_ts += 1
        else:
            first_ts = moment if first_ts is None or moment < first_ts else first_ts
            last_ts = moment if last_ts is None or moment > last_ts else last_ts

        symbol = str(row.get("symbol") or "UNKNOWN")
        side = str(row.get("side") or "UNKNOWN")
        kind_raw = row.get("kind")
        kind = str(kind_raw) if kind_raw not in (None, "") else "UNKNOWN"
        key = (symbol, side, kind)
        group = groups.setdefault(key, _blank_group())

        qty = abs(as_float(row.get("qty")) or 0.0)
        price = as_float(row.get("price")) or 0.0
        commission = as_float(row.get("commission"))
        slippage = as_float(row.get("slippage"))
        reference = as_float(row.get("reference_price"))

        for target in (group, totals):
            target["fills"] += 1
            target["quantity"] += qty
            target["quote_turnover"] += abs(qty * price)
            if commission is None:
                target["missing_commission"] += 1
            else:
                target["commission"] += commission
            if slippage is None:
                target["missing_slippage"] += 1
            else:
                target["slippage_quote"] += slippage
            if reference is None:
                target["missing_reference_price"] += 1
            if kind_raw in (None, ""):
                target["missing_kind"] += 1
            if not row.get("trade_id"):
                target["missing_trade_id"] += 1
            if row.get("reconciliation"):
                target["reconciliation_fills"] += 1
            if "liquidity" in available:
                label = str(row.get("liquidity") or "UNKNOWN")
                target["liquidity"][label] = target["liquidity"].get(label, 0) + 1
            if "exec_role" in available:
                label = str(row.get("exec_role") or "UNKNOWN")
                target["exec_role"][label] = target["exec_role"].get(label, 0) + 1
            if "impl_shortfall_bps" in available:
                bps = as_float(row.get("impl_shortfall_bps"))
                if bps is not None:
                    target["_shortfall_bps"].append(bps)
                    target["_shortfall_weights"].append(abs(qty * price))

    def finish(group: dict) -> dict:
        bps = group.pop("_shortfall_bps")
        weights = group.pop("_shortfall_weights")
        out = dict(group)
        out["avg_shortfall_bps"] = (sum(bps) / len(bps)) if bps else None
        weight_total = sum(weights)
        out["weighted_shortfall_bps"] = (
            sum(v * w for v, w in zip(bps, weights)) / weight_total if weight_total else None
        )
        out["median_shortfall_bps"] = _median(bps)
        out["shortfall_sample"] = len(bps)
        if not out["liquidity"]:
            out.pop("liquidity")
        if not out["exec_role"]:
            out.pop("exec_role")
        return out

    grouped = [
        {"symbol": symbol, "side": side, "kind": kind, **finish(group)}
        for (symbol, side, kind), group in sorted(groups.items())
    ]
    total = finish(totals)

    by_symbol: dict[str, dict] = {}
    for entry in grouped:
        agg = by_symbol.setdefault(
            entry["symbol"],
            {"fills": 0, "quote_turnover": 0.0, "commission": 0.0, "slippage_quote": 0.0},
        )
        agg["fills"] += entry["fills"]
        agg["quote_turnover"] += entry["quote_turnover"]
        agg["commission"] += entry["commission"]
        agg["slippage_quote"] += entry["slippage_quote"]

    return {
        "available": bool(rows),
        "total_fills": len(rows),
        "first_ts": first_ts.isoformat() if first_ts else None,
        "last_ts": last_ts.isoformat() if last_ts else None,
        "duplicate_fill_ids": duplicate_ids,
        "unparsable_timestamps": unparsable_ts,
        "slippage_unit": SLIPPAGE_UNIT,
        "slippage_unit_note": (
            "signed quote currency (USDT); positive means worse than the reference "
            "price; embedded in the execution price, not a separate cash expense"
        ),
        "telemetry_columns_present": telemetry_present,
        "telemetry_columns_missing": telemetry_missing,
        "execution_telemetry_available": not telemetry_missing,
        "totals": total,
        "by_symbol": dict(sorted(by_symbol.items())),
        "groups": grouped,
    }
