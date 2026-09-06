"""Explicit exchange filter snapshots, Decimal checks and linear perpetual instruments."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity


def precision(value: str) -> int:
    return max(0, -Decimal(value).normalize().as_tuple().exponent)


def rounded(value: float, step: str, *, up: bool = False) -> float:
    increment = Decimal(step)
    return float(
        (Decimal(str(value)) / increment).to_integral_value(
            rounding=ROUND_CEILING if up else ROUND_FLOOR
        )
        * increment
    )


def reject_reason(
    qty: float, price: float, filters: dict, *, reduce_only: bool = False
) -> str | None:
    q, p = Decimal(str(qty)), Decimal(str(price))
    if not q.is_finite() or not p.is_finite() or q <= 0 or p <= 0:
        return "nonpositive_or_nonfinite"
    if q % Decimal(filters["step_size"]):
        return "quantity_precision"
    if p % Decimal(filters["tick_size"]):
        return "price_precision"
    if q < Decimal(filters["min_qty"]) or q > Decimal(filters["max_qty"]):
        return "quantity_limits"
    if not reduce_only and q * p < Decimal(filters["min_notional"]):
        return "minimum_notional"
    return None


def exchange_filters(payload: dict, symbols: list[str]) -> dict:
    out = {}
    for item in payload["symbols"]:
        if item["symbol"] not in symbols:
            continue
        if item["contractType"] != "PERPETUAL" or item["quoteAsset"] != "USDT":
            raise ValueError("expected USDT linear perpetual")
        f = {f["filterType"]: f for f in item["filters"]}
        lot = f.get("MARKET_LOT_SIZE", f["LOT_SIZE"])
        out[item["symbol"]] = {
            "tick_size": f["PRICE_FILTER"]["tickSize"],
            "step_size": lot["stepSize"],
            "min_qty": lot["minQty"],
            "max_qty": lot["maxQty"],
            "min_notional": f["MIN_NOTIONAL"]["notional"],
        }
    if set(out) != set(symbols):
        raise ValueError("exchangeInfo missing requested symbols")
    return out


def instrument(symbol: str, filters: dict, fee_bps: float, leverage: float) -> CryptoPerpetual:
    pp, sp = precision(filters["tick_size"]), precision(filters["step_size"])
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str(f"{symbol}.BINANCE"),
        raw_symbol=Symbol(symbol),
        base_currency=Currency.from_str(symbol[:-4]),
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=pp,
        size_precision=sp,
        price_increment=Price(float(filters["tick_size"]), pp),
        size_increment=Quantity(float(filters["step_size"]), sp),
        min_quantity=Quantity(float(filters["min_qty"]), sp),
        max_quantity=Quantity(float(filters["max_qty"]), sp),
        min_notional=Money(filters["min_notional"], USDT),
        margin_init=Decimal(str(1 / leverage)),
        margin_maint=Decimal("0.005"),
        maker_fee=Decimal(str(fee_bps / 1e4)),
        taker_fee=Decimal(str(fee_bps / 1e4)),
        ts_event=0,
        ts_init=0,
    )
