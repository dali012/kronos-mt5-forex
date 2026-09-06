"""Explicit exchange filter snapshots, Decimal checks and linear perpetual instruments.

Exchange precision is an exact decimal property, so every filter check here works
on :class:`~decimal.Decimal` values. Nautilus ``Price``/``Quantity`` carry a
lossless decimal representation via ``as_decimal()``; ``float(price)`` does not.
A Nautilus fixed-point value divided by its raw scale can land on a different
double than the shortest-round-trip literal, so ``str(float(price))`` may render
an exact ``88187.9`` as ``88187.90000000001`` and make a tick-aligned price look
off-tick. Never convert a ``Price`` or ``Quantity`` to float before validating it.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity


def precision(value: str) -> int:
    return max(0, -Decimal(value).normalize().as_tuple().exponent)


def exact(value) -> Decimal:
    """Return the exact decimal value of an exchange price/quantity/limit.

    Accepts ``Decimal``, Nautilus ``Price``/``Quantity`` (anything exposing
    ``as_decimal()``), ``str`` and ``int`` losslessly. ``float`` is supported for
    external callers only and is interpreted through its shortest round-trip
    repr, which is exact *only* when the float was produced from a canonical
    decimal string. Internal execution validation must pass ``Price``/``Quantity``
    objects so that no float ever reaches a precision comparison.
    """

    if isinstance(value, Decimal):
        return value
    as_decimal = getattr(value, "as_decimal", None)
    if callable(as_decimal):
        return as_decimal()
    if isinstance(value, bool):
        raise TypeError("bool is not an exchange value")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    raise TypeError(f"unsupported exchange value type: {type(value).__name__}")


def rounded(value, step: str, *, up: bool = False) -> float:
    """Round ``value`` down (or up) onto ``step`` and return it as a float."""

    return float(rounded_decimal(value, step, up=up))


def rounded_decimal(value, step: str, *, up: bool = False) -> Decimal:
    """Round ``value`` down (or up) onto ``step``, exactly, as a ``Decimal``."""

    increment = Decimal(step)
    return (exact(value) / increment).to_integral_value(
        rounding=ROUND_CEILING if up else ROUND_FLOOR
    ) * increment


def exact_price(inst, value) -> Price:
    """Build a ``Price`` exactly equal to ``value``, without a float round-trip.

    Nautilus accepts a ``Decimal`` directly, so no float is involved. The result
    is still compared back to the intended decimal: a value needing more digits
    than the instrument's precision would otherwise be silently rounded, moving
    the price by up to a whole tick.
    """

    target = exact(value)
    price = Price(target, inst.price_precision)
    if price.as_decimal() != target:
        raise ValueError(
            f"price {target} is not representable at precision {inst.price_precision}: "
            f"got {price.as_decimal()}"
        )
    return price


def exact_quantity(inst, value) -> Quantity:
    """Build a ``Quantity`` exactly equal to ``value``, without a float round-trip."""

    target = exact(value)
    quantity = Quantity(target, inst.size_precision)
    if quantity.as_decimal() != target:
        raise ValueError(
            f"quantity {target} is not representable at precision {inst.size_precision}: "
            f"got {quantity.as_decimal()}"
        )
    return quantity


def tick_aligned_price(inst, value, tick: str, *, up: bool = False) -> Price:
    """Return a ``Price`` snapped onto ``tick``, built and verified exactly.

    Simulated quotes are submitted to execution validation, so a synthetic price
    that drifted off the tick would surface as a genuine exchange rejection.
    Both checks are needed: equality catches a value silently moved by a whole
    tick during construction, modulo catches one that is merely off-grid.
    """

    snapped = rounded_decimal(value, tick, up=up)
    price = exact_price(inst, snapped)
    if price.as_decimal() != snapped:
        raise ValueError(f"synthetic quote {price} does not equal snapped value {snapped}")
    if price.as_decimal() % Decimal(tick):
        raise ValueError(f"synthetic quote {price} is not aligned to tick size {tick}")
    return price


def step_aligned_quantity(inst, value, step: str, *, up: bool = False) -> Quantity:
    """Return a ``Quantity`` snapped onto ``step``, built and verified exactly."""

    snapped = rounded_decimal(value, step, up=up)
    quantity = exact_quantity(inst, snapped)
    if quantity.as_decimal() != snapped:
        raise ValueError(f"quantity {quantity} does not equal snapped value {snapped}")
    if quantity.as_decimal() % Decimal(step):
        raise ValueError(f"quantity {quantity} is not aligned to step size {step}")
    return quantity


def reject_reason(qty, price, filters: dict, *, reduce_only: bool = False) -> str | None:
    """Return the exchange rejection reason for an order, or ``None`` if valid.

    ``qty`` and ``price`` should be Nautilus ``Quantity``/``Price`` objects (or
    ``Decimal``/``str``) so the comparison uses exact exchange values.
    """

    try:
        q, p = exact(qty), exact(price)
    except (TypeError, ValueError, InvalidOperation):
        return "nonpositive_or_nonfinite"
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


def rejection_detail(
    symbol: str,
    ts_ns: int,
    reason: str,
    qty,
    price,
    filters: dict,
    *,
    decision_ts_ns: int | None = None,
) -> dict:
    """Build a fully attributed rejection record with exact attempted values."""

    def render(value):
        """Render an exact value as plain decimal text, never scientific."""

        try:
            return format(exact(value), "f")
        except (TypeError, ValueError, InvalidOperation):
            return None

    return {
        "symbol": symbol,
        "ts_ns": ts_ns,
        "reason": reason,
        "attempted_price": render(price),
        "attempted_qty": render(qty),
        "tick_size": filters["tick_size"],
        "step_size": filters["step_size"],
        "min_notional": filters["min_notional"],
        "decision_ts_ns": decision_ts_ns,
    }


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
