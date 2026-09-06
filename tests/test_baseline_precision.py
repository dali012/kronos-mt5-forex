"""Exact-decimal exchange validation for the fixed-strategy baseline replay.

A Nautilus ``Price`` holds a fixed-point integer. ``float(price)`` divides that
raw value by its scale, which can land on a different double than the shortest
round-trip literal, so ``str(float(price))`` renders the exact ``88187.9`` as
``88187.90000000001``. Validating that string against a ``0.1`` tick size marks a
perfectly valid price as off-tick. These tests pin the exact-decimal path.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Price, Quantity

from kronos_mt5.baseline.config import BaselineConfig
from kronos_mt5.baseline.engine import (
    DAY_NS,
    DECISION_TAG_PREFIX,
    ReplayStrategy,
    decision_ts_from_tags,
    run_engine,
)
from kronos_mt5.baseline.instruments import (
    exact,
    exact_price,
    exact_quantity,
    instrument,
    reject_reason,
    rejection_detail,
    rounded,
    rounded_decimal,
    step_aligned_quantity,
    tick_aligned_price,
)
from kronos_mt5.baseline.metrics import rejection_counts, summarize
from kronos_mt5.baseline.synthetic import fixture

# The pinned public perpetual filter snapshot used by the full baseline.
PRODUCTION_FILTERS = {
    "BTCUSDT": {
        "tick_size": "0.10",
        "step_size": "0.001",
        "min_qty": "0.001",
        "max_qty": "120",
        "min_notional": "50",
    },
    "ETHUSDT": {
        "tick_size": "0.01",
        "step_size": "0.001",
        "min_qty": "0.001",
        "max_qty": "2000",
        "min_notional": "20",
    },
    "BNBUSDT": {
        "tick_size": "0.010",
        "step_size": "0.01",
        "min_qty": "0.01",
        "max_qty": "2000",
        "min_notional": "5",
    },
    "XRPUSDT": {
        "tick_size": "0.0001",
        "step_size": "0.1",
        "min_qty": "0.1",
        "max_qty": "2000000",
        "min_notional": "5",
    },
    "ADAUSDT": {
        "tick_size": "0.00010",
        "step_size": "1",
        "min_qty": "1",
        "max_qty": "3000000",
        "min_notional": "5",
    },
    "SOLUSDT": {
        "tick_size": "0.0100",
        "step_size": "0.01",
        "min_qty": "0.01",
        "max_qty": "80000",
        "min_notional": "5",
    },
    "LTCUSDT": {
        "tick_size": "0.01",
        "step_size": "0.001",
        "min_qty": "0.001",
        "max_qty": "100000",
        "min_notional": "20",
    },
    "LINKUSDT": {
        "tick_size": "0.001",
        "step_size": "0.01",
        "min_qty": "0.01",
        "max_qty": "50000",
        "min_notional": "20",
    },
}

# Representative reference prices per symbol, all exactly on their tick.
REFERENCE_PRICES = {
    "BTCUSDT": ("88187.9", "0.113"),
    "ETHUSDT": ("3421.37", "1.234"),
    "BNBUSDT": ("612.34", "1.23"),
    "XRPUSDT": ("2.3457", "1234.5"),
    "ADAUSDT": ("0.8231", "12345"),
    "SOLUSDT": ("187.43", "12.34"),
    "LTCUSDT": ("113.27", "1.234"),
    "LINKUSDT": ("23.457", "12.34"),
}


def _instruments():
    config = BaselineConfig()
    return {
        symbol: instrument(symbol, filters, config.commission_bps, config.leverage)
        for symbol, filters in PRODUCTION_FILTERS.items()
    }


def test_exact_btc_reproduction_of_false_price_precision_rejection():
    """The confirmed defect: exact 88187.9 on a 0.1 tick must not be rejected."""

    filters = PRODUCTION_FILTERS["BTCUSDT"]
    price = _instruments()["BTCUSDT"].make_price(88187.9)

    # The exact exchange value is on-tick, but the float round-trip is not.
    assert price.as_decimal() == Decimal("88187.9")
    assert str(float(price)) == "88187.90000000001"
    assert Decimal(str(float(price))) % Decimal("0.1") != 0
    assert price.as_decimal() % Decimal("0.1") == 0

    # Exact validation must therefore return no rejection at all.
    qty = _instruments()["BTCUSDT"].make_qty(0.113)
    assert reject_reason(qty, price, filters) is None
    assert reject_reason(qty, price, filters) != "price_precision"


def test_exact_reference_prices_for_every_configured_symbol():
    instruments = _instruments()
    for symbol, (price_str, qty_str) in REFERENCE_PRICES.items():
        filters = PRODUCTION_FILTERS[symbol]
        inst = instruments[symbol]
        price = inst.make_price(float(Decimal(price_str)))
        qty = inst.make_qty(float(Decimal(qty_str)))
        assert price.as_decimal() == Decimal(price_str), symbol
        assert qty.as_decimal() == Decimal(qty_str), symbol
        assert reject_reason(qty, price, filters) is None, symbol


def test_valid_quantities_across_step_sizes():
    instruments = _instruments()
    cases = {
        "BTCUSDT": ("0.001", "0.113", "1.234"),  # step 0.001
        "BNBUSDT": ("0.01", "1.23", "199.99"),  # step 0.01
        "XRPUSDT": ("0.1", "1234.5", "99999.9"),  # step 0.1
        "ADAUSDT": ("1", "12345", "999999"),  # step 1
    }
    for symbol, quantities in cases.items():
        filters = PRODUCTION_FILTERS[symbol]
        inst = instruments[symbol]
        price = inst.make_price(float(Decimal(REFERENCE_PRICES[symbol][0])))
        for qty_str in quantities:
            qty = inst.make_qty(float(Decimal(qty_str)))
            assert qty.as_decimal() == Decimal(qty_str), (symbol, qty_str)
            reason = reject_reason(qty, price, filters)
            assert reason != "quantity_precision", (symbol, qty_str)


def test_genuine_off_tick_prices_are_still_rejected():
    filters = PRODUCTION_FILTERS["BTCUSDT"]
    for off_tick in ("88187.95", "88187.99", "0.05"):
        assert reject_reason(Decimal("0.113"), Decimal(off_tick), filters) == "price_precision"
    # A genuinely off-tick ETH price on a 0.01 tick.
    assert (
        reject_reason(Decimal("1.234"), Decimal("3421.375"), PRODUCTION_FILTERS["ETHUSDT"])
        == "price_precision"
    )


def test_genuine_off_step_quantities_are_still_rejected():
    filters = PRODUCTION_FILTERS["BTCUSDT"]
    price = Decimal("88187.9")
    for off_step in ("0.0001", "0.1135", "0.00099"):
        assert reject_reason(Decimal(off_step), price, filters) == "quantity_precision"
    # ADA trades in whole units.
    assert (
        reject_reason(Decimal("12345.5"), Decimal("0.8231"), PRODUCTION_FILTERS["ADAUSDT"])
        == "quantity_precision"
    )


def test_minimum_notional_failures_remain_rejected():
    filters = PRODUCTION_FILTERS["BTCUSDT"]  # min_notional 50
    assert reject_reason(Decimal("0.001"), Decimal("100.0"), filters) == "minimum_notional"
    assert reject_reason(Decimal("0.001"), Decimal("100.0"), filters, reduce_only=True) is None
    assert reject_reason(Decimal("0.001"), Decimal("88187.9"), filters) is None


def test_maximum_quantity_failures_remain_rejected():
    filters = PRODUCTION_FILTERS["BTCUSDT"]  # max_qty 120
    assert reject_reason(Decimal(121), Decimal("88187.9"), filters) == "quantity_limits"
    assert reject_reason(Decimal("0.0001"), Decimal("88187.9"), filters) == "quantity_precision"
    # Below min_qty but on-step.
    assert (
        reject_reason(Decimal(1), Decimal("0.8231"), PRODUCTION_FILTERS["ADAUSDT"])
        != "quantity_limits"
    )
    assert (
        reject_reason(Decimal(3000001), Decimal("0.8231"), PRODUCTION_FILTERS["ADAUSDT"])
        == "quantity_limits"
    )


def test_nonpositive_and_nonfinite_are_rejected():
    filters = PRODUCTION_FILTERS["BTCUSDT"]
    assert reject_reason(Decimal(0), Decimal("88187.9"), filters) == "nonpositive_or_nonfinite"
    assert reject_reason(Decimal(-1), Decimal("88187.9"), filters) == "nonpositive_or_nonfinite"
    assert reject_reason(Decimal("0.113"), Decimal(0), filters) == "nonpositive_or_nonfinite"
    assert reject_reason(Decimal("NaN"), Decimal(1), filters) == "nonpositive_or_nonfinite"
    assert reject_reason(None, Decimal(1), filters) == "nonpositive_or_nonfinite"


def test_exact_accepts_supported_types_and_rejects_others():
    assert exact(Decimal("1.5")) == Decimal("1.5")
    assert exact(Price(88187.9, 1)) == Decimal("88187.9")
    assert exact(Quantity(0.113, 3)) == Decimal("0.113")
    assert exact("0.10") == Decimal("0.10")
    assert exact(7) == Decimal(7)
    # Floats are supported for external callers via their shortest repr.
    assert exact(0.1) == Decimal("0.1")
    with pytest.raises(TypeError):
        exact(True)
    with pytest.raises(TypeError):
        exact(object())


@pytest.mark.parametrize("symbol", sorted(PRODUCTION_FILTERS))
def test_property_every_tick_aligned_price_and_step_aligned_quantity_passes(symbol):
    """Every aligned Price/Quantity for every configured symbol must validate."""

    filters = PRODUCTION_FILTERS[symbol]
    inst = _instruments()[symbol]
    tick, step = Decimal(filters["tick_size"]), Decimal(filters["step_size"])
    base_price = Decimal(REFERENCE_PRICES[symbol][0])
    base_qty = Decimal(REFERENCE_PRICES[symbol][1])
    price_origin = (base_price / tick).to_integral_value()
    qty_origin = (base_qty / step).to_integral_value()

    # Sweep both directions while keeping every generated value strictly positive.
    span = 400
    low = -min(span, int(price_origin) - 1, int(qty_origin) - 1)
    checked = 0
    for i in range(low, span):
        price_decimal = (price_origin + i) * tick
        qty_decimal = (qty_origin + i) * step
        if price_decimal <= 0 or qty_decimal <= 0:
            continue
        price = inst.make_price(float(price_decimal))
        qty = inst.make_qty(float(qty_decimal))
        # Construction must be lossless for aligned values.
        assert price.as_decimal() == price_decimal.normalize() or price.as_decimal() == (
            price_decimal
        ), (symbol, price_decimal)
        reason = reject_reason(qty, price, filters)
        assert reason not in {"price_precision", "quantity_precision"}, (
            symbol,
            price_decimal,
            qty_decimal,
            reason,
        )
        checked += 1
    assert checked >= 500 and low < 0


@pytest.mark.parametrize("symbol", sorted(PRODUCTION_FILTERS))
def test_tick_aligned_price_helper_snaps_and_verifies(symbol):
    filters = PRODUCTION_FILTERS[symbol]
    inst = _instruments()[symbol]
    tick = filters["tick_size"]
    base = Decimal(REFERENCE_PRICES[symbol][0])
    for factor in ("0.9993", "1.0", "1.00071", "0.87"):
        raw = base * Decimal(factor)
        down = tick_aligned_price(inst, raw, tick)
        up = tick_aligned_price(inst, raw, tick, up=True)
        assert down.as_decimal() % Decimal(tick) == 0
        assert up.as_decimal() % Decimal(tick) == 0
        assert down.as_decimal() <= raw <= up.as_decimal()
        assert (
            reject_reason(inst.make_qty(float(Decimal(REFERENCE_PRICES[symbol][1]))), down, filters)
            != "price_precision"
        )


def test_rounded_helpers_are_exact_and_consistent():
    assert rounded(1.2349, "0.001") == 1.234
    assert rounded_decimal(Decimal("1.2349"), "0.001") == Decimal("1.234")
    assert rounded_decimal(Decimal("1.2341"), "0.001", up=True) == Decimal("1.235")
    # A Nautilus Price must round through its exact decimal, not through float.
    assert rounded_decimal(Price(88187.9, 1), "0.10") == Decimal("88187.90")
    assert rounded_decimal(Quantity(0.113, 3), "0.001") == Decimal("0.113")


def test_rejection_detail_carries_symbol_and_exact_attempted_values():
    filters = PRODUCTION_FILTERS["BTCUSDT"]
    inst = _instruments()["BTCUSDT"]
    detail = rejection_detail(
        "BTCUSDT",
        1234,
        "price_precision",
        inst.make_qty(0.113),
        inst.make_price(88187.9),
        filters,
        decision_ts_ns=1000,
    )
    assert detail["symbol"] == "BTCUSDT"
    assert detail["ts_ns"] == 1234
    assert detail["reason"] == "price_precision"
    assert detail["attempted_price"] == "88187.9"  # exact, never 88187.90000000001
    assert detail["attempted_qty"] == "0.113"
    assert detail["tick_size"] == "0.10"
    assert detail["step_size"] == "0.001"
    assert detail["min_notional"] == "50"
    assert detail["decision_ts_ns"] == 1000
    # An unrenderable value must still keep its symbol and reason.
    broken = rejection_detail("ETHUSDT", 1, "production_filters", None, None, filters)
    assert broken["symbol"] == "ETHUSDT" and broken["attempted_price"] is None


def test_rejection_counts_group_by_reason_and_symbol():
    rejections = [
        {"symbol": "BTCUSDT", "reason": "price_precision"},
        {"symbol": "BTCUSDT", "reason": "price_precision"},
        {"symbol": "ETHUSDT", "reason": "leverage_limit"},
        {"symbol": None, "reason": "leverage_limit"},
    ]
    assert rejection_counts(rejections, "reason") == {"leverage_limit": 2, "price_precision": 2}
    assert rejection_counts(rejections, "symbol") == {
        "BTCUSDT": 2,
        "ETHUSDT": 1,
        "UNATTRIBUTED": 1,
    }
    assert rejection_counts(rejections, "symbol", "reason") == {
        "BTCUSDT/price_precision": 2,
        "ETHUSDT/leverage_limit": 1,
        "UNATTRIBUTED/leverage_limit": 1,
    }


@pytest.fixture(scope="module")
def precision_replay():
    """Replay the synthetic fixture on a BTC-like 0.1 tick that triggers the bug."""

    frames, funding, filters, config = fixture()
    filters = {
        "BTCUSDT": {
            **filters["BTCUSDT"],
            "tick_size": "0.10",
            "step_size": "0.001",
            "min_notional": "5",
        }
    }
    origin = int(frames["BTCUSDT"].iloc[0].open_time) * 1_000_000
    result = run_engine(
        frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    return frames, funding, filters, config, origin, result


def test_next_open_valid_quotes_produce_no_price_precision_rejections(precision_replay):
    *_, result = precision_replay
    reasons = [r["reason"] for r in result["rejections"]]
    assert "price_precision" not in reasons
    assert "quantity_precision" not in reasons
    assert result["fills"], "the replay must still execute orders"


def test_every_replay_rejection_is_attributed(precision_replay):
    *_, result = precision_replay
    for rejection in result["rejections"]:
        assert rejection["symbol"], rejection
        assert rejection["reason"], rejection
        assert "tick_size" in rejection and "step_size" in rejection
        assert "min_notional" in rejection and "decision_ts_ns" in rejection


def test_replay_stays_deterministic_and_reconciles(precision_replay):
    frames, funding, filters, config, origin, result = precision_replay
    repeat = run_engine(
        frames, funding, filters, config, origin + 253 * DAY_NS, origin + 300 * DAY_NS
    )
    assert result == repeat
    metrics, repeated_metrics = summarize(result), summarize(repeat)
    assert metrics == repeated_metrics
    assert abs(metrics["accounting_residual"]) < 1e-5
    assert metrics["net_pnl"] == pytest.approx(
        metrics["gross_pnl"] - sum(metrics["costs"].values())
    )
    assert metrics["rejected_orders"] == sum(metrics["rejections_by_reason"].values())
    assert metrics["rejected_orders"] == sum(metrics["rejections_by_symbol"].values())
    assert "UNATTRIBUTED" not in metrics["rejections_by_symbol"]


def test_synthetic_quotes_are_tick_aligned(precision_replay):
    """Every simulated quote reaching execution must already be on the tick."""

    from kronos_mt5.baseline.engine import events_for_frame

    frames, _, filters, config, *_ = precision_replay
    inst = instrument("BTCUSDT", filters["BTCUSDT"], config.commission_bps, config.leverage)
    tick = Decimal(filters["BTCUSDT"]["tick_size"])
    events = events_for_frame(frames["BTCUSDT"], inst, config, [])
    quotes = [e for e in events if hasattr(e, "bid_price")]
    assert quotes
    for quote in quotes:
        assert quote.bid_price.as_decimal() % tick == 0
        assert quote.ask_price.as_decimal() % tick == 0


# --- Exact quantity survival through pending execution -----------------------


def _pending_strategy(inst, filters, ts_ns=1234):
    """A minimal stand-in for the parts of ReplayStrategy under test."""

    return SimpleNamespace(
        symbol=inst.id.symbol.value,
        instrument=inst,
        instrument_id=inst.id,
        filters=filters,
        decisions=[],
        rejections=[],
        pending=None,
        clock=SimpleNamespace(timestamp_ns=lambda: ts_ns),
    )


@pytest.mark.parametrize(
    ("symbol", "quantities"),
    [
        ("BTCUSDT", ("0.001", "0.113", "1.234", "119.999")),  # step 0.001
        ("BNBUSDT", ("0.01", "1.23", "199.99")),  # step 0.01
        ("XRPUSDT", ("0.1", "1234.5", "99999.9")),  # step 0.1
        ("ADAUSDT", ("1", "12345", "999999")),  # step 1
    ],
)
def test_exact_quantity_survives_pending_execution(symbol, quantities):
    """Quantity -> pending decision -> next-open execution must not drift a step."""

    filters = PRODUCTION_FILTERS[symbol]
    inst = _instruments()[symbol]
    step = Decimal(filters["step_size"])
    price = inst.make_price(float(Decimal(REFERENCE_PRICES[symbol][0])))

    for qty_str in quantities:
        original = inst.make_qty(float(Decimal(qty_str)))
        assert original.as_decimal() == Decimal(qty_str), (symbol, qty_str)

        strategy = _pending_strategy(inst, filters)
        ReplayStrategy._submit_entry_order(strategy, OrderSide.BUY, original, price)

        # Private pending state carries the Quantity itself, never a float.
        pending_qty = strategy.pending["qty"]
        assert isinstance(pending_qty, Quantity), (symbol, qty_str)
        assert pending_qty is original

        # The executable quantity is rebuilt from the exact decimal only.
        executable = step_aligned_quantity(inst, exact(pending_qty), filters["step_size"])
        assert executable.as_decimal() == original.as_decimal(), (symbol, qty_str)
        assert executable.as_decimal() == Decimal(qty_str), (symbol, qty_str)
        assert abs(executable.as_decimal() - original.as_decimal()) < step
        assert reject_reason(executable, price, filters) != "quantity_precision"


def test_submit_entry_order_keeps_exact_quantity_and_serializable_record():
    inst = _instruments()["BTCUSDT"]
    strategy = _pending_strategy(inst, PRODUCTION_FILTERS["BTCUSDT"])
    quantity = inst.make_qty(0.113)

    ReplayStrategy._submit_entry_order(strategy, OrderSide.BUY, quantity, 88187.9)

    decision = strategy.decisions[-1]
    # The exported decision record stays JSON-serializable...
    assert json.loads(json.dumps(decision)) == decision
    assert decision["exact_qty"] == "0.113"
    assert decision["qty"] == pytest.approx(0.113)
    assert decision["ts_ns"] == 1234
    # ...while private pending state holds the exact Quantity object itself.
    assert strategy.pending["decision"] is decision
    assert isinstance(strategy.pending["qty"], Quantity)
    assert exact(strategy.pending["qty"]) == Decimal("0.113")


def test_replay_decisions_record_exact_quantities(precision_replay):
    *_, result = precision_replay
    assert result["decisions"]
    for decision in result["decisions"]:
        assert Decimal(decision["exact_qty"]) % Decimal("0.001") == 0, decision
        assert json.loads(json.dumps(decision)) == decision


# --- Engine rejection telemetry ----------------------------------------------


def test_decision_ts_from_tags_is_defensive():
    assert decision_ts_from_tags([f"{DECISION_TAG_PREFIX}42"]) == 42
    assert decision_ts_from_tags(f"{DECISION_TAG_PREFIX}7") == 7
    assert decision_ts_from_tags(["other", f"{DECISION_TAG_PREFIX}9"]) == 9
    # Protective and unrelated orders carry no baseline tag.
    assert decision_ts_from_tags(None) is None
    assert decision_ts_from_tags([]) is None
    assert decision_ts_from_tags(["protective_stop"]) is None
    assert decision_ts_from_tags([f"{DECISION_TAG_PREFIX}not-an-int"]) is None


def _rejection_strategy(inst, filters, order):
    quote = SimpleNamespace(bid_price=inst.make_price(88187.9), ask_price=inst.make_price(88188.0))
    return SimpleNamespace(
        symbol=inst.id.symbol.value,
        filters=filters,
        instrument_id=inst.id,
        rejections=[],
        shared={"quotes": {inst.id: quote}},
        cache=SimpleNamespace(order=lambda _cid: order),
    )


@pytest.mark.parametrize(
    ("side", "expected_price"),
    [(OrderSide.BUY, "88188.0"), (OrderSide.SELL, "88187.9")],
)
def test_engine_rejection_records_executable_side_and_decision_ts(side, expected_price):
    """BUY must record the ask and SELL the bid, with the decision timestamp."""

    inst = _instruments()["BTCUSDT"]
    order = SimpleNamespace(
        side=side, quantity=inst.make_qty(0.113), tags=[f"{DECISION_TAG_PREFIX}555"]
    )
    strategy = _rejection_strategy(inst, PRODUCTION_FILTERS["BTCUSDT"], order)

    record = ReplayStrategy._record_engine_rejection(
        strategy, SimpleNamespace(client_order_id="O-1", ts_event=999, reason="INVALID_ORDER")
    )

    assert strategy.rejections[-1] is record
    assert record["symbol"] == "BTCUSDT"
    # Exact value, never the 88187.90000000001 float artifact.
    assert Decimal(record["attempted_price"]) == Decimal(expected_price)
    assert Decimal(record["attempted_qty"]) == Decimal("0.113")
    assert record["decision_ts_ns"] == 555
    assert record["reason"] == "INVALID_ORDER"
    assert record["tick_size"] == "0.10" and record["step_size"] == "0.001"


def test_engine_rejection_without_baseline_tag_keeps_symbol():
    inst = _instruments()["BTCUSDT"]
    order = SimpleNamespace(side=OrderSide.SELL, quantity=inst.make_qty(0.113), tags=None)
    strategy = _rejection_strategy(inst, PRODUCTION_FILTERS["BTCUSDT"], order)

    record = ReplayStrategy._record_engine_rejection(
        strategy, SimpleNamespace(client_order_id="O-2", ts_event=1, reason="REJECTED")
    )
    assert record["symbol"] == "BTCUSDT"
    assert record["decision_ts_ns"] is None
    assert Decimal(record["attempted_price"]) == Decimal("88187.9")


def test_engine_rejection_with_unknown_order_still_keeps_symbol():
    inst = _instruments()["BTCUSDT"]
    strategy = _rejection_strategy(inst, PRODUCTION_FILTERS["BTCUSDT"], None)
    record = ReplayStrategy._record_engine_rejection(
        strategy, SimpleNamespace(client_order_id="O-3", ts_event=2, reason="UNKNOWN")
    )
    assert record["symbol"] == "BTCUSDT"
    assert record["attempted_price"] is None
    assert record["attempted_qty"] is None
    assert record["decision_ts_ns"] is None


def test_every_baseline_entry_order_carries_a_decision_tag():
    """Baseline entry orders must be traceable back to their decision."""

    tag = f"{DECISION_TAG_PREFIX}{1234}"
    assert decision_ts_from_tags([tag]) == 1234
    inst = _instruments()["BTCUSDT"]
    order = SimpleNamespace(side=OrderSide.BUY, quantity=inst.make_qty(0.113), tags=[tag])
    strategy = _rejection_strategy(inst, PRODUCTION_FILTERS["BTCUSDT"], order)
    record = ReplayStrategy._record_engine_rejection(
        strategy, SimpleNamespace(client_order_id="O-4", ts_event=3, reason="X")
    )
    assert record["decision_ts_ns"] == 1234


def test_exact_price_and_quantity_raise_on_precision_loss():
    inst = _instruments()["BTCUSDT"]  # price precision 1, size precision 3
    assert exact_price(inst, Decimal("88187.9")).as_decimal() == Decimal("88187.9")
    assert exact_quantity(inst, Decimal("0.113")).as_decimal() == Decimal("0.113")
    # A value needing more digits than the instrument supports must not be
    # silently rounded: that can move a price by a whole tick.
    with pytest.raises(ValueError, match="not representable"):
        exact_price(inst, Decimal("88187.95"))
    with pytest.raises(ValueError, match="not representable"):
        exact_quantity(inst, Decimal("0.1135"))


def test_step_aligned_quantity_snaps_and_verifies():
    inst = _instruments()["BTCUSDT"]
    assert step_aligned_quantity(inst, Decimal("0.1139"), "0.001").as_decimal() == Decimal("0.113")
    assert step_aligned_quantity(inst, Decimal("0.1131"), "0.001", up=True).as_decimal() == Decimal(
        "0.114"
    )
    assert step_aligned_quantity(inst, Quantity(0.113, 3), "0.001").as_decimal() == Decimal("0.113")
