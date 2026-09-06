"""Patient-limit execution: lifecycle, fresh-price/drift guards and telemetry.

Everything here runs on deterministic doubles — no Binance, no engine, no network.
The strategy is built the same way `test_strategy_extras.py` does it: bypass the
Cython `Strategy.__init__` with `__new__` and shadow the read-only engine
properties with plain properties on the Python subclass.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict, deque
from types import SimpleNamespace

import pytest

from kronos_mt5.companion import store
from kronos_mt5.execution import (
    ACTION_SKIP,
    ROLE_PATIENT_FALLBACK,
    ROLE_PATIENT_LIMIT,
    SKIP_ADVERSE_DRIFT,
    SKIP_ALREADY_RESOLVED,
    SKIP_BELOW_THRESHOLD,
    SKIP_NO_FRESH_PRICE,
    SKIP_TARGET_REACHED,
    SKIP_UNOWNED_RESTING_ORDER,
    STATE_CANCEL_PENDING,
    STATE_CANCEL_RETRY,
    STATE_SETTLING,
    STATE_WORKING,
    TAG_DECISION_TS,
    PatientOrder,
    adverse_drift_bps,
    classify_exec_role,
    evaluate_fallback,
    implementation_shortfall_bps,
    implementation_shortfall_quote,
    liquidity_label,
    tag_float,
    tag_int,
)
from kronos_mt5.strategies.trend_strategy import TrendStrategy

# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


STRATEGY_ID = "TrendStrategy-TEST"  # what the engine reports for our own orders
OWNER_TAG = "OWNER_STRATEGY=TrendStrategy:BTCUSDT-PERP.BINANCE"  # stable identity


class _Num(float):
    """Stands in for Quantity/Price (float that survives str()/comparisons)."""


class _FakeInstrument:
    min_quantity = 0.001
    max_quantity = None
    min_notional = 5.0

    def make_qty(self, x):
        return _Num(round(float(x), 6))

    def make_price(self, x):
        return _Num(round(float(x), 2))


class _FakeInstrumentId:
    def __init__(self, text: str = "BTCUSDT-PERP.BINANCE") -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text

    def __eq__(self, other) -> bool:
        return str(other) == self._text

    def __hash__(self) -> int:
        return hash(self._text)

    @property
    def symbol(self):
        return SimpleNamespace(value=self._text.split(".")[0])


class _FakeOrder:
    def __init__(self, kind: str, kwargs: dict, coid: str) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.client_order_id = coid
        self.is_closed = False
        self.is_reduce_only = bool(kwargs.get("reduce_only", False))
        self.side = kwargs.get("order_side")
        self.quantity = kwargs.get("quantity")
        self.tags = kwargs.get("tags")
        self.price = kwargs.get("price")
        self.order_type = SimpleNamespace(
            name={"limit": "LIMIT", "market": "MARKET"}.get(kind, "OTHER")
        )
        self.strategy_id = STRATEGY_ID


class _FakeFactory:
    def __init__(self) -> None:
        self.orders: list[_FakeOrder] = []

    def _mk(self, kind: str, kwargs: dict) -> _FakeOrder:
        order = _FakeOrder(kind, kwargs, f"{kind}-{len(self.orders) + 1}")
        self.orders.append(order)
        return order

    def limit(self, **kwargs):
        return self._mk("limit", kwargs)

    def market(self, **kwargs):
        return self._mk("market", kwargs)

    def stop_market(self, **kwargs):
        return self._mk("stop", kwargs)

    def trailing_stop_market(self, **kwargs):
        return self._mk("trail", kwargs)


class _FakeClock:
    def __init__(self, now_ns: int = 1_000_000_000_000) -> None:
        self.now_ns = now_ns
        self.timers: dict[str, object] = {}
        self.fired: list[str] = []

    def timestamp_ns(self) -> int:
        return self.now_ns

    def set_timer(self, name, interval, callback):
        self.timers[name] = (callback, interval)

    def cancel_timer(self, name):
        self.timers.pop(name, None)

    @property
    def timer_names(self):
        return list(self.timers)

    def fire(self, prefix: str) -> bool:
        """Fire the single timer whose name starts with `prefix` (like the engine
        would when the interval elapses). Returns False when none is armed."""
        for name in list(self.timers):
            if name.startswith(prefix):
                callback, interval = self.timers.pop(name)
                self.now_ns += int(interval.total_seconds() * 1e9)
                self.fired.append(name)
                callback(None)
                return True
        return False

    def armed(self, prefix: str) -> bool:
        return any(name.startswith(prefix) for name in self.timers)


class _FakeLog:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def _add(self, level):
        def _log(message):
            self.messages.append((level, str(message)))

        return _log

    def __getattr__(self, level):
        return self._add(level)

    def text(self) -> str:
        return "\n".join(m for _, m in self.messages)


def _patient_strat(
    monkeypatch,
    *,
    net_position: float = 0.0,
    mark: float | None = 100.0,
    mark_age_secs: float = 1.0,
    post_only: bool = False,
    max_adverse_bps: float = 15.0,
    fallback: bool = True,
    timeout_secs: int = 300,
    mode: str = "run",
    market_data_stale: bool = False,
    flatten_seq: int = 0,
) -> TrendStrategy:
    s = TrendStrategy.__new__(TrendStrategy)
    cfg = SimpleNamespace(
        use_patient_limit=True,
        patient_limit_offset_bps=2.0,
        patient_limit_timeout_secs=timeout_secs,
        patient_limit_market_fallback=fallback,
        patient_limit_post_only=post_only,
        patient_limit_max_adverse_bps=max_adverse_bps,
    )
    s.cfg = lambda: cfg
    s.instrument = _FakeInstrument()
    s._fake_factory = _FakeFactory()
    s._fake_clock = _FakeClock()
    s._fake_log = _FakeLog()
    s._fake_cache = SimpleNamespace(
        quote_tick=lambda iid: None,
        trade_tick=lambda iid: None,
        orders_open=lambda instrument_id=None: [],
    )
    s._net_position = net_position
    s._fake_portfolio = SimpleNamespace(net_position=lambda iid: s._net_position)

    monkeypatch.setattr(TrendStrategy, "order_factory", property(lambda self: self._fake_factory))
    monkeypatch.setattr(TrendStrategy, "cache", property(lambda self: self._fake_cache))
    monkeypatch.setattr(TrendStrategy, "clock", property(lambda self: self._fake_clock))
    monkeypatch.setattr(TrendStrategy, "portfolio", property(lambda self: self._fake_portfolio))
    monkeypatch.setattr(TrendStrategy, "log", property(lambda self: self._fake_log))
    monkeypatch.setattr(
        TrendStrategy, "instrument_id", property(lambda self: _FakeInstrumentId()), raising=False
    )
    monkeypatch.setattr(TrendStrategy, "id", property(lambda self: STRATEGY_ID), raising=False)

    clock_now = s._fake_clock.now_ns
    s.risk_state = SimpleNamespace(
        mode=mode,
        # the live runner always attaches a MarkPriceMonitor, so the watchdog flag
        # is the discriminator between "live" and "bar-only simulation"
        mark_watchdog_enabled=True,
        market_data_stale=market_data_stale,
        flatten_seq=flatten_seq,
        flatten_target="all",
        mark_prices=({str(_FakeInstrumentId()): mark} if mark is not None else {}),
        mark_received_ns=(
            {str(_FakeInstrumentId()): clock_now - int(mark_age_secs * 1e9)}
            if mark is not None
            else {}
        ),
    )
    s._patient = None
    s._patient_order = None
    s._retiring = {}
    s._tombstones = OrderedDict()
    s._orders_by_coid = {}
    s._deferred = None
    s._owned_coids = set()
    s._closes = deque([100.0], maxlen=8)
    s._protective_orders = []
    s._protection_signature = None
    s._protection_position_key = None
    s._dynamic_stop_pct = 0.20
    s._last_flatten_seq = 0
    s.submitted: list[_FakeOrder] = []
    s.canceled: list[_FakeOrder] = []
    s.cancel_raises = False
    s.submit_order = s.submitted.append

    def _cancel(order):
        if s.cancel_raises:
            raise RuntimeError("venue rejected the cancel request")
        s.canceled.append(order)

    s.cancel_order = _cancel
    s._refresh_protection = lambda *a, **k: None
    return s


def _side(name: str):
    from nautilus_trader.model.enums import OrderSide

    return OrderSide.BUY if name == "BUY" else OrderSide.SELL


def _submit_patient(s: TrendStrategy, *, side: str = "BUY", units: float = 10.0, price=100.0):
    s._submit_entry_order(_side(side), s.instrument.make_qty(abs(units)), price, target_units=units)
    return s._patient_order


def _markets(s: TrendStrategy) -> list[_FakeOrder]:
    return [o for o in s.submitted if o.kind == "market"]


def _fill(order, qty, *, closed: bool, iid=None):
    order.is_closed = closed
    return SimpleNamespace(
        instrument_id=iid or _FakeInstrumentId(),
        client_order_id=order.client_order_id,
        last_qty=qty,
    )


# --------------------------------------------------------------------------
# helpers for the cancel-confirmed / settling lifecycle
# --------------------------------------------------------------------------


def _settle(s: TrendStrategy) -> bool:
    """Fire the settling timer, as the engine would when the window elapses."""
    return s._fake_clock.fire("patient_settle_")


def _confirm_cancel(s: TrendStrategy, order) -> None:
    order.is_closed = True
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))


def _timeout_then_cancel(s: TrendStrategy, order) -> None:
    """Full happy path: timeout -> cancel requested -> confirmed -> settled."""
    s._on_patient_timeout()
    _confirm_cancel(s, order)
    _settle(s)


def _external_limit(coid: str = "manual-1") -> _FakeOrder:
    """A resting entry limit this strategy cannot prove it owns."""
    order = _FakeOrder("limit", {}, coid)
    order.tags = None
    order.strategy_id = "SomeoneElse-001"
    return order


def _reconciled_own_limit(coid: str = "orphan-1", qty: float = 3.0) -> _FakeOrder:
    """Our own limit as it comes back after a restart: the engine no longer
    reports a strategy id, so only the stable OWNER_STRATEGY tag identifies it."""
    order = _FakeOrder("limit", {"quantity": qty}, coid)
    order.tags = [f"EXEC_ROLE={ROLE_PATIENT_LIMIT}", OWNER_TAG]
    order.strategy_id = None
    return order


# --------------------------------------------------------------------------
# 1-4: cancellation is terminal before anything is submitted
# --------------------------------------------------------------------------


def test_timeout_requests_cancel_without_submitting_a_market_order(monkeypatch):
    """1. The timeout must never place the fallback itself."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s)

    s._on_patient_timeout()

    assert s.canceled == [order]
    assert _markets(s) == []  # nothing taken before the venue confirms
    assert s._patient is None  # no longer the active order...
    (retiring,) = s._retiring.values()  # ...but still tracked by client order id
    assert retiring.state == STATE_CANCEL_PENDING
    assert retiring.fallback_pending is True


def test_no_fallback_until_the_settling_window_has_elapsed(monkeypatch):
    """The cancel callback alone is not enough: resolution waits for settling."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()

    _confirm_cancel(s, order)
    assert _markets(s) == []  # still nothing — a late fill could still arrive
    assert s._retiring  # ownership retained across the settling window

    assert _settle(s) is True
    assert len(_markets(s)) == 1
    assert not s._retiring


def test_partial_fill_falls_back_only_for_the_remaining_delta(monkeypatch):
    """2. A 4-unit partial of a 10-unit target must fall back for 6, not 10."""
    s = _patient_strat(monkeypatch, net_position=0.0)
    order = _submit_patient(s, units=10.0)

    s._net_position = 4.0
    s.on_order_filled(_fill(order, 4.0, closed=False))
    assert s._patient is not None  # a partial must NOT clear the state
    assert s._patient.state == STATE_WORKING

    _timeout_then_cancel(s, order)

    (fallback,) = _markets(s)
    assert float(fallback.kwargs["quantity"]) == pytest.approx(6.0)
    assert fallback.kwargs["order_side"].name == "BUY"
    assert f"EXEC_ROLE={ROLE_PATIENT_FALLBACK}" in fallback.kwargs["tags"]
    assert s._patient is None


def test_full_fill_while_cancel_pending_submits_no_fallback(monkeypatch):
    """3. The late fill completed the target — there is nothing left to take."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()

    s._net_position = 10.0
    s.on_order_filled(_fill(order, 10.0, closed=True))
    _settle(s)

    assert _markets(s) == []
    assert s._patient is None
    assert f"reason={SKIP_TARGET_REACHED}" in s._fake_log.text()


def test_duplicate_terminal_callbacks_create_exactly_one_fallback(monkeypatch):
    """4. Repeated and conflicting terminal callbacks are idempotent."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()
    order.is_closed = True

    event = SimpleNamespace(client_order_id=order.client_order_id)
    s.on_order_canceled(event)
    s.on_order_canceled(event)  # duplicate from the venue
    s.on_order_expired(event)  # and a conflicting terminal
    _settle(s)
    s.on_order_canceled(event)  # and one more, after resolution
    _settle(s)

    assert len(_markets(s)) == 1
    assert s._patient is None


def test_repeated_timeouts_do_not_re_request_cancellation(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s)

    s._on_patient_timeout()
    s._on_patient_timeout()
    s._on_patient_timeout()

    assert s.canceled == [order]
    assert _markets(s) == []


# --------------------------------------------------------------------------
# late fills arriving after the cancellation notification
# --------------------------------------------------------------------------


def test_late_full_fill_after_cancel_notification_causes_no_fallback(monkeypatch):
    """11. The order was reported canceled, then a full fill landed. Resolution
    happens after settling, so the fallback quantity must be zero."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()
    _confirm_cancel(s, order)

    # ... the fill arrives AFTER the cancel notification but BEFORE settling
    s._net_position = 10.0
    s.on_order_filled(_fill(order, 10.0, closed=True))
    _settle(s)

    assert _markets(s) == []
    assert f"reason={SKIP_TARGET_REACHED}" in s._fake_log.text()


def test_late_partial_fill_after_cancel_notification_reduces_the_fallback(monkeypatch):
    """12. A late partial must shrink the fallback, not leave it at full size."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()
    _confirm_cancel(s, order)

    s._net_position = 7.5  # late partial lands during the settling window
    s.on_order_filled(_fill(order, 7.5, closed=True))
    _settle(s)

    (fallback,) = _markets(s)
    assert float(fallback.kwargs["quantity"]) == pytest.approx(2.5)


def test_fill_after_settlement_is_recorded_and_left_to_the_next_cycle(monkeypatch):
    """A bounded tombstone keeps a very late fill attributable without re-acting."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    _timeout_then_cancel(s, order)
    assert len(_markets(s)) == 1

    s.on_order_filled(_fill(order, 1.0, closed=True))

    assert len(_markets(s)) == 1  # no second reaction
    assert order.client_order_id in s._tombstones
    assert "late fill on retired patient order" in s._fake_log.text()


# --------------------------------------------------------------------------
# supersede: never replace before the predecessor is terminal
# --------------------------------------------------------------------------


def test_supersede_requests_cancel_and_submits_no_replacement(monkeypatch):
    """1. A newer target must not put a second entry order on the book."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)

    second = _submit_patient(s, units=12.0)

    assert second is None  # nothing new was placed; the field is not clear
    assert s.canceled == [first]
    assert len(s.submitted) == 1  # still only the original limit
    assert s._patient is None
    assert s._deferred is not None and s._deferred.target_units == 12.0
    (retiring,) = s._retiring.values()
    assert retiring.state == STATE_CANCEL_PENDING


def test_replacement_is_submitted_only_after_the_predecessor_settles(monkeypatch):
    """2. Cancel confirmation alone is not enough; settling must complete too."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    _submit_patient(s, units=12.0)

    _confirm_cancel(s, first)
    assert len(s.submitted) == 1  # still nothing new during settling

    _settle(s)

    assert len(s.submitted) == 2
    replacement = s.submitted[1]
    assert replacement.kind == "limit"
    assert float(replacement.kwargs["quantity"]) == pytest.approx(12.0)
    assert s._patient.target_units == 12.0
    assert s._deferred is None


def test_partial_predecessor_fill_reduces_the_replacement_quantity(monkeypatch):
    """3. Replacement is sized from target - current, not from the old order."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    _submit_patient(s, units=12.0)

    s._net_position = 4.0
    s.on_order_filled(_fill(first, 4.0, closed=True))
    _settle(s)

    replacement = s.submitted[1]
    assert float(replacement.kwargs["quantity"]) == pytest.approx(8.0)  # 12 - 4


def test_full_predecessor_fill_eliminates_the_replacement(monkeypatch):
    """4. If the retiring order already reached the new target, submit nothing."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    _submit_patient(s, units=10.0)

    s._net_position = 10.0
    s.on_order_filled(_fill(first, 10.0, closed=True))
    _settle(s)

    assert len(s.submitted) == 1
    assert f"reason={SKIP_TARGET_REACHED}" in s._fake_log.text()
    assert s._deferred is None


def test_multiple_supersedes_coalesce_onto_the_newest_target(monkeypatch):
    """5. A burst of rebalances must never queue a burst of orders."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    _submit_patient(s, units=12.0)
    _submit_patient(s, units=14.0)
    _submit_patient(s, units=9.0)

    assert s._deferred.target_units == 9.0
    assert s._deferred.requests == 3
    assert s.canceled == [first]  # cancelled once, not once per request

    _confirm_cancel(s, first)
    _settle(s)

    assert len(s.submitted) == 2
    assert float(s.submitted[1].kwargs["quantity"]) == pytest.approx(9.0)


def test_replacement_below_the_rebalance_threshold_is_not_submitted(monkeypatch):
    """A delta that no longer clears the threshold must trade nothing."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    s._submit_entry_order(
        _side("BUY"),
        s.instrument.make_qty(10.05),
        100.0,
        target_units=10.05,
        threshold_notional=500.0,
    )

    s._net_position = 10.0
    s.on_order_filled(_fill(first, 10.0, closed=True))
    _settle(s)

    assert len(s.submitted) == 1
    assert f"reason={SKIP_BELOW_THRESHOLD}" in s._fake_log.text()


# --------------------------------------------------------------------------
# restart orphan recovery + ownership
# --------------------------------------------------------------------------


def test_owned_restart_orphan_blocks_the_new_entry_until_terminal(monkeypatch):
    """6. An owned limit left by a previous run is retired, not raced."""
    s = _patient_strat(monkeypatch)
    orphan = _reconciled_own_limit("orphan-from-previous-run")
    s._fake_cache.orders_open = lambda instrument_id=None: [orphan]

    _submit_patient(s, units=10.0)

    assert s.canceled == [orphan]
    assert s.submitted == []  # blocked: nothing new while the orphan is unresolved
    assert s._deferred is not None and s._deferred.target_units == 10.0
    assert orphan.client_order_id in s._retiring

    s._fake_cache.orders_open = lambda instrument_id=None: []
    _confirm_cancel(s, orphan)
    _settle(s)

    assert len(s.submitted) == 1  # only now
    assert float(s.submitted[0].kwargs["quantity"]) == pytest.approx(10.0)


def test_orphan_late_fill_reduces_the_deferred_entry(monkeypatch):
    s = _patient_strat(monkeypatch)
    orphan = _reconciled_own_limit("orphan-1")
    s._fake_cache.orders_open = lambda instrument_id=None: [orphan]
    _submit_patient(s, units=10.0)

    s._fake_cache.orders_open = lambda instrument_id=None: []
    s._net_position = 3.0
    s.on_order_filled(_fill(orphan, 3.0, closed=True))
    _settle(s)

    assert float(s.submitted[0].kwargs["quantity"]) == pytest.approx(7.0)


def test_unknown_external_limit_is_never_cancelled(monkeypatch):
    """7. Manual / other-strategy / unknown reconciled orders are not ours."""
    s = _patient_strat(monkeypatch)
    foreign = _external_limit()
    s._fake_cache.orders_open = lambda instrument_id=None: [foreign]

    _submit_patient(s, units=10.0)

    assert s.canceled == []  # we do not touch what we cannot prove we own


def test_unknown_external_limit_blocks_the_new_entry(monkeypatch):
    """8. Blocked rather than racing an order of unknown provenance."""
    s = _patient_strat(monkeypatch)
    foreign = _external_limit()
    s._fake_cache.orders_open = lambda instrument_id=None: [foreign]

    _submit_patient(s, units=10.0)

    assert s.submitted == []
    assert f"reason={SKIP_UNOWNED_RESTING_ORDER}" in s._fake_log.text()

    # once it disappears, entries resume on the next attempt — no sticky block
    s._fake_cache.orders_open = lambda instrument_id=None: []
    _submit_patient(s, units=10.0)
    assert len(s.submitted) == 1


def test_matching_engine_strategy_id_proves_ownership(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _FakeOrder("limit", {}, "c-1")
    order.tags = None
    order.strategy_id = STRATEGY_ID

    owned, why = s._ownership(order)
    assert owned is True and "strategy_id" in why


def test_mismatching_strategy_id_is_foreign_even_with_an_exec_role_tag(monkeypatch):
    """A purpose tag must never override an explicit foreign owner."""
    s = _patient_strat(monkeypatch)
    order = _FakeOrder("limit", {}, "c-2")
    order.tags = [f"EXEC_ROLE={ROLE_PATIENT_LIMIT}", OWNER_TAG]
    order.strategy_id = "SomeoneElse-001"

    owned, why = s._ownership(order)
    assert owned is False
    assert "another strategy" in why


def test_owner_tag_proves_ownership_when_the_engine_reports_no_strategy_id(monkeypatch):
    """The restart-reconciliation path: stable identity, deterministic."""
    s = _patient_strat(monkeypatch)
    order = _reconciled_own_limit("c-3")

    owned, why = s._ownership(order)
    assert owned is True and "OWNER_STRATEGY=" in why


def test_mismatching_owner_tag_is_foreign(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _FakeOrder("limit", {}, "c-4")
    order.tags = ["OWNER_STRATEGY=TrendStrategy:ETHUSDT-PERP.BINANCE"]
    order.strategy_id = None

    owned, why = s._ownership(order)
    assert owned is False and "another strategy" in why


def test_exec_role_alone_cannot_prove_ownership(monkeypatch):
    """EXEC_ROLE describes purpose, not provenance."""
    s = _patient_strat(monkeypatch)
    order = _FakeOrder("limit", {}, "c-5")
    order.tags = [f"EXEC_ROLE={ROLE_PATIENT_LIMIT}"]  # no owner tag, no strategy id
    order.strategy_id = None

    owned, why = s._ownership(order)
    assert owned is False
    assert "purpose is not ownership" in why


def test_client_order_id_from_this_process_proves_ownership(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _FakeOrder("limit", {}, "c-6")
    order.tags = None
    order.strategy_id = None
    s._owned_coids.add("c-6")

    assert s._owns_order(order) is True


def test_manual_and_other_strategy_orders_are_never_cancelled(monkeypatch):
    s = _patient_strat(monkeypatch)
    manual = _FakeOrder("limit", {}, "manual-order")
    manual.tags = None
    manual.strategy_id = None
    other = _external_limit("other-strategy-order")
    tagged_but_foreign = _FakeOrder("limit", {}, "foreign-tagged")
    tagged_but_foreign.tags = [f"EXEC_ROLE={ROLE_PATIENT_LIMIT}"]
    tagged_but_foreign.strategy_id = "SomeoneElse-001"
    s._fake_cache.orders_open = lambda instrument_id=None: [manual, other, tagged_but_foreign]

    _submit_patient(s, units=10.0)

    assert s.canceled == []
    assert s.submitted == []


def test_our_own_orders_carry_a_stable_owner_tag(monkeypatch):
    """The tag must be deterministic across restarts — no per-process value."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)

    assert OWNER_TAG in order.kwargs["tags"]
    assert s._strategy_identity() == "TrendStrategy:BTCUSDT-PERP.BINANCE"
    # a second, independent instance derives the identical identity
    other = _patient_strat(monkeypatch)
    assert other._strategy_identity() == s._strategy_identity()


def test_a_correctly_tagged_order_survives_restart_and_is_adopted(monkeypatch):
    """Simulates a restart: fresh strategy, no _owned_coids, engine reports no
    strategy id — the owner tag alone must allow safe adoption."""
    first = _patient_strat(monkeypatch)
    submitted = _submit_patient(first, units=10.0)
    tags = list(submitted.kwargs["tags"])

    restarted = _patient_strat(monkeypatch)  # new process: nothing remembered
    assert restarted._owned_coids == set()
    reconciled = _FakeOrder("limit", {"quantity": 10.0}, submitted.client_order_id)
    reconciled.tags = tags
    reconciled.strategy_id = None
    restarted._fake_cache.orders_open = lambda instrument_id=None: [reconciled]

    owned, why = restarted._ownership(reconciled)
    assert owned is True and "OWNER_STRATEGY=" in why

    _submit_patient(restarted, units=10.0)
    assert restarted.canceled == [reconciled]  # adopted and retired, not raced
    assert restarted.submitted == []  # and it blocks the new entry until terminal


# --------------------------------------------------------------------------
# control modes
# --------------------------------------------------------------------------


def test_halt_cancels_the_resting_entry_and_keeps_tracking_it(monkeypatch):
    """9. 'Keep positions, no new entries' — a resting entry IS a new entry."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s.risk_state.mode = "halt"

    assert s._apply_control() is True

    assert s.canceled == [order]
    assert s._patient is None
    (retiring,) = s._retiring.values()  # tracking retained, not dropped
    assert retiring.state == STATE_CANCEL_PENDING
    assert retiring.fallback_forbidden is True
    assert not s._fake_clock.armed("patient_timeout_")

    # no fill fallback and no replacement may follow
    _confirm_cancel(s, order)
    _settle(s)
    assert _markets(s) == []
    assert len(s.submitted) == 1


@pytest.mark.parametrize(
    ("control", "reason"),
    [
        ({"mode": "halt"}, "control_halt"),
        ({"mode": "kill"}, "control_kill"),
        ({"flatten_seq": 7}, "control_flatten"),
        ({"market_data_stale": True}, "market_data_stale"),
    ],
)
def test_control_paths_retain_retirement_tracking_until_terminal(monkeypatch, control, reason):
    """10. Kill/flatten may also close positions, but the entry-order cancellation
    stays tracked until the venue confirms it."""
    s = _patient_strat(monkeypatch)
    s._flatten_self = lambda: None
    order = _submit_patient(s, units=10.0)
    for key, value in control.items():
        setattr(s.risk_state, key, value)

    assert s._apply_control() is True

    assert s.canceled == [order]
    (retiring,) = s._retiring.values()
    assert retiring.is_retiring and retiring.fallback_forbidden is True

    _confirm_cancel(s, order)
    _settle(s)
    assert _markets(s) == []
    assert not s._retiring
    assert order.client_order_id in s._tombstones


def test_control_blocks_the_fallback_of_an_already_timed_out_order(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()
    s.risk_state.mode = "halt"

    _confirm_cancel(s, order)
    _settle(s)

    assert _markets(s) == []
    assert "reason=control_halt" in s._fake_log.text()


def test_control_drops_a_parked_replacement(monkeypatch):
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    _submit_patient(s, units=12.0)
    assert s._deferred is not None

    s.risk_state.mode = "halt"
    s._apply_control()
    _confirm_cancel(s, first)
    _settle(s)

    assert s._deferred is None
    assert len(s.submitted) == 1


def test_strategy_stop_retires_the_entry_and_clears_timers(monkeypatch):
    s = _patient_strat(monkeypatch)
    s.cfg().flatten_on_stop = False
    order = _submit_patient(s)

    s.on_stop()

    assert s.canceled == [order]
    assert s._patient is None
    assert s._retiring  # in-flight callbacks during shutdown stay attributable
    assert s._fake_clock.timers == {}


# --------------------------------------------------------------------------
# cancellation failures
# --------------------------------------------------------------------------


def test_synchronous_cancel_exception_is_logged_retried_and_blocks_fallback(monkeypatch):
    """14. A raising cancel_order() must never be swallowed or fall back."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s.cancel_raises = True

    s._on_patient_timeout()

    assert _markets(s) == []
    (retiring,) = s._retiring.values()
    assert retiring.state == STATE_CANCEL_RETRY
    assert retiring.fallback_forbidden is True
    log = s._fake_log.text()
    assert "patient cancel FAILED" in log
    assert order.client_order_id in log
    assert retiring.symbol in log
    assert s._fake_clock.armed("patient_watch_")  # never stuck without a timer

    # the retry eventually succeeds, and still no fallback is taken
    s.cancel_raises = False
    assert s._fake_clock.fire("patient_watch_") is True
    assert s.canceled == [order]
    _confirm_cancel(s, order)
    _settle(s)
    assert _markets(s) == []


def test_cancel_retry_backoff_is_bounded_and_always_rearmed(monkeypatch):
    s = _patient_strat(monkeypatch)
    _submit_patient(s, units=10.0)
    s.cancel_raises = True
    s._on_patient_timeout()

    for _ in range(8):
        assert s._fake_clock.armed("patient_watch_")
        s._fake_clock.fire("patient_watch_")

    (retiring,) = s._retiring.values()
    assert retiring.state == STATE_CANCEL_RETRY
    assert s._fake_clock.armed("patient_watch_")  # still timed, never stuck
    assert _markets(s) == []


def test_cancel_rejection_keeps_fallback_and_replacement_blocked(monkeypatch):
    """15. A refused cancel means the limit may still fill — take nothing."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()

    s.on_order_cancel_rejected(
        SimpleNamespace(client_order_id=order.client_order_id, reason="UNKNOWN_ORDER")
    )

    (retiring,) = s._retiring.values()
    assert retiring.state == STATE_CANCEL_RETRY
    assert retiring.fallback_forbidden is True
    assert _markets(s) == []
    assert s._fake_clock.armed("patient_watch_")

    s._fake_clock.fire("patient_watch_")
    _confirm_cancel(s, order)
    _settle(s)
    assert _markets(s) == []  # still never falls back


def test_cancel_failure_blocks_a_deferred_replacement_too(monkeypatch):
    s = _patient_strat(monkeypatch)
    _submit_patient(s, units=10.0)
    s.cancel_raises = True
    _submit_patient(s, units=12.0)  # supersede while cancel fails

    assert len(s.submitted) == 1
    assert s._deferred is not None
    _settle(s)
    assert len(s.submitted) == 1  # replacement stays blocked


# --------------------------------------------------------------------------
# watchdogs: no nonterminal state without a future action
# --------------------------------------------------------------------------


def _watchdog(s: TrendStrategy) -> bool:
    return s._fake_clock.fire("patient_watch_")


def test_accepted_cancel_with_no_terminal_callback_is_checked_again(monkeypatch):
    """1. cancel_order() succeeded locally but the venue never called back."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)

    s._on_patient_timeout()

    assert s.canceled == [order]
    assert s._fake_clock.armed("patient_watch_")  # armed even though nothing failed
    assert s._unmonitored_states() == []

    _watchdog(s)  # ... and the watchdog actually re-checks

    log = s._fake_log.text()
    assert "patient cancel watchdog expired" in log
    assert order.client_order_id in log


def test_watchdog_retries_cancel_while_the_order_is_still_open(monkeypatch):
    """2. Still open ⇒ retry the cancel, never fall back or replace."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()

    for expected in (2, 3, 4):
        _watchdog(s)
        assert s.canceled == [order] * expected  # the same order, retried
        assert _markets(s) == []
        assert len(s.submitted) == 1
        assert s._unmonitored_states() == []

    (retiring,) = s._retiring.values()
    assert retiring.watchdog_expiries == 3
    assert retiring.state == STATE_CANCEL_PENDING


def test_watchdog_finds_the_order_closed_and_enters_settling(monkeypatch):
    """3. A missed terminal callback still resolves, via settling."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()

    order.is_closed = True  # the venue closed it; we simply never heard
    _watchdog(s)

    (retiring,) = s._retiring.values()
    assert retiring.state == STATE_SETTLING
    assert "watchdog_found_closed" in s._fake_log.text()
    assert _markets(s) == []  # still nothing until settling completes

    _settle(s)
    assert len(_markets(s)) == 1


def test_on_settled_rearms_when_an_order_is_still_open(monkeypatch):
    """4. A terminal callback that disagrees with the order state must not
    resolve — it must re-arm and keep blocking."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()
    # cancel callback arrives, but the order still reports open
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))

    _settle(s)

    assert _markets(s) == []
    assert s._retiring  # not resolved
    assert s._fake_clock.armed("patient_watch_")  # re-armed before returning
    assert s._unmonitored_states() == []
    assert "settle waiting" in s._fake_log.text()

    order.is_closed = True
    _watchdog(s)
    _settle(s)
    assert len(_markets(s)) == 1


def test_multiple_retiring_orders_each_keep_their_own_watchdog(monkeypatch):
    """5. Per-order timers: one retirement must not unmonitor another."""
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    _submit_patient(s, units=12.0)  # supersede -> first retires

    # the venue closes the first, and a second owned orphan appears
    orphan = _reconciled_own_limit("orphan-2", qty=1.0)
    s._fake_cache.orders_open = lambda instrument_id=None: [orphan]
    s._submit_entry_order(_side("BUY"), s.instrument.make_qty(12.0), 100.0, target_units=12.0)

    assert len(s._retiring) == 2
    names = [n for n in s._fake_clock.timers if n.startswith("patient_watch_")]
    assert len(names) == 2  # one per order, not one shared
    assert str(first.client_order_id) in " ".join(names)
    assert "orphan-2" in " ".join(names)
    assert s._unmonitored_states() == []

    # resolving one leaves the other monitored
    first.is_closed = True
    s.on_order_canceled(SimpleNamespace(client_order_id=first.client_order_id))
    _settle(s)
    assert s._unmonitored_states() == []
    assert _markets(s) == []
    assert len(s._retiring) == 2  # the orphan still holds the field


def test_timer_scheduling_failure_is_visible_and_blocks_new_exposure(monkeypatch):
    """6. A failed schedule must never be silently treated as 'always timed'."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)

    def _boom(name, interval, callback):
        raise RuntimeError("clock refused the timer")

    s._fake_clock.set_timer = _boom
    s._on_patient_timeout()

    (retiring,) = s._retiring.values()
    assert retiring.timer_failed is True
    assert retiring.watchdog_armed is False
    assert retiring.fallback_forbidden is True
    assert retiring.fallback_pending is False
    log = s._fake_log.text()
    assert "FAILED to schedule timer" in log
    assert "watchdog NOT armed" in log
    assert _markets(s) == []

    # the periodic control timer is the recovery path — it must not tight-loop
    s._fake_clock.set_timer = _FakeClock.set_timer.__get__(s._fake_clock)
    order.is_closed = True
    s._on_control_timer(None)
    assert "re-arming lost patient watchdog" in s._fake_log.text()
    assert s._unmonitored_states() == []


def test_control_timer_recovery_acts_inline_when_rearming_also_fails(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)

    def _boom(name, interval, callback):
        raise RuntimeError("clock refused the timer")

    s._fake_clock.set_timer = _boom
    s._on_patient_timeout()
    order.is_closed = True

    s._on_control_timer(None)  # re-arm fails -> run the watchdog check inline

    (retiring,) = s._retiring.values()
    assert retiring.state == STATE_SETTLING  # progress was still made
    assert _markets(s) == []  # and never any new exposure


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(lambda s, o: None, id="WORKING"),
        pytest.param(lambda s, o: s._on_patient_timeout(), id="CANCEL_PENDING"),
        pytest.param(
            lambda s, o: (setattr(s, "cancel_raises", True), s._on_patient_timeout()),
            id="CANCEL_RETRY",
        ),
        pytest.param(
            lambda s, o: (
                s._on_patient_timeout(),
                setattr(o, "is_closed", True),
                s.on_order_canceled(SimpleNamespace(client_order_id=o.client_order_id)),
            ),
            id="SETTLING",
        ),
    ],
)
def test_no_nonterminal_state_is_left_without_a_future_action(monkeypatch, arrange):
    """7. WORKING, CANCEL_PENDING, CANCEL_RETRY and SETTLING all stay monitored."""
    s = _patient_strat(monkeypatch)
    arrange(s, _submit_patient(s, units=10.0))

    assert s._unmonitored_states() == []
    assert s._fake_clock.timers  # something is always scheduled


# --------------------------------------------------------------------------
# 5-7: adverse-drift cap
# --------------------------------------------------------------------------


def test_adverse_buy_drift_above_cap_blocks_the_fallback(monkeypatch):
    """5. Price ran up 50 bps against a BUY; the cap is 15 bps."""
    s = _patient_strat(monkeypatch, mark=100.5, max_adverse_bps=15.0)
    order = _submit_patient(s, side="BUY", units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert _markets(s) == []
    assert f"reason={SKIP_ADVERSE_DRIFT}" in s._fake_log.text()


def test_adverse_sell_drift_above_cap_blocks_the_fallback(monkeypatch):
    """6. Price fell 50 bps against a SELL; the cap is 15 bps."""
    s = _patient_strat(monkeypatch, mark=99.5, max_adverse_bps=15.0)
    order = _submit_patient(s, side="SELL", units=-10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert _markets(s) == []
    assert f"reason={SKIP_ADVERSE_DRIFT}" in s._fake_log.text()


def test_favourable_drift_does_not_trip_the_adverse_cap(monkeypatch):
    """7. A BUY whose price fell 50 bps is a gift, not an adverse move."""
    s = _patient_strat(monkeypatch, mark=99.5, max_adverse_bps=15.0)
    order = _submit_patient(s, side="BUY", units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    (fallback,) = _markets(s)
    assert float(fallback.kwargs["quantity"]) == pytest.approx(10.0)
    assert any(t.startswith("FB_DRIFT_BPS=-") for t in fallback.kwargs["tags"])


def test_zero_cap_disables_the_adverse_check(monkeypatch):
    s = _patient_strat(monkeypatch, mark=150.0, max_adverse_bps=0.0)
    order = _submit_patient(s, side="BUY", units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert len(_markets(s)) == 1  # backward-compatible escape hatch


# --------------------------------------------------------------------------
# 8: fresh price
# --------------------------------------------------------------------------


def test_missing_fresh_price_blocks_the_fallback(monkeypatch):
    """8a. No mark, no ticks — never reuse the stale decision price."""
    s = _patient_strat(monkeypatch, mark=None)
    order = _submit_patient(s, units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert _markets(s) == []
    assert f"reason={SKIP_NO_FRESH_PRICE}" in s._fake_log.text()


def test_stale_mark_price_blocks_the_fallback(monkeypatch):
    """8b. A five-minute-old mark is treated as missing, not executable."""
    s = _patient_strat(monkeypatch, mark=100.0, mark_age_secs=300.0)
    order = _submit_patient(s, units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert _markets(s) == []
    assert f"reason={SKIP_NO_FRESH_PRICE}" in s._fake_log.text()


def test_fresh_quote_tick_is_used_when_no_mark_is_available(monkeypatch):
    s = _patient_strat(monkeypatch, mark=None)
    quote = SimpleNamespace(
        ts_event=s._fake_clock.now_ns - 1_000_000_000,
        bid_price=SimpleNamespace(as_double=lambda: 99.0),
        ask_price=SimpleNamespace(as_double=lambda: 101.0),
    )
    s._fake_cache.quote_tick = lambda iid: quote
    order = _submit_patient(s, units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert len(_markets(s)) == 1  # mid of 99/101 == the decision price, zero drift


def test_bar_only_simulation_uses_the_latest_close_as_the_fresh_price(monkeypatch):
    """Backtests have no mark stream; the last bar close is the freshest price
    that exists there, so patient-limit backtests keep working unchanged."""
    s = _patient_strat(monkeypatch, mark=None)
    s.risk_state.mark_watchdog_enabled = False
    s._closes = deque([100.0], maxlen=8)
    order = _submit_patient(s, units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert len(_markets(s)) == 1


def test_live_never_falls_back_to_the_bar_close(monkeypatch):
    """The same setup with the live watchdog on must refuse to price the fallback."""
    s = _patient_strat(monkeypatch, mark=None)
    assert s.risk_state.mark_watchdog_enabled is True
    order = _submit_patient(s, units=10.0, price=100.0)

    _timeout_then_cancel(s, order)

    assert _markets(s) == []


# --------------------------------------------------------------------------
# 10-11: post-only
# --------------------------------------------------------------------------


def test_post_only_is_passed_to_the_patient_limit(monkeypatch):
    """10. The flag must reach the NautilusTrader limit constructor."""
    s = _patient_strat(monkeypatch, post_only=True)
    order = _submit_patient(s, side="BUY", units=10.0, price=100.0)

    assert order.kwargs["post_only"] is True
    assert float(order.kwargs["price"]) == pytest.approx(99.98)  # 2 bps inside
    assert f"EXEC_ROLE={ROLE_PATIENT_LIMIT}" in order.kwargs["tags"]
    assert s._patient.post_only is True


def test_post_only_defaults_to_false_for_backward_compatibility(monkeypatch):
    s = _patient_strat(monkeypatch, post_only=False)
    order = _submit_patient(s)
    assert order.kwargs["post_only"] is False


def test_post_only_rejection_settles_without_resubmitting(monkeypatch):
    """11. A rejected post-only limit must not loop: no cancel, no new order."""
    s = _patient_strat(monkeypatch, post_only=True)
    order = _submit_patient(s, units=10.0)
    order.is_closed = True

    s.on_order_rejected(
        SimpleNamespace(client_order_id=order.client_order_id, reason="POST_ONLY_REJECT")
    )
    _settle(s)

    assert s._patient is None
    assert not s._retiring
    assert _markets(s) == []
    assert s.canceled == []
    assert len(s.submitted) == 1  # only the original limit
    assert not s._fake_clock.armed("patient_timeout_")


# --------------------------------------------------------------------------
# 16: fallback telemetry
# --------------------------------------------------------------------------


def test_fallback_tags_retain_the_original_limit_price_and_decision(monkeypatch):
    """16. A PATIENT_FALLBACK fill must carry the whole decision story."""
    s = _patient_strat(monkeypatch, mark=100.05)
    order = _submit_patient(s, side="BUY", units=10.0, price=100.0)
    decision_ts = s._patient.decision_ts_ns
    limit_price = s._patient.limit_price

    _timeout_then_cancel(s, order)

    (fallback,) = _markets(s)
    tags = fallback.kwargs["tags"]
    assert f"REF_PX={100.0:.12g}" in tags  # original decision price
    assert f"LIMIT_PX={limit_price:.12g}" in tags  # original limit price
    assert f"DEC_TS={decision_ts}" in tags  # original decision timestamp
    assert "FB_REASON=timeout" in tags
    drift = next(t for t in tags if t.startswith("FB_DRIFT_BPS="))
    assert float(drift.removeprefix("FB_DRIFT_BPS=")) == pytest.approx(5.0, rel=1e-3)

    role = classify_exec_role("MARKET", tags=tags)
    assert role == ROLE_PATIENT_FALLBACK
    assert tag_float(tags, "LIMIT_PX=") == pytest.approx(limit_price)


# --------------------------------------------------------------------------
# pure decision layer
# --------------------------------------------------------------------------


def _patient(**overrides) -> PatientOrder:
    kwargs = {
        "client_order_id": "c-1",
        "symbol": "BTCUSDT-PERP",
        "side": "BUY",
        "target_units": 10.0,
        "decision_price": 100.0,
        "decision_ts_ns": 0,
        "limit_price": 99.98,
        "submitted_units": 10.0,
        "fallback_pending": True,
    }
    kwargs.update(overrides)
    return PatientOrder(**kwargs)


def test_evaluate_fallback_is_idempotent_once_resolved():
    patient = _patient(fallback_resolved=True)
    decision = evaluate_fallback(
        patient, current_units=0.0, fresh_price=100.0, fallback_enabled=True
    )
    assert decision.action == ACTION_SKIP and decision.reason == SKIP_ALREADY_RESOLVED


def test_evaluate_fallback_skips_a_residual_below_the_minimum_lot():
    decision = evaluate_fallback(
        _patient(),
        current_units=9.9995,
        fresh_price=100.0,
        fallback_enabled=True,
        min_units=0.001,
    )
    assert decision.reason == SKIP_TARGET_REACHED


def test_drift_sign_convention_is_side_aware():
    assert adverse_drift_bps("BUY", 100.0, 100.5) == pytest.approx(50.0)
    assert adverse_drift_bps("BUY", 100.0, 99.5) == pytest.approx(-50.0)
    assert adverse_drift_bps("SELL", 100.0, 99.5) == pytest.approx(50.25, rel=1e-3)
    assert adverse_drift_bps("SELL", 100.0, 100.5) == pytest.approx(-49.75, rel=1e-3)


# --------------------------------------------------------------------------
# 12: telemetry sign conventions
# --------------------------------------------------------------------------


def test_implementation_shortfall_signs_for_buy_and_sell():
    """12. Positive = worse than the decision price; negative = improvement."""
    # BUY paying up is a cost
    assert implementation_shortfall_bps("BUY", 100.0, 100.2) == pytest.approx(20.0)
    assert implementation_shortfall_quote("BUY", 100.0, 100.2, 5.0) == pytest.approx(1.0)
    # BUY filled below the decision price is price improvement
    assert implementation_shortfall_bps("BUY", 100.0, 99.8) == pytest.approx(-20.0)
    assert implementation_shortfall_quote("BUY", 100.0, 99.8, 5.0) == pytest.approx(-1.0)
    # SELL filled below the decision price is a cost
    assert implementation_shortfall_bps("SELL", 100.0, 99.8) == pytest.approx(20.04, rel=1e-3)
    assert implementation_shortfall_quote("SELL", 100.0, 99.8, 5.0) == pytest.approx(1.0)
    # SELL filled above it is price improvement
    assert implementation_shortfall_bps("SELL", 100.0, 100.2) == pytest.approx(-19.96, rel=1e-3)
    assert implementation_shortfall_quote("SELL", 100.0, 100.2, 5.0) == pytest.approx(-1.0)


def test_tag_int_round_trips_epoch_nanoseconds_exactly():
    """Epoch nanoseconds (~1.8e18) are far past a float's 53-bit integer range,
    so parsing must never route through float."""
    for ns in (
        1_757_112_345_678_901_234,
        1_700_000_000_123_456_789,
        2**62 + 12345,
    ):
        assert tag_int([f"{TAG_DECISION_TS}{ns}"], TAG_DECISION_TS) == ns
        assert int(float(str(ns))) != ns  # the old float path really did lose digits

    assert tag_int(["DEC_TS=-42"], "DEC_TS=") == -42
    assert tag_int(["DEC_TS= 1234 "], "DEC_TS=") == 1234
    assert tag_int(["DEC_TS=42.0"], "DEC_TS=") == 42  # small legacy float form
    assert tag_int(["DEC_TS=4.5"], "DEC_TS=") is None  # not an integer
    assert tag_int(["DEC_TS=1.7e18"], "DEC_TS=") is None  # float cannot be exact here
    assert tag_int(["DEC_TS=nope"], "DEC_TS=") is None
    assert tag_int(None, "DEC_TS=") is None


def test_decision_timestamp_survives_the_tag_round_trip(monkeypatch):
    """The strategy writes DEC_TS and the recorder reads it back, bit for bit."""
    s = _patient_strat(monkeypatch)
    s._fake_clock.now_ns = 1_757_112_345_678_901_234
    order = _submit_patient(s, units=10.0)

    assert tag_int(order.kwargs["tags"], TAG_DECISION_TS) == 1_757_112_345_678_901_234


def test_liquidity_and_role_classification_of_untagged_orders():
    assert liquidity_label("MAKER") == "MAKER"
    assert liquidity_label("NO_LIQUIDITY_SIDE") == "UNKNOWN"
    assert liquidity_label(None) == "UNKNOWN"
    # pre-telemetry orders carry no EXEC_ROLE tag and still classify
    assert classify_exec_role("STOP_MARKET", tags=None) == "HARD_STOP"
    assert classify_exec_role("LIMIT", reduce_only=True, tags=None) == "TAKE_PROFIT"
    assert classify_exec_role("MARKET", tags=None) == "TREND_MARKET"
    assert classify_exec_role("MARKET", tags=["EXEC_ROLE=PATIENT_FALLBACK"]) == "PATIENT_FALLBACK"


# --------------------------------------------------------------------------
# 13-15: store migration + aggregates
# --------------------------------------------------------------------------

_OLD_FILLS_SCHEMA = """
CREATE TABLE fills (
    fill_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT,
    qty REAL, price REAL, kind TEXT, trade_id TEXT, commission REAL,
    reference_price REAL, slippage REAL, reconciliation INTEGER
);
"""


def _legacy_db(tmp_path) -> str:
    db = str(tmp_path / "legacy.db")
    con = sqlite3.connect(db)
    con.executescript(_OLD_FILLS_SCHEMA)
    con.execute(
        "INSERT INTO fills (fill_id, ts, symbol, side, qty, price, kind, trade_id, "
        "commission, reference_price, slippage, reconciliation) "
        "VALUES ('f-legacy', '2026-01-01T00:00:00+00:00', 'BTCUSDT-PERP', 'BUY', "
        "1.0, 100.0, 'TREND', 't1', 0.04, 99.0, 1.0, 0)"
    )
    con.commit()
    con.close()
    return db


def test_additive_migration_upgrades_an_old_schema(tmp_path):
    """13. init_db must add the telemetry columns to an existing VPS database."""
    db = _legacy_db(tmp_path)

    store.init_db(db)

    con = sqlite3.connect(db)
    columns = {r[1] for r in con.execute("PRAGMA table_info(fills)")}
    con.close()
    assert {
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
    } <= columns
    store.init_db(db)  # re-running the migration is a no-op


def test_legacy_rows_with_null_telemetry_stay_readable(tmp_path):
    """14. Pre-migration fills must survive and read back with NULL telemetry."""
    db = _legacy_db(tmp_path)
    store.init_db(db)

    (row,) = store.read_fills(db_path=db)
    assert row["symbol"] == "BTCUSDT-PERP" and row["qty"] == 1.0
    assert row["slippage"] == 1.0  # the old column is untouched
    assert row["liquidity"] is None
    assert row["impl_shortfall_bps"] is None
    assert row["exec_role"] is None

    summary = store.execution_summary(db)
    assert summary["fills_with_telemetry"] == 0
    assert summary["maker_fills"] == 0
    assert summary["shortfall_bps_vw"] is None


def test_execution_summary_on_a_pre_migration_database(tmp_path):
    db = _legacy_db(tmp_path)  # never migrated
    summary = store.execution_summary(db)
    assert summary["maker_fills"] == 0 and summary["shortfall_bps_median"] is None


def test_execution_summary_aggregates_maker_taker_latency_and_shortfall(tmp_path):
    """15. Counts, notionals, volume-weighting, median/p95 and latency."""
    db = str(tmp_path / "c.db")
    store.init_db(db)
    rows = [
        # (id, liquidity, role, qty, price, shortfall_bps, latency_ms)
        ("f1", "MAKER", ROLE_PATIENT_LIMIT, 1.0, 100.0, -2.0, 1_000.0),
        ("f2", "MAKER", ROLE_PATIENT_LIMIT, 1.0, 100.0, -2.0, 3_000.0),
        ("f3", "TAKER", ROLE_PATIENT_FALLBACK, 2.0, 100.0, 20.0, 300_000.0),
        ("f4", "UNKNOWN", "HARD_STOP", 1.0, 100.0, 8.0, 5_000.0),
    ]
    for fill_id, liquidity, role, qty, price, bps, latency in rows:
        store.record_fill(
            fill_id=fill_id,
            symbol="BTCUSDT-PERP",
            side="BUY",
            qty=qty,
            price=price,
            db_path=db,
            liquidity=liquidity,
            exec_role=role,
            impl_shortfall_bps=bps,
            impl_shortfall_quote=bps / 1e4 * price * qty,
            decision_to_fill_ms=latency,
        )

    summary = store.execution_summary(db)

    assert summary["fills_with_telemetry"] == 4
    assert summary["maker_fills"] == 2
    assert summary["taker_fills"] == 1
    assert summary["unknown_liquidity_fills"] == 1
    assert summary["maker_notional"] == pytest.approx(200.0)
    assert summary["taker_notional"] == pytest.approx(200.0)
    assert summary["patient_fallback_fills"] == 1
    # volume weighted: (-2*100 + -2*100 + 20*200 + 8*100) / 500
    assert summary["shortfall_bps_vw"] == pytest.approx(8.8)
    assert summary["shortfall_bps_median"] == pytest.approx(3.0)  # (-2, -2, 8, 20)
    assert summary["shortfall_bps_p95"] == pytest.approx(20.0)
    assert summary["latency_ms_median"] == pytest.approx(4_000.0)
    assert summary["latency_ms_p95"] == pytest.approx(300_000.0)

    by_role = summary["shortfall_bps_by_role"]
    assert by_role[ROLE_PATIENT_LIMIT]["fills"] == 2
    assert by_role[ROLE_PATIENT_LIMIT]["shortfall_bps_vw"] == pytest.approx(-2.0)
    assert by_role[ROLE_PATIENT_FALLBACK]["shortfall_bps_vw"] == pytest.approx(20.0)


def test_performance_summary_exposes_execution_aggregates(tmp_path):
    db = str(tmp_path / "p.db")
    store.init_db(db)
    performance = store.performance_summary(1000.0, 1000.0, 0.0, db)
    assert "execution" in performance
    assert performance["execution"]["maker_fills"] == 0
