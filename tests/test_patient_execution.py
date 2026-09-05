"""Patient-limit execution: lifecycle, fresh-price/drift guards and telemetry.

Everything here runs on deterministic doubles — no Binance, no engine, no network.
The strategy is built the same way `test_strategy_extras.py` does it: bypass the
Cython `Strategy.__init__` with `__new__` and shadow the read-only engine
properties with plain properties on the Python subclass.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from types import SimpleNamespace

import pytest

from kronos_mt5.companion import store
from kronos_mt5.execution import (
    ACTION_SKIP,
    ROLE_PATIENT_FALLBACK,
    ROLE_PATIENT_LIMIT,
    SKIP_ADVERSE_DRIFT,
    SKIP_ALREADY_RESOLVED,
    SKIP_NO_FRESH_PRICE,
    SKIP_TARGET_REACHED,
    STATE_CANCEL_PENDING,
    STATE_WORKING,
    PatientOrder,
    adverse_drift_bps,
    classify_exec_role,
    evaluate_fallback,
    implementation_shortfall_bps,
    implementation_shortfall_quote,
    liquidity_label,
)
from kronos_mt5.strategies.trend_strategy import TrendStrategy

# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


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
        self.side = kwargs.get("order_side")
        self.quantity = kwargs.get("quantity")
        self.tags = kwargs.get("tags")
        self.price = kwargs.get("price")


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

    def timestamp_ns(self) -> int:
        return self.now_ns

    def set_timer(self, name, interval, callback):
        self.timers[name] = callback

    def cancel_timer(self, name):
        self.timers.pop(name, None)


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
    s._closes = deque([100.0], maxlen=8)
    s._protective_orders = []
    s._protection_signature = None
    s._protection_position_key = None
    s._dynamic_stop_pct = 0.20
    s._last_flatten_seq = 0
    s.submitted: list[_FakeOrder] = []
    s.canceled: list[_FakeOrder] = []
    s.submit_order = s.submitted.append
    s.cancel_order = s.canceled.append
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
# 1-4: lifecycle ordering
# --------------------------------------------------------------------------


def test_timeout_requests_cancel_without_submitting_a_market_order(monkeypatch):
    """1. The timeout must never place the fallback itself."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s)

    s._on_patient_timeout()

    assert s.canceled == [order]
    assert _markets(s) == []  # nothing taken before the venue confirms
    assert s._patient.state == STATE_CANCEL_PENDING
    assert s._patient.fallback_pending is True


def test_partial_fill_falls_back_only_for_the_remaining_delta(monkeypatch):
    """2. A 4-unit partial of a 10-unit target must fall back for 6, not 10."""
    s = _patient_strat(monkeypatch, net_position=0.0)
    order = _submit_patient(s, units=10.0)

    s._net_position = 4.0
    s.on_order_filled(_fill(order, 4.0, closed=False))
    assert s._patient is not None  # a partial must NOT clear the state
    assert s._patient.state == STATE_WORKING

    s._on_patient_timeout()
    order.is_closed = True
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))

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

    assert _markets(s) == []
    assert s._patient is None
    assert f"reason={SKIP_TARGET_REACHED}" in s._fake_log.text()


def test_duplicate_terminal_callbacks_create_exactly_one_fallback(monkeypatch):
    """4. Repeated cancel/fill callbacks must be idempotent."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0)
    s._on_patient_timeout()
    order.is_closed = True

    event = SimpleNamespace(client_order_id=order.client_order_id)
    s.on_order_canceled(event)
    s.on_order_canceled(event)  # duplicate from the venue
    s.on_order_expired(event)  # and a conflicting terminal
    s.on_order_filled(_fill(order, 10.0, closed=True))  # and a late fill

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
# 5-7: adverse-drift cap
# --------------------------------------------------------------------------


def _timeout_then_cancel(s, order):
    s._on_patient_timeout()
    order.is_closed = True
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))


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
# 8-9: fresh price and control gates
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


def test_untracked_resting_entry_limit_is_cancelled_before_a_new_entry(monkeypatch):
    """A limit left working by a previous process must not fill on top of us."""
    s = _patient_strat(monkeypatch)
    orphan = _FakeOrder("limit", {"order_side": None}, "orphan-from-previous-run")
    orphan.order_type = SimpleNamespace(name="LIMIT")
    orphan.is_reduce_only = False
    protective = _FakeOrder("stop", {}, "protective")
    protective.order_type = SimpleNamespace(name="STOP_MARKET")
    protective.is_reduce_only = True
    s._fake_cache.orders_open = lambda instrument_id=None: [orphan, protective]

    _submit_patient(s, units=10.0)

    assert s.canceled == [orphan]  # the reduce-only stop is left alone


@pytest.mark.parametrize(
    ("control", "expected"),
    [
        ({"mode": "halt"}, "control_halt"),
        ({"mode": "kill"}, "control_kill"),
        ({"flatten_seq": 7}, "control_flatten"),
        ({"market_data_stale": True}, "market_data_stale"),
    ],
)
def test_control_modes_block_the_fallback(monkeypatch, control, expected):
    """9. Halt, kill, flatten and stale market data all veto new exposure."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s, units=10.0, price=100.0)
    s._on_patient_timeout()
    for key, value in control.items():
        setattr(s.risk_state, key, value)

    order.is_closed = True
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))

    assert _markets(s) == []
    assert f"reason={expected}" in s._fake_log.text()
    assert s._patient is None


def test_control_takeover_cancels_a_resting_patient_limit(monkeypatch):
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s)
    s.risk_state.market_data_stale = True
    s._flatten_self = lambda: None

    assert s._apply_control() is True
    assert s.canceled == [order]
    assert s._patient is None

    # a terminal callback arriving afterwards is a harmless no-op
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))
    assert _markets(s) == []


def test_strategy_stop_clears_patient_state(monkeypatch):
    s = _patient_strat(monkeypatch)
    s.cfg().flatten_on_stop = False
    order = _submit_patient(s)

    s.on_stop()

    assert s.canceled == [order]
    assert s._patient is None
    assert s._fake_clock.timers == {}


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


def test_post_only_rejection_clears_state_without_resubmitting(monkeypatch):
    """11. A rejected post-only limit must not loop: no cancel, no new order."""
    s = _patient_strat(monkeypatch, post_only=True)
    order = _submit_patient(s, units=10.0)

    s.on_order_rejected(
        SimpleNamespace(client_order_id=order.client_order_id, reason="POST_ONLY_REJECT")
    )

    assert s._patient is None
    assert s._patient_order is None
    assert _markets(s) == []
    assert s.canceled == []
    assert len(s.submitted) == 1  # only the original limit
    assert s._fake_clock.timers == {}


def test_cancel_rejection_disables_fallback_and_keeps_retrying_cancel(monkeypatch):
    """A refused cancel means the limit may still fill — never take on top of it."""
    s = _patient_strat(monkeypatch)
    order = _submit_patient(s)
    s._on_patient_timeout()

    s.on_order_cancel_rejected(
        SimpleNamespace(client_order_id=order.client_order_id, reason="UNKNOWN_ORDER")
    )

    assert s._patient.state == STATE_WORKING
    assert s._patient.fallback_forbidden is True
    assert _markets(s) == []

    s._on_patient_timeout()  # retry the cancel...
    assert s.canceled == [order, order]
    order.is_closed = True
    s.on_order_canceled(SimpleNamespace(client_order_id=order.client_order_id))
    assert _markets(s) == []  # ...but still never fall back
    assert s._patient is None


def test_new_entry_supersedes_and_cancels_a_stale_patient_limit(monkeypatch):
    s = _patient_strat(monkeypatch)
    first = _submit_patient(s, units=10.0)
    second = _submit_patient(s, units=12.0)

    assert s.canceled == [first]
    assert s._patient_order is second
    assert s._patient.target_units == 12.0


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
