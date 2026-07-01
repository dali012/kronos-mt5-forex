"""Funding fetcher/loader round-trip + new TrendStrategy config/helpers (CI coverage
for the realism features that otherwise only run live)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model.data import MarkPriceUpdate
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price

from backtest.trend import load_funding
from kronos_mt5.strategies.risk_manager import RiskState
from kronos_mt5.strategies.trend_strategy import TrendStrategy, TrendStrategyConfig
from scripts.download_funding import _ms, write_csv

JAN1 = 1_704_067_200_000  # 2024-01-01 00:00 UTC in ms
JAN2 = 1_704_153_600_000  # 2024-01-02 00:00 UTC in ms


def test_download_funding_ms():
    assert _ms("2024-01-01") == JAN1


def test_funding_write_load_roundtrip(tmp_path):
    rows = [
        {"symbol": "BTCUSDT", "fundingTime": JAN1, "fundingRate": "0.0001", "markPrice": "42000"},
        {"symbol": "BTCUSDT", "fundingTime": JAN2, "fundingRate": "-0.0002", "markPrice": "42100"},
    ]
    write_csv("BTCUSDT", rows, tmp_path)
    s = load_funding("BTCUSDT", tmp_path)
    assert s is not None and len(s) == 2
    assert isinstance(s.index, pd.DatetimeIndex)
    assert abs(s.iloc[0] - 0.0001) < 1e-12 and abs(s.iloc[1] + 0.0002) < 1e-12


def test_funding_sums_intraday_events(tmp_path):
    # multiple funding events on the SAME day are summed into one daily cost
    rows = [
        {"symbol": "ETHUSDT", "fundingTime": JAN1, "fundingRate": "0.0001", "markPrice": ""},
        {"symbol": "ETHUSDT", "fundingTime": JAN1 + 28_800_000, "fundingRate": "0.0002", "markPrice": ""},
    ]
    write_csv("ETHUSDT", rows, tmp_path)
    s = load_funding("ETHUSDT", tmp_path)
    assert len(s) == 1 and abs(s.iloc[0] - 0.0003) < 1e-12


def test_load_funding_missing_returns_none(tmp_path):
    assert load_funding("NOPE", tmp_path) is None


def test_trend_config_patient_limit_and_safe_defaults():
    cfg = TrendStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL",
        use_patient_limit=True,
        patient_limit_offset_bps=3.0,
        patient_limit_timeout_secs=120,
    )
    assert cfg.use_patient_limit and cfg.patient_limit_offset_bps == 3.0
    assert cfg.patient_limit_market_fallback is True  # sane default: don't get stuck unfilled

    # all the new realism features are OFF by default (defaults preserve old behavior)
    d = TrendStrategyConfig(instrument_id="X.Y", bar_type="X.Y-1-DAY-LAST-EXTERNAL")
    assert not d.use_patient_limit and not d.use_vol_stop and not d.use_correlation_scaling
    assert not d.funding_filter_enabled and not d.cost_aware_rebalance
    assert not d.use_portfolio_allocator and not d.funding_continuous_sizing
    assert not d.shadow_enabled and d.portfolio_target_vol == 0.10
    assert not d.use_trailing_stop and d.trailing_activation_r == 1.0
    assert d.flatten_on_stop is False


def test_strategy_as_float_helper():
    class _M:
        def as_double(self):
            return 1.5

    assert TrendStrategy._as_float(None) == 0.0
    assert TrendStrategy._as_float("2.5") == 2.5
    assert TrendStrategy._as_float(_M()) == 1.5
    assert TrendStrategy._as_float("not-a-number") == 0.0


def _mark(iid: InstrumentId, value: str) -> MarkPriceUpdate:
    return MarkPriceUpdate(iid, Price.from_str(value), ts_event=0, ts_init=0)


def test_mark_watchdog_fails_closed_recovers_and_detects_staleness():
    btc = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    eth = InstrumentId.from_str("ETHUSDT-PERP.BINANCE")
    iids = (btc, eth)
    risk = RiskState()

    assert not risk.halted  # backtests/live setups without a monitor remain unchanged
    risk.enable_mark_watchdog(iids)
    assert risk.halted and risk.stale_mark_symbols == ("BTCUSDT-PERP", "ETHUSDT-PERP")

    risk.update_mark(_mark(btc, "63000"), received_ns=1_000_000_000)
    risk.update_mark(_mark(eth, "1700"), received_ns=2_000_000_000)
    assert risk.evaluate_mark_freshness(iids, now_ns=3_000_000_000, stale_after_secs=15) == ()
    assert not risk.halted
    assert risk.mark_prices[str(btc)] == 63000.0
    assert risk.mark_age_secs == {"BTCUSDT-PERP": 2.0, "ETHUSDT-PERP": 1.0}

    stale = risk.evaluate_mark_freshness(iids, now_ns=18_000_000_000, stale_after_secs=15)
    assert stale == ("BTCUSDT-PERP", "ETHUSDT-PERP")
    assert risk.halted and risk.mode == "run"  # watchdog does not overwrite manual control


def test_mark_watchdog_only_flags_the_stale_symbol():
    btc = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    eth = InstrumentId.from_str("ETHUSDT-PERP.BINANCE")
    risk = RiskState()
    risk.enable_mark_watchdog((btc, eth))
    risk.update_mark(_mark(btc, "63000"), received_ns=20_000_000_000)
    risk.update_mark(_mark(eth, "1700"), received_ns=1_000_000_000)

    assert risk.evaluate_mark_freshness(
        (btc, eth), now_ns=21_000_000_000, stale_after_secs=15
    ) == ("ETHUSDT-PERP",)


def test_correlation_metrics_count_effective_bets_not_symbols():
    risk = RiskState()
    base = np.linspace(-0.02, 0.02, 90)
    for symbol in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "LTC", "LINK"):
        risk.update_returns(symbol, base.copy())

    metrics = risk.correlation_metrics(window=90, threshold=0.65, min_scalar=0.50)
    assert metrics["series_count"] == 8
    assert metrics["average_abs_correlation"] == pytest.approx(1.0)
    assert metrics["effective_bets"] == pytest.approx(1.0)
    assert metrics["risk_scalar"] == pytest.approx(0.50)
    assert metrics["volatility_regime"] == "LOW_VOL"


def test_portfolio_allocator_activates_one_common_scale_next_cycle():
    risk = RiskState()
    returns = np.linspace(-0.02, 0.02, 90)
    kwargs = dict(
        expected_symbols=("BTC", "ETH"),
        window=90,
        target_vol=0.10,
        ppy=365,
        shrinkage=0.0,
        min_scale=0.10,
        max_scale=1.25,
        max_gross=2.0,
        min_observations=30,
        scale_up_alpha=1.0,
        scale_deadband=0.0,
    )
    first_btc = risk.publish_portfolio_target(
        "live", "BTC", 1.0, returns, "day-1", **kwargs
    )
    first_eth = risk.publish_portfolio_target(
        "live", "ETH", 1.0, returns, "day-1", **kwargs
    )
    computed = risk.allocation_scales["live"]
    assert first_btc == first_eth == 1.0
    assert 0.10 <= computed < 1.0
    assert risk.allocation_metrics["live"]["ready"] is True

    # Both sibling strategies capture the same lagged scale before day-2 is finalized.
    second_btc = risk.publish_portfolio_target(
        "live", "BTC", 1.0, returns, "day-2", **kwargs
    )
    second_eth = risk.publish_portfolio_target(
        "live", "ETH", 1.0, returns, "day-2", **kwargs
    )
    assert second_btc == pytest.approx(computed)
    assert second_eth == pytest.approx(computed)


def test_portfolio_allocator_gross_cap_overrides_minimum_scale():
    risk = RiskState()
    returns = np.linspace(-0.01, 0.01, 60)
    for symbol in ("BTC", "ETH"):
        risk.publish_portfolio_target(
            "gross",
            symbol,
            4.0,
            returns,
            "day-1",
            ("BTC", "ETH"),
            window=60,
            target_vol=1.0,
            ppy=365,
            shrinkage=0.25,
            min_scale=0.50,
            max_scale=1.25,
            max_gross=1.0,
            min_observations=30,
            scale_up_alpha=1.0,
            scale_deadband=0.0,
        )
    metrics = risk.allocation_metrics["gross"]
    assert metrics["gross_before"] == pytest.approx(4.0)
    assert metrics["scale"] == pytest.approx(0.25)
    assert metrics["gross_after"] == pytest.approx(1.0)


def test_portfolio_allocator_smooths_small_scale_increases():
    risk = RiskState()
    quiet = np.linspace(-0.001, 0.001, 60)
    risk.publish_portfolio_target(
        "smooth",
        "BTC",
        0.1,
        quiet,
        "day-1",
        ("BTC",),
        window=60,
        target_vol=0.50,
        ppy=365,
        shrinkage=0.25,
        min_scale=0.25,
        max_scale=1.25,
        max_gross=2.0,
        min_observations=30,
        scale_up_alpha=0.20,
        scale_deadband=0.10,
    )
    metrics = risk.allocation_metrics["smooth"]
    assert metrics["candidate_scale"] == pytest.approx(1.25)
    assert metrics["scale"] == pytest.approx(1.0)  # 5% increase is inside 10% deadband


def test_continuous_funding_sizing_is_smooth_and_never_boosts():
    strategy = TrendStrategy.__new__(TrendStrategy)
    strategy.cfg = lambda: SimpleNamespace(
        funding_rate_limit=0.0010,
        funding_rate_soft_limit=0.0001,
        funding_min_scalar=0.0,
    )
    assert strategy._funding_scalar(1.0, 0.0001, continuous=True) == 1.0
    assert strategy._funding_scalar(1.0, 0.00055, continuous=True) == pytest.approx(0.5)
    assert strategy._funding_scalar(1.0, 0.0010, continuous=True) == 0.0
    assert strategy._funding_scalar(-1.0, -0.00055, continuous=True) == pytest.approx(0.5)
    assert strategy._funding_scalar(1.0, -0.0010, continuous=True) == 1.0


# --- protection idempotency (the live duplicate-stop bug) --------------------
class _FakeOrder:
    def __init__(self, coid: str, kind: str, kwargs: dict) -> None:
        self.client_order_id = coid
        self.kind = kind
        self.kwargs = kwargs
        self.is_closed = False
        self.is_reduce_only = True


class _FakeFactory:
    def __init__(self) -> None:
        self.n = 0

    def _mk(self, tag: str, kwargs: dict) -> _FakeOrder:
        self.n += 1
        return _FakeOrder(f"{tag}-{self.n}", tag, kwargs)

    def stop_market(self, **kwargs):
        return self._mk("stop", kwargs)

    def limit(self, **kwargs):
        return self._mk("limit", kwargs)

    def trailing_stop_market(self, **kwargs):
        return self._mk("trail", kwargs)


class _Num(float):
    """Stands in for Quantity/Price: float supports <= and str() for the signature."""


class _FakeInstrument:
    def make_qty(self, x):
        return _Num(x)

    def make_price(self, x):
        return _Num(x)


def _protection_strat(
    monkeypatch,
    use_tp: bool = False,
    use_trail: bool = False,
) -> TrendStrategy:
    """A TrendStrategy with just enough wired up to exercise _set_protection
    without a broker/engine (bypasses Strategy.__init__ via __new__). order_factory,
    cache and instrument_id are read-only Cython properties on the base class, so we
    shadow them with properties on the TrendStrategy subclass (precedes base in MRO)."""
    s = TrendStrategy.__new__(TrendStrategy)
    cfg = SimpleNamespace(
        use_stop_loss=True,
        use_take_profit=use_tp,
        tp_pct=0.5,
        use_trailing_stop=use_trail,
        trailing_activation_r=1.0,
        trailing_vol_mult=3.0,
        min_trailing_pct=0.005,
        max_trailing_pct=0.10,
        use_vol_stop=True,
        stop_pct=0.20,
        stop_vol_mult=4.0,
        min_stop_pct=0.08,
        max_stop_pct=0.30,
        ppy=365,
    )
    s.cfg = lambda: cfg
    s.instrument = _FakeInstrument()
    s._fake_factory = _FakeFactory()
    s._fake_cache = SimpleNamespace(orders_open=lambda instrument_id=None: [])
    monkeypatch.setattr(TrendStrategy, "order_factory", property(lambda self: self._fake_factory))
    monkeypatch.setattr(TrendStrategy, "cache", property(lambda self: self._fake_cache))
    monkeypatch.setattr(
        TrendStrategy, "instrument_id", property(lambda self: "BTCUSDT.BINANCE"), raising=False
    )
    s._protective_orders = []
    s._protection_signature = None
    s._protection_position_key = None
    s._dynamic_stop_pct = 0.20
    s._dynamic_trailing_pct = 0.05
    s.risk_state = SimpleNamespace(mark_prices={})
    s.submitted = []
    s.canceled = []
    s.submit_order = s.submitted.append
    s.cancel_order = lambda o: (setattr(o, "is_closed", True), s.canceled.append(o))
    return s


def test_set_protection_idempotent_no_duplicate_stops(monkeypatch):
    # the VPS bug: repeated refreshes (restart reconciliation, overlapping events)
    # must NOT stack duplicate stops.
    s = _protection_strat(monkeypatch)
    s._set_protection(2.0, 100.0)
    assert len(s.submitted) == 1
    for _ in range(5):  # burst of identical refreshes
        s._set_protection(2.0, 100.0)
    assert len(s.submitted) == 1  # still exactly one stop, no duplicates


def test_set_protection_replaces_on_position_resize(monkeypatch):
    s = _protection_strat(monkeypatch)
    s._set_protection(2.0, 100.0)
    s._set_protection(3.0, 100.0)  # qty changed -> cancel old (incl. in-flight), submit new
    assert len(s.submitted) == 2
    assert s.submitted[0].is_closed and not s.submitted[1].is_closed


def test_set_protection_flat_cancels_all(monkeypatch):
    s = _protection_strat(monkeypatch)
    s._set_protection(2.0, 100.0)
    s._set_protection(0.0, 100.0)  # position flat -> no protection should remain
    assert s.submitted[0].is_closed
    assert s._protection_signature is None
    assert not s._has_live_protection()


def test_entry_fill_reconciles_protection_after_position_resize(monkeypatch):
    s = _protection_strat(monkeypatch)
    pending = SimpleNamespace(client_order_id="entry-1")
    s._pending_entry_order = pending
    refreshed = []
    s._refresh_protection = lambda: refreshed.append(True)
    event = SimpleNamespace(instrument_id=s.instrument_id, client_order_id="entry-1")

    s.on_order_filled(event)

    assert s._pending_entry_order is None
    assert refreshed == [True]


def test_native_vol_trail_keeps_hard_stop_and_arms_at_one_r(monkeypatch):
    s = _protection_strat(monkeypatch, use_trail=True)
    s._set_protection(2.0, 100.0)

    assert [o.kind for o in s.submitted] == ["stop", "trail"]
    hard, trail = s.submitted
    assert float(hard.kwargs["trigger_price"]) == 80.0
    assert float(trail.kwargs["activation_price"]) == 120.0
    assert float(trail.kwargs["trailing_offset"]) == 500.0  # 5% = 500 bps
    assert trail.kwargs["trailing_offset_type"].name == "BASIS_POINTS"
    assert trail.kwargs["trigger_type"].name == "MARK_PRICE"
    assert trail.kwargs["reduce_only"] is True


def test_native_vol_trail_short_side_and_already_activated(monkeypatch):
    s = _protection_strat(monkeypatch, use_trail=True)
    s.risk_state.mark_prices["BTCUSDT.BINANCE"] = 75.0  # beyond short's +1R level (80)
    s._set_protection(-2.0, 100.0)

    hard, trail = s.submitted
    assert float(hard.kwargs["trigger_price"]) == 120.0
    assert hard.kwargs["order_side"].name == "BUY"
    assert trail.kwargs["activation_price"] is None  # arm from current mark, avoid -2021
    assert trail.kwargs["order_side"].name == "BUY"


def test_native_trail_watermark_is_not_reset_by_daily_vol_update(monkeypatch):
    s = _protection_strat(monkeypatch, use_trail=True)
    s._set_protection(2.0, 100.0)
    original_orders = list(s.submitted)

    s._dynamic_stop_pct = 0.10
    s._dynamic_trailing_pct = 0.02
    s._set_protection(2.0, 100.0)

    assert s.submitted == original_orders
    assert not s.canceled


def test_trailing_distance_uses_daily_vol_and_binance_bounds(monkeypatch):
    s = _protection_strat(monkeypatch, use_trail=True)
    s._update_dynamic_stop(inst_vol=0.38)
    expected = 3.0 * 0.38 / (365**0.5)
    assert abs(s._dynamic_trailing_pct - expected) < 1e-12

    s.cfg().trailing_vol_mult = 100.0
    s._update_dynamic_stop(inst_vol=1.0)
    assert s._dynamic_trailing_pct == 0.10  # Binance maximum callback rate


def test_trailing_callback_rate_uses_binance_precision(monkeypatch):
    s = _protection_strat(monkeypatch, use_trail=True)
    s._dynamic_trailing_pct = 0.092911670832

    s._set_protection(2.0, 100.0)

    trail = s.submitted[1]
    assert str(trail.kwargs["trailing_offset"]) == "930.0"  # 9.3%, exact basis points


def test_funding_rate_updater_constructs():
    # regression: self.state collided with Component's read-only .state property and
    # crashed node startup whenever BINANCE_FUNDING_FILTER_ENABLED=true.
    from kronos_mt5.live.run_trend_binance import (
        FundingRateState,
        FundingRateUpdater,
        FundingRateUpdaterConfig,
    )

    st = FundingRateState()
    updater = FundingRateUpdater(FundingRateUpdaterConfig(symbols=("BTCUSDT", "ETHUSDT")), st)
    assert updater.funding_state is st
    assert updater.state == 0  # Component lifecycle property intact (pre-initialized)


def test_funding_rate_state_normalizes_nautilus_perpetual_symbols():
    from kronos_mt5.live.run_trend_binance import FundingRateState

    state = FundingRateState()
    state.rates["BTCUSDT"] = 0.0001
    assert state.get("BTCUSDT") == 0.0001
    assert state.get("BTCUSDT-PERP") == 0.0001
    assert state.get("BTCUSDT-PERP.BINANCE") == 0.0001


def test_restore_control_state_does_not_replay_flatten(tmp_path):
    from kronos_mt5.companion import store
    from kronos_mt5.live.run_trend_binance import _restore_control_state

    db = str(tmp_path / "control.db")
    store.init_db(db)
    store.set_mode("run", db)
    assert store.request_flatten("all", db) == 1
    risk = RiskState()

    _restore_control_state(risk, db)

    assert risk.mode == "run"
    assert risk.flatten_seq == 1
    assert risk.flatten_target == "all"


def test_strategy_start_syncs_persisted_flatten_cursor():
    strategy = TrendStrategy.__new__(TrendStrategy)
    strategy.risk_state = SimpleNamespace(flatten_seq=7)
    strategy._last_flatten_seq = 0

    strategy._sync_flatten_cursor()

    assert strategy._last_flatten_seq == 7


class _FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def timestamp_ns(self) -> int:
        return self.ns


def _watchdog_stub(db_path: str, iids):
    """A MarkPriceMonitor-shaped object that exercises _evaluate_freshness without
    standing up the full Nautilus Strategy machinery."""
    return SimpleNamespace(
        risk_state=RiskState(),
        clock=_FakeClock(),
        log=SimpleNamespace(error=lambda *a, **k: None, warning=lambda *a, **k: None),
        config=SimpleNamespace(stale_after_secs=15, incident_min_secs=10, db_path=db_path),
        _iids=iids,
        _stale_since_ns=None,
        _incident_open=False,
        _warmed_up=False,
    )


def test_mark_watchdog_debounces_transient_flaps(tmp_path):
    """Entry gate reacts instantly; incidents only log when staleness persists."""
    from kronos_mt5.companion import store
    from kronos_mt5.strategies.risk_manager import MarkPriceMonitor

    db = str(tmp_path / "watchdog.db")
    store.init_db(db)
    btc = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    eth = InstrumentId.from_str("ETHUSDT-PERP.BINANCE")
    iids = (btc, eth)
    ev = MarkPriceMonitor._evaluate_freshness

    m = _watchdog_stub(db, iids)
    m.risk_state.enable_mark_watchdog(iids)
    sec = 1_000_000_000

    # Startup warmup: stale until marks arrive, but this is never an incident.
    m.clock.ns = 100 * sec
    ev(m)
    assert not store.has_open_incident("MARK_STALE", db)
    m.risk_state.update_mark(_mark(btc, "63000"), received_ns=m.clock.ns)
    m.risk_state.update_mark(_mark(eth, "1700"), received_ns=m.clock.ns)
    ev(m)  # first fresh read -> warmed up
    assert m._warmed_up and not store.has_open_incident("MARK_STALE", db)

    # Transient flap: goes stale then recovers before incident_min_secs -> no incident.
    m.clock.ns = 120 * sec  # marks now 20s old (> 15s stale threshold)
    ev(m)
    assert m.risk_state.market_data_stale  # entry gate closed immediately (fail-safe)
    assert not store.has_open_incident("MARK_STALE", db)  # but not yet logged
    m.risk_state.update_mark(_mark(btc, "63010"), received_ns=125 * sec)
    m.risk_state.update_mark(_mark(eth, "1701"), received_ns=125 * sec)
    m.clock.ns = 125 * sec
    ev(m)
    assert not store.has_open_incident("MARK_STALE", db)

    # Persistent outage: stale beyond incident_min_secs -> exactly one incident.
    m.clock.ns = 145 * sec  # marks 20s old again
    ev(m)
    assert not store.has_open_incident("MARK_STALE", db)  # debounce not elapsed
    m.clock.ns = 156 * sec  # 11s of continuous staleness
    ev(m)
    assert store.has_open_incident("MARK_STALE", db)

    # Recovery resolves the single incident.
    m.risk_state.update_mark(_mark(btc, "63020"), received_ns=156 * sec)
    m.risk_state.update_mark(_mark(eth, "1702"), received_ns=156 * sec)
    ev(m)
    assert not store.has_open_incident("MARK_STALE", db)
    assert len([r for r in store.read_incidents(db_path=db) if r["kind"] == "MARK_STALE"]) == 1
