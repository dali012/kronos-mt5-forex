"""Funding fetcher/loader round-trip + new TrendStrategy config/helpers (CI coverage
for the realism features that otherwise only run live)."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backtest.trend import load_funding
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
    assert d.flatten_on_stop is False


def test_strategy_as_float_helper():
    class _M:
        def as_double(self):
            return 1.5

    assert TrendStrategy._as_float(None) == 0.0
    assert TrendStrategy._as_float("2.5") == 2.5
    assert TrendStrategy._as_float(_M()) == 1.5
    assert TrendStrategy._as_float("not-a-number") == 0.0


# --- protection idempotency (the live duplicate-stop bug) --------------------
class _FakeOrder:
    def __init__(self, coid: str) -> None:
        self.client_order_id = coid
        self.is_closed = False
        self.is_reduce_only = True


class _FakeFactory:
    def __init__(self) -> None:
        self.n = 0

    def _mk(self, tag: str) -> _FakeOrder:
        self.n += 1
        return _FakeOrder(f"{tag}-{self.n}")

    def stop_market(self, **_kw):
        return self._mk("stop")

    def limit(self, **_kw):
        return self._mk("limit")


class _Num(float):
    """Stands in for Quantity/Price: float supports <= and str() for the signature."""


class _FakeInstrument:
    def make_qty(self, x):
        return _Num(x)

    def make_price(self, x):
        return _Num(x)


def _protection_strat(monkeypatch, use_tp: bool = False) -> TrendStrategy:
    """A TrendStrategy with just enough wired up to exercise _set_protection
    without a broker/engine (bypasses Strategy.__init__ via __new__). order_factory,
    cache and instrument_id are read-only Cython properties on the base class, so we
    shadow them with properties on the TrendStrategy subclass (precedes base in MRO)."""
    s = TrendStrategy.__new__(TrendStrategy)
    cfg = SimpleNamespace(use_stop_loss=True, use_take_profit=use_tp, tp_pct=0.5, ppy=365)
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
    s._dynamic_stop_pct = 0.20
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
