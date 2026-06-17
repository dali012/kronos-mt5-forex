"""Funding fetcher/loader round-trip + new TrendStrategy config/helpers (CI coverage
for the realism features that otherwise only run live)."""
from __future__ import annotations

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
