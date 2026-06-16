"""Smoke tests — first thing to get green (Stage 1).

These run on CPU with no broker. The pure risk/signal tests are instant; the
brain test loads Kronos and runs real inference, so it is marked `slow`
(deselect with `-m "not slow"`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from kronos_mt5.brain.kronos_predictor import ForecastResult
from kronos_mt5.risk.sizing import Side, make_signal, position_size


def _fake_forecast(expected_return: float) -> ForecastResult:
    return ForecastResult(
        last_close=1.1000,
        horizon_close_mean=1.1000 * (1 + expected_return),
        expected_return=expected_return,
        prob_up=0.6 if expected_return > 0 else 0.4,
        predicted_path=pd.DataFrame(),
        horizon_bars=12,
    )


def test_cost_filter_blocks_small_moves():
    # tiny expected move, wide spread -> FLAT
    sig = make_signal(_fake_forecast(0.00005), spread_in_price=0.00010, threshold_bps=8)
    assert sig.side is Side.FLAT


def test_cost_filter_allows_clear_long():
    sig = make_signal(_fake_forecast(0.0030), spread_in_price=0.00010, threshold_bps=8)
    assert sig.side is Side.LONG


def test_cost_filter_allows_clear_short():
    sig = make_signal(_fake_forecast(-0.0030), spread_in_price=0.00010, threshold_bps=8)
    assert sig.side is Side.SHORT


def test_position_size_scales_with_risk():
    small = position_size(10_000, 0.005, stop_distance_in_price=0.0020, pip_value_per_unit=1.0)
    big = position_size(10_000, 0.010, stop_distance_in_price=0.0020, pip_value_per_unit=1.0)
    assert big > small > 0


def test_position_size_zero_on_bad_stop():
    assert position_size(10_000, 0.005, 0.0, 1.0) == 0.0


@pytest.mark.slow
def test_brain_forecasts_from_csv():
    """Stage 1 deliverable: a real Kronos forecast from a CSV slice.

    Downloads Kronos-small + the tokenizer from Hugging Face on first run and
    runs on CPU, so it is slow (tens of seconds). Marked `slow`; deselect with
    `-m "not slow"` for fast unit runs.
    """
    from kronos_mt5.brain.kronos_predictor import KronosBrain

    brain = KronosBrain("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small", device="cpu")
    df = pd.read_csv("data/EURUSD_15M.csv").tail(400).reset_index(drop=True)
    x_ts = pd.to_datetime(df["timestamps"])
    y_ts = pd.Series(pd.date_range(x_ts.iloc[-1], periods=13, freq="15min")[1:])
    result = brain.forecast(df=df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=12, sample_count=4)

    assert isinstance(result, ForecastResult)
    assert result.last_close == pytest.approx(float(df["close"].iloc[-1]))
    assert result.horizon_bars == 12
    assert len(result.predicted_path) == 12
    assert list(result.predicted_path.columns) == ["open", "high", "low", "close"]
    assert 0.0 <= result.prob_up <= 1.0
    assert result.horizon_close_mean > 0.0
