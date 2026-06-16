"""Signal generation + position sizing.

Pure functions: no NautilusTrader, no broker, no GPU. This keeps the part that
decides *whether and how much to trade* fully unit-testable and is where most of
the real edge (or lack of it) lives.

Design principles:
  - The cost-aware filter is non-negotiable. A forecast that does not clear
    spread + threshold is FLAT. Win rate is a trap; expectancy after costs is
    what matters.
  - Sizing is risk-first: risk a fixed fraction of equity per trade, derive units
    from the stop distance. Never size from a fixed lot.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kronos_mt5.brain.kronos_predictor import ForecastResult


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Signal:
    side: Side
    expected_return: float
    prob_up: float
    reason: str


def make_signal(
    forecast: ForecastResult,
    spread_in_price: float,
    threshold_bps: float,
) -> Signal:
    """Cost-aware signal.

    Only act if the expected move exceeds (spread + threshold). `threshold_bps`
    is the extra edge required ON TOP of the round-trip cost, in basis points.
    """
    cost_return = spread_in_price / forecast.last_close          # round-trip-ish proxy; refine
    threshold_return = threshold_bps / 1e4
    required = cost_return + threshold_return

    er = forecast.expected_return
    if er > required:
        return Signal(Side.LONG, er, forecast.prob_up,
                      f"er={er:.5f} > required={required:.5f}")
    if er < -required:
        return Signal(Side.SHORT, er, forecast.prob_up,
                      f"er={er:.5f} < -required={required:.5f}")
    return Signal(Side.FLAT, er, forecast.prob_up,
                  f"|er|={abs(er):.5f} <= required={required:.5f} (filtered by cost)")


def position_size(
    equity: float,
    risk_per_trade: float,
    stop_distance_in_price: float,
    pip_value_per_unit: float,
) -> float:
    """Fixed-fractional sizing.

    units = (equity * risk_per_trade) / (stop_distance * value_per_unit_per_price)

    TODO(claude-code): replace pip_value_per_unit with the instrument's real
    tick/point value from the MT5 InstrumentProvider (contract size, point,
    quote-currency conversion). Round to the instrument's volume step and clamp
    to volume_min/volume_max.
    """
    if stop_distance_in_price <= 0:
        return 0.0
    risk_amount = equity * risk_per_trade
    raw_units = risk_amount / (stop_distance_in_price * pip_value_per_unit)
    return max(raw_units, 0.0)


def atr_stop_distance(atr: float, mult: float) -> float:
    """Stop distance in price = ATR * multiplier."""
    return atr * mult
