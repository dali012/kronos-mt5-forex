"""What data the *deployed* strategy actually needs.

Every value here is derived from production code rather than guessed, and
`tests/test_marketdata.py` fails if the production side changes without this
being updated.
"""

from __future__ import annotations

import re

# The live runner builds its bar type as f"{instrument_id}-1-DAY-LAST-EXTERNAL"
# in `kronos_mt5.live.run_trend_binance`, so the strategy consumes ONE-DAY bars.
PRODUCTION_BAR_SPEC = "1-DAY-LAST-EXTERNAL"
PRODUCTION_INTERVAL = "1d"

# Nautilus bar-spec step/aggregation -> Binance archive interval token.
_BAR_SPEC_TO_INTERVAL = {
    ("1", "MINUTE"): "1m",
    ("5", "MINUTE"): "5m",
    ("15", "MINUTE"): "15m",
    ("30", "MINUTE"): "30m",
    ("1", "HOUR"): "1h",
    ("4", "HOUR"): "4h",
    ("1", "DAY"): "1d",
}

# The universe the audit observed trading (config BINANCE_SYMBOLS).
PRODUCTION_UNIVERSE = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "LTCUSDT",
    "LINKUSDT",
)

DEFAULT_START = "2023-01-01"

# Chronological split. The holdout is never used for parameter selection.
DEFAULT_DEV_END = "2025-12-31"
DEFAULT_HOLDOUT_START = "2026-01-01"


def interval_from_bar_spec(bar_spec: str) -> str:
    """Map a Nautilus bar spec such as `1-DAY-LAST-EXTERNAL` to `1d`."""
    match = re.match(r"^(\d+)-([A-Z]+)-", bar_spec.upper())
    if not match:
        raise ValueError(f"unrecognised bar spec: {bar_spec!r}")
    key = (match.group(1), match.group(2))
    if key not in _BAR_SPEC_TO_INTERVAL:
        raise ValueError(f"no Binance interval for bar spec {bar_spec!r}")
    return _BAR_SPEC_TO_INTERVAL[key]


def required_intervals(bar_specs: tuple[str, ...] = (PRODUCTION_BAR_SPEC,)) -> tuple[str, ...]:
    """Intervals the deployed strategy needs, derived from its bar specs."""
    return tuple(dict.fromkeys(interval_from_bar_spec(spec) for spec in bar_specs))


def interval_to_timedelta_seconds(interval: str) -> int:
    supported = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
    if interval not in supported:
        raise ValueError(f"unsupported fixed UTC interval: {interval!r}")
    return int(interval[:-1]) * {"m": 60, "h": 3600, "d": 86400}[interval[-1]]


def strategy_warmup_bars(lookbacks: tuple[int, ...], vol_window: int) -> int:
    """Bars the strategy must see before it may trade.

    `TrendStrategy._rebalance` returns early until `len(closes) > max(lookbacks)`,
    and volatility needs `vol_window` returns, so the binding constraint is the
    larger of the two, plus one bar to form the first return.
    """
    return max(max(lookbacks) + 1, vol_window + 1)
