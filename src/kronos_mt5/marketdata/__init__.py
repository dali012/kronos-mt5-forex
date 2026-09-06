"""Reproducible historical market data for offline research.

Downloads, validates and stores Binance USD-M perpetual futures klines and
funding rates. Nothing here touches the live bot: it needs no credentials, writes
only under a gitignored research directory, and never reads production state.

See `docs/historical_data.md`.
"""

from kronos_mt5.marketdata.spec import (
    DEFAULT_START,
    PRODUCTION_INTERVAL,
    PRODUCTION_UNIVERSE,
    required_intervals,
)

__all__ = [
    "DEFAULT_START",
    "PRODUCTION_INTERVAL",
    "PRODUCTION_UNIVERSE",
    "required_intervals",
]
