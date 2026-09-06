"""Small deterministic OHLCV/funding fixtures, generated without network access."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BaselineConfig
from .engine import DAY_NS, run_engine


def fixture(days: int = 300) -> tuple[dict, dict, dict, BaselineConfig]:
    start = pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000
    t = np.arange(days)
    close = 100 * np.exp(0.001 * t + 0.08 * np.sin(t / 13))
    open_ = np.concatenate(([100.0], close[:-1])) * (1 + 0.001 * np.sin(t))
    frame = pd.DataFrame(
        {
            "open_time": start + t * 86_400_000,
            "open": open_,
            "high": np.maximum(open_, close) * 1.012,
            "low": np.minimum(open_, close) * 0.988,
            "close": close,
            "volume": np.full(days, 10000.0),
            "close_time": start + (t + 1) * 86_400_000 - 1,
            "trades": np.full(days, 100),
        }
    )
    funding = pd.DataFrame(
        {
            "funding_time": start + np.arange(days * 3) * 28_800_000,
            "funding_rate": np.full(days * 3, 0.0001),
            "mark_price": None,
        }
    )
    filters = {
        "BTCUSDT": {
            "tick_size": "0.01",
            "step_size": "0.001",
            "min_qty": "0.001",
            "max_qty": "100000",
            "min_notional": "5",
        }
    }
    return {"BTCUSDT": frame}, {"BTCUSDT": funding}, filters, BaselineConfig(symbols=("BTCUSDT",))


def smoke() -> dict:
    frames, funding, filters, config = fixture()
    start = int(frames["BTCUSDT"].iloc[0].open_time) * 1_000_000
    return run_engine(frames, funding, filters, config, start + 253 * DAY_NS, start + 300 * DAY_NS)
