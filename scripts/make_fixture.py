"""Generate a synthetic but realistic EUR/USD 15-minute OHLCV CSV fixture.

This is a *test fixture* for offline development (Stage 1 smoke test, early
backtests) so the pipeline can run without a Windows MT5 terminal. It is NOT
real market data — replace it with `scripts/download_data.py` output (real MT5
bars) before drawing any conclusion about edge.

Output columns match what the brain and backtest wrangler expect:
    timestamps, open, high, low, close, volume   (volume = tick-volume proxy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "EURUSD_15M.csv"
N_BARS = 1200
START = pd.Timestamp("2024-01-01 00:00", tz=None)
SEED = 7


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Build a 15-minute index over calendar time, then drop the FX weekend
    # (Sat all day + Sunday before 22:00 UTC) so weekday/hour time features look
    # like a real FX session calendar.
    raw = pd.date_range(START, periods=N_BARS * 2, freq="15min")
    is_weekend = (raw.weekday == 5) | ((raw.weekday == 6) & (raw.hour < 22))
    ts = raw[~is_weekend][:N_BARS]

    # Close path: geometric random walk with mild mean reversion and a per-bar
    # vol consistent with ~0.5%/day EURUSD over ~96 bars/day.
    sigma = 0.0005
    n = len(ts)
    shocks = rng.normal(0.0, sigma, size=n)
    log_close = np.cumsum(shocks) + np.log(1.10)
    # gentle mean reversion toward the anchor so it doesn't drift unrealistically
    log_close = 0.999 * log_close + 0.001 * np.log(1.10)
    close = np.exp(log_close)

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]  # open = previous close (continuous session)

    # Intrabar range proportional to a noisy fraction of price.
    span = np.abs(rng.normal(0.0, sigma, size=n)) + 0.5 * sigma
    high = np.maximum(open_, close) + span * close
    low = np.minimum(open_, close) - span * close

    # Tick-volume proxy: larger when the bar moved more.
    move = np.abs(close - open_) / close
    volume = (200 + move * 1e5 + rng.integers(0, 150, size=n)).astype(int)

    df = pd.DataFrame(
        {
            "timestamps": ts,
            "open": np.round(open_, 5),
            "high": np.round(high, 5),
            "low": np.round(low, 5),
            "close": np.round(close, 5),
            "volume": volume,
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} bars to {OUT}")
    print(df.head())
    print("...")
    print(df.tail())


if __name__ == "__main__":
    main()
