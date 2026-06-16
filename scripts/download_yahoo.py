"""Pull long daily OHLC history for an FX pair from Yahoo Finance.

The MT5 export only covers ~2.5y; a robust *daily*-timeframe test needs more
bars. Yahoo gives ~20y of daily FX OHLC (volume is null for FX -> 0, which is
fine: the daily diagnostic uses the volume-free path).

Output columns match the brain / loader:
    timestamps, open, high, low, close, volume

Usage:
    python scripts/download_yahoo.py --symbol EURUSD=X --range 20y \
        --out data/EURUSD_1D.csv
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download daily FX OHLC from Yahoo")
    ap.add_argument("--symbol", default="EURUSD=X", help="Yahoo ticker, e.g. EURUSD=X, GBPUSD=X")
    ap.add_argument("--range", default="20y")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    url = URL.format(sym=args.symbol, rng=args.range)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())

    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(ts, unit="s"),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": [v or 0 for v in q.get("volume", [0] * len(ts))],
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} daily bars to {out}")
    print(df.head(2).to_string())
    print(df.tail(2).to_string())


if __name__ == "__main__":
    main()
