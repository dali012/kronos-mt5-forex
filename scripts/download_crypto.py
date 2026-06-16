"""Pull BTC/USDT (or any Binance pair) OHLCV bars into data/<SYM>_<TF>.csv.

Crypto is Kronos' demonstrated domain (the upstream repo demos BTC/USDT) and,
unlike FX, has REAL volume — which the OHLCV model was trained on. This lets us
test whether Kronos has any edge on a market it suits, vs efficient FX majors.

Output columns match the brain / backtest loader:
    timestamps, open, high, low, close, volume   (volume = base-asset volume)

Usage:
    python scripts/download_crypto.py --symbol BTCUSDT --interval 15m \
        --start 2024-01-01 --out data/BTCUSDT_15M.csv
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.binance.com/api/v3/klines"


def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        q = urllib.parse.urlencode(
            {"symbol": symbol, "interval": interval, "startTime": cur,
             "endTime": end_ms, "limit": 1000}
        )
        with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1000:
            break
        print(f"  fetched {len(rows)} bars (through {pd.Timestamp(batch[-1][0], unit='ms')})",
              flush=True)
        time.sleep(0.25)  # be polite to the API
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Binance OHLCV bars")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="default: now")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(
        (pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")).timestamp()
        * 1000
    )

    print(f"Fetching {args.symbol} {args.interval} from {args.start}...")
    raw = fetch(args.symbol, args.interval, start_ms, end_ms)
    df = pd.DataFrame(raw).iloc[:, :6]
    df.columns = ["timestamps", "open", "high", "low", "close", "volume"]
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "data" / (
        f"{args.symbol}_{args.interval.upper()}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} bars to {out}")
    print(df.head(2).to_string())
    print(df.tail(2).to_string())


if __name__ == "__main__":
    main()
