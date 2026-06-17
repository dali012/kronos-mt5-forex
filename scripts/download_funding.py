"""Download Binance USDT-M perpetual funding history for backtests.

Writes data/funding/SYMBOL_funding.csv with Binance's fundingRate records.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "funding"
BASE = "https://fapi.binance.com/fapi/v1/fundingRate"


def _ms(dt: str) -> int:
    return int(datetime.fromisoformat(dt).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_symbol(symbol: str, start: str, end: str, limit: int = 1000) -> list[dict]:
    rows: list[dict] = []
    start_ms = _ms(start)
    end_ms = _ms(end)
    while start_ms <= end_ms:
        params = urllib.parse.urlencode(
            {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit}
        )
        with urllib.request.urlopen(f"{BASE}?{params}", timeout=30) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        rows.extend(batch)
        next_ms = int(batch[-1]["fundingTime"]) + 1
        if next_ms <= start_ms:
            break
        start_ms = next_ms
        time.sleep(0.15)
    return rows


def write_csv(symbol: str, rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_funding.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "fundingTime", "fundingRate", "markPrice"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "symbol": row.get("symbol", symbol),
                    "fundingTime": row["fundingTime"],
                    "fundingRate": row["fundingRate"],
                    "markPrice": row.get("markPrice", ""),
                }
            )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance USDT-M funding rates")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    for symbol in args.symbols:
        rows = fetch_symbol(symbol.upper(), args.start, args.end)
        path = write_csv(symbol.upper(), rows, args.out_dir)
        print(f"{symbol.upper()}: {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
