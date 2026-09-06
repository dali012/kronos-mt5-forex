"""Noninteractive historical research commands; no trading credentials accepted."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .manifest import save_manifest
from .pipeline import download, safe_output, validate_dataset
from .sources import DownloadError, http_get, latest_complete_utc_day
from .spec import DEFAULT_START, PRODUCTION_UNIVERSE, required_intervals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="download/update checksummed UTC historical data")
    fetch.add_argument("--symbols", nargs="+", default=list(PRODUCTION_UNIVERSE))
    fetch.add_argument("--intervals", nargs="+", default=list(required_intervals()))
    fetch.add_argument(
        "--start", type=date.fromisoformat, default=date.fromisoformat(DEFAULT_START)
    )
    fetch.add_argument("--end", type=date.fromisoformat, default=latest_complete_utc_day())
    fetch.add_argument("--destination", type=Path, default=Path("research/data"))
    fetch.add_argument(
        "--no-funding",
        action="store_true",
        help="candles only; baseline will require explicit funding omission",
    )
    verify = commands.add_parser(
        "validate", help="validate files and cross-partition coverage without network"
    )
    verify.add_argument("--manifest", type=Path, required=True)
    filters = commands.add_parser(
        "exchange-info", help="snapshot current public perpetual exchange filters"
    )
    filters.add_argument("--symbols", nargs="+", default=list(PRODUCTION_UNIVERSE))
    filters.add_argument("--output", type=Path, default=Path("research/exchange-info.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            path = download(
                args.destination,
                args.symbols,
                args.intervals,
                args.start,
                args.end,
                funding=not args.no_funding,
            )
            print(json.dumps({"manifest": str(path), **validate_dataset(path)}, sort_keys=True))
        elif args.command == "validate":
            print(json.dumps(validate_dataset(args.manifest), sort_keys=True))
        else:
            import hashlib

            from kronos_mt5.baseline.instruments import exchange_filters

            from .store import utcnow_iso

            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            raw = http_get(url)
            symbols = exchange_filters(json.loads(raw), args.symbols)
            safe_output(args.output.parent)
            save_manifest(
                args.output,
                {
                    "schema_version": 1,
                    "source": url,
                    "downloaded_at": utcnow_iso(),
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "symbols": symbols,
                    "historical_filters": False,
                    "leverage_tiers": "unavailable without credentials; replay caps gross leverage at 2",
                },
            )
            print(json.dumps({"ok": True, "exchange_info": str(args.output)}))
    except (ValueError, OSError, KeyError, TypeError, DownloadError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
