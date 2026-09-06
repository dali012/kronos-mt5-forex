"""Partitioned Parquet store with atomic writes.

Layout (all under a gitignored research root):

    <root>/klines/venue=binance-um/symbol=BTCUSDT/interval=1d/part-2024-03-<content>.parquet
    <root>/funding/venue=binance-um/symbol=BTCUSDT/part-2024-03-<content>.parquet
    <root>/manifests/binance-um__BTCUSDT__1d.json
    <root>/manifests/dataset-<ID>.json

One Parquet file per calendar month keeps memory bounded: a symbol-year is read
as a handful of row groups rather than the whole universe at once.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VENUE = "binance-um"


@dataclass(frozen=True)
class StoreLayout:
    root: Path

    @property
    def klines(self) -> Path:
        return self.root / "klines" / f"venue={VENUE}"

    @property
    def funding(self) -> Path:
        return self.root / "funding" / f"venue={VENUE}"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def kline_dir(self, symbol: str, interval: str) -> Path:
        return self.klines / f"symbol={symbol}" / f"interval={interval}"

    def kline_partition(self, symbol: str, interval: str, partition: str) -> Path:
        return self.kline_dir(symbol, interval) / f"part-{partition}.parquet"

    def funding_dir(self, symbol: str) -> Path:
        return self.funding / f"symbol={symbol}"

    def funding_partition(self, symbol: str, partition: str) -> Path:
        return self.funding_dir(symbol) / f"part-{partition}.parquet"

    def manifest_path(self, symbol: str, interval: str) -> Path:
        return self.manifests / f"{VENUE}__{symbol}__{interval}.json"


def atomic_write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    """Write via a temp file in the same directory, then rename.

    `os.replace` is atomic within a filesystem, so an interrupted run can leave a
    stray `.tmp` but never a truncated file that later looks valid.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        frame.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        os.replace(tmp_path, destination)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_partition(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
