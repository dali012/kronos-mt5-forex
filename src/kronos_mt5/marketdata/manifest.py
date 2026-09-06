"""Versioned JSON manifests; write atomically and never accept corrupt metadata."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def load_manifest(path: Path, symbol: str, interval: str) -> dict:
    if not path.is_file():
        return {
            "manifest_version": "1.0.0",
            "venue": "binance-um",
            "symbol": symbol,
            "interval": interval,
            "partitions": {},
        }
    data = json.loads(path.read_text())
    if data.get("symbol") != symbol or data.get("interval") != interval:
        raise ValueError("manifest identity mismatch")
    if data.get("manifest_version") != "1.0.0" or not isinstance(data.get("partitions"), dict):
        raise ValueError("unsupported series manifest")
    return data


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
