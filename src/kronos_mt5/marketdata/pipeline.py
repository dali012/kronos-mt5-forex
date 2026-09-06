"""Bounded partition downloads, immutable dataset snapshots and offline validation."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import re
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from . import sources
from .klines import drop_incomplete, parse_kline_csv, validate
from .manifest import load_manifest, save_manifest
from .spec import interval_to_timedelta_seconds
from .store import StoreLayout, atomic_write_parquet, read_partition, utcnow_iso

DAY_MS = 86_400_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_output(root: Path) -> Path:
    root = root.resolve()
    repo = Path(__file__).resolve().parents[3]
    if root == repo or repo in root.parents:
        probe = root / ".research-output"
        result = subprocess.run(["git", "check-ignore", "-q", str(probe)], cwd=repo, check=False)
        if result.returncode or any(
            p in {".git", ".agents", ".codex", "logs"} for p in root.relative_to(repo).parts
        ):
            raise ValueError(
                "research output inside the repository must be gitignored (use research/)"
            )
    root.mkdir(parents=True, exist_ok=True)
    return root


def ms(day: date) -> int:
    return int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)


def check_rows(rows: list[dict], interval: str, start: int, end: int) -> dict:
    report = validate(rows, interval).as_dict()
    step = interval_to_timedelta_seconds(interval) * 1000
    expected = (end - start) // step
    times = {r["open_time"] for r in rows}
    missing = expected - sum(start <= t < end and (t - start) % step == 0 for t in times)
    report["missing_intervals"] = max(report["missing_intervals"], missing)
    report["expected_rows"] = expected
    report["outside_range"] = sum(not start <= r["open_time"] < end for r in rows)
    report["ok"] = report["ok"] and bool(rows) and missing == 0 and not report["outside_range"]
    return report


def check_funding(rows: list[dict], start: int, end: int) -> dict:
    times = [r["funding_time"] for r in rows]
    duplicates = len(times) - len(set(times))
    unordered = sum(a > b for a, b in itertools.pairwise(times))
    invalid = sum(
        not math.isfinite(r["funding_rate"])
        or not start <= r["funding_time"] < end
        or (
            r["mark_price"] is not None
            and (not math.isfinite(r["mark_price"]) or r["mark_price"] <= 0)
        )
        for r in rows
    )
    # Interval changes are allowed, but never silently bridge more than eight hours.
    gaps = sum(b - a > 8 * 3600 * 1000 + 60_000 for a, b in zip([start, *times], [*times, end]))
    return {
        "rows": len(rows),
        "duplicates": duplicates,
        "out_of_order": unordered,
        "missing_intervals": gaps,
        "invalid_rows": invalid,
        "timestamp_jitter_tolerance_ms": 60_000,
        "mark_price_missing": sum(r["mark_price"] is None for r in rows),
        "ok": bool(rows) and not (duplicates or unordered or invalid or gaps),
    }


def funding_csv(payload: bytes) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
        rows.append(
            {
                "funding_time": int(r["calc_time"]),
                "funding_rate": float(r["last_funding_rate"]),
                "mark_price": None,
            }
        )
    return rows


def funding_api(symbol: str, start: int, end: int) -> tuple[list[dict], str]:
    endpoint = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows, cursor = [], start
    while cursor < end:
        query = urlencode(
            {"symbol": symbol, "startTime": cursor, "endTime": end - 1, "limit": 1000}
        )
        time.sleep(1.0)  # below the funding endpoint's shared 500 requests/5min cap
        batch = json.loads(sources.http_get(f"{endpoint}?{query}"))
        if not batch:
            break
        if not isinstance(batch, list):
            raise sources.DownloadError("invalid funding API response")
        times = [int(r["fundingTime"]) for r in batch]
        if (
            times[0] < cursor
            or times[-1] >= end
            or any(a >= b for a, b in itertools.pairwise(times))
        ):
            raise sources.DownloadError(
                "funding pagination made no progress or returned invalid timestamps"
            )
        rows.extend(
            {
                "funding_time": int(r["fundingTime"]),
                "funding_rate": float(r["fundingRate"]),
                "mark_price": float(r["markPrice"]) if r.get("markPrice") else None,
            }
            for r in batch
        )
        cursor = times[-1] + 1
    return rows, f"{endpoint}?" + urlencode(
        {"symbol": symbol, "startTime": start, "endTime": end - 1}
    )


def checked_file(root: Path, entry: dict) -> Path:
    path = (root / entry["file"]).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("manifest file escapes dataset root")
    if digest(path) != entry["file_sha256"]:
        raise ValueError(f"stored checksum mismatch: {entry['file']}")
    return path


def validate_entry(root: Path, entry: dict) -> dict:
    rows = read_partition(checked_file(root, entry)).to_dict("records")
    # Arrow's nullable float is NaN when materialized through pandas.
    if entry["kind"] == "fundingRate":
        for r in rows:
            if pd.isna(r["mark_price"]):
                r["mark_price"] = None
        result = check_funding(rows, entry["start_ms"], entry["end_ms"])
    else:
        result = check_rows(rows, entry["interval"], entry["start_ms"], entry["end_ms"])
        if any(r["close_time"] >= int(datetime.now(timezone.utc).timestamp() * 1000) for r in rows):
            result["ok"] = False
            result["incomplete"] = True
    if not result["ok"] or len(rows) != entry["rows"] or result != entry["validation"]:
        raise ValueError(f"invalid partition or metadata: {entry['file']}: {result}")
    return result


def validate_dataset(path: Path) -> dict:
    snapshot = json.loads(path.read_text())
    if snapshot.get("schema_version") != 1 or not snapshot.get("partitions"):
        raise ValueError("unsupported or empty dataset manifest")
    root = path.parent.parent
    groups: dict[tuple, list] = {}
    results = []
    for entry in snapshot["partitions"]:
        results.append(validate_entry(root, entry))
        groups.setdefault((entry["symbol"], entry["interval"], entry["kind"]), []).append(entry)
    expected = {(s, i, "klines") for s in snapshot["symbols"] for i in snapshot["intervals"]}
    if snapshot["funding"]:
        expected |= {(s, "funding", "fundingRate") for s in snapshot["symbols"]}
    if set(groups) != expected:
        raise ValueError("manifest universe/interval coverage mismatch")
    start, end = snapshot["start_ms"], snapshot["end_ms"]
    for key, entries in groups.items():
        cursor = start
        for entry in sorted(entries, key=lambda e: e["start_ms"]):
            lo, hi = max(start, entry["start_ms"]), min(end, entry["end_ms"])
            if lo != cursor or hi <= lo:
                raise ValueError(f"cross-partition gap or overlap: {key}")
            cursor = hi
        if cursor != end:
            raise ValueError(f"missing boundary coverage: {key}")
    return {
        "ok": True,
        "partitions": len(results),
        "rows": sum(r["rows"] for r in results),
        "gaps": sum(r["missing_intervals"] for r in results),
        "duplicates": sum(r["duplicates"] for r in results),
        "manifest_sha256": digest(path),
    }


def download(
    root: Path,
    symbols: list[str],
    intervals: list[str],
    start: date,
    end: date,
    *,
    funding: bool = True,
) -> Path:
    """Inclusive UTC dates; completed days only. Memory is bounded to one month."""
    if start > end or end > sources.latest_complete_utc_day():
        raise ValueError("range must be ordered and end on a fully completed UTC day")
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not re.fullmatch(r"[A-Z0-9]+USDT", s) for s in symbols)
    ):
        raise ValueError("expected unique uppercase USD-M symbols, e.g. BTCUSDT")
    if not intervals or len(set(intervals)) != len(intervals):
        raise ValueError("expected unique intervals")
    for interval in intervals:
        interval_to_timedelta_seconds(interval)
    root = safe_output(root)
    layout = StoreLayout(root)
    entries = []
    for symbol in symbols:
        for interval in [*intervals, *(["funding"] if funding else [])]:
            kind = "fundingRate" if interval == "funding" else "klines"
            manifest_path = layout.manifest_path(symbol, interval)
            manifest = load_manifest(manifest_path, symbol, interval)
            for year, month in sources.month_range(start, end):
                left, right = sources.month_bounds(year, month)
                lo, hi = (
                    max(ms(start), int(left.timestamp() * 1000)),
                    min(ms(end) + DAY_MS, int(right.timestamp() * 1000)),
                )
                ref = sources.ArchiveRef(symbol, interval, year, month, kind=kind)
                key = ref.partition
                cached = manifest["partitions"].get(key)
                if cached and cached.get("start_ms", hi) <= lo and cached.get("end_ms", lo) >= hi:
                    try:
                        validate_entry(root, cached)
                        entries.append(cached)
                        continue
                    except (OSError, ValueError, KeyError):
                        pass
                try:
                    archive = sources.fetch_archive(ref)
                    rows = (
                        funding_csv(archive["csv"])
                        if kind == "fundingRate"
                        else parse_kline_csv(archive["csv"])
                    )
                    sources_used = [{k: v for k, v in archive.items() if k != "csv"}]
                except sources.NotFound:
                    rows, sources_used = [], []
                    if kind == "fundingRate":
                        rows, url = funding_api(symbol, lo, hi)
                        sources_used.append({"url": url, "checksum_verified": False})
                    else:
                        for day_ms in range(lo, hi, DAY_MS):
                            day = datetime.fromtimestamp(day_ms / 1000, timezone.utc)
                            daily = sources.ArchiveRef(symbol, interval, year, month, day.day)
                            # Persist each fallback day independently, so an interrupted tail resumes.
                            daily_key = f"{key}-{day.day:02d}"
                            cached_day = manifest["partitions"].get(daily_key)
                            if cached_day:
                                try:
                                    validate_entry(root, cached_day)
                                    rows.extend(
                                        read_partition(root / cached_day["file"]).to_dict("records")
                                    )
                                    sources_used.extend(cached_day["sources"])
                                    continue
                                except (OSError, ValueError, KeyError):
                                    pass
                            try:
                                archive = sources.fetch_archive(daily)
                                day_rows = parse_kline_csv(archive["csv"])
                                day_sources = [{k: v for k, v in archive.items() if k != "csv"}]
                            except sources.NotFound:
                                raw = sources.fetch_api_klines(
                                    symbol, interval, day_ms, day_ms + DAY_MS
                                )
                                day_rows = parse_kline_csv(sources.api_rows_to_csv(raw))
                                day_sources = [
                                    {
                                        "url": sources.FAPI
                                        + "?"
                                        + urlencode(
                                            {
                                                "symbol": symbol,
                                                "interval": interval,
                                                "startTime": day_ms,
                                                "endTime": day_ms + DAY_MS - 1,
                                            }
                                        ),
                                        "checksum_verified": False,
                                    }
                                ]
                            day_entry = store_entry(
                                root,
                                layout,
                                symbol,
                                interval,
                                kind,
                                daily_key,
                                day_rows,
                                day_ms,
                                day_ms + DAY_MS,
                                day_sources,
                            )
                            manifest["partitions"][daily_key] = day_entry
                            save_manifest(manifest_path, manifest)
                            rows.extend(day_rows)
                            sources_used.extend(day_sources)
                time_key = "funding_time" if kind == "fundingRate" else "open_time"
                rows = [r for r in rows if lo <= r[time_key] < hi]
                if kind == "klines":
                    rows, _ = drop_incomplete(rows, interval)
                entry = store_entry(
                    root, layout, symbol, interval, kind, key, rows, lo, hi, sources_used
                )
                manifest["partitions"][key] = entry
                save_manifest(manifest_path, manifest)
                entries.append(entry)
    snapshot = {
        "schema_version": 1,
        "venue": "binance-um",
        "symbols": symbols,
        "intervals": intervals,
        "start_ms": ms(start),
        "end_ms": ms(end) + DAY_MS,
        "funding": funding,
        "downloaded_at": utcnow_iso(),
        "partitions": entries,
    }
    identifier = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:16]
    path = layout.manifests / f"dataset-{identifier}.json"
    save_manifest(path, snapshot)
    validate_dataset(path)
    return path


def store_entry(
    root: Path,
    layout: StoreLayout,
    symbol: str,
    interval: str,
    kind: str,
    key: str,
    rows: list[dict],
    lo: int,
    hi: int,
    sources_used: list[dict],
) -> dict:
    result = (
        check_funding(rows, lo, hi) if kind == "fundingRate" else check_rows(rows, interval, lo, hi)
    )
    if not result["ok"]:
        failure = {
            "symbol": symbol,
            "interval": interval,
            "partition": key,
            "sources": sources_used,
            "start_ms": lo,
            "end_ms": hi,
            "validation": result,
        }
        save_manifest(layout.manifests / f"failure-{symbol}-{interval}-{key}.json", failure)
        raise ValueError(f"partition validation failed: {symbol}/{interval}/{key}: {result}")
    # Content-addressed files keep older dataset snapshots reproducible after range expansion.
    content = hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:20]
    file_key = f"{key}-{content}"
    path = (
        layout.funding_partition(symbol, file_key)
        if kind == "fundingRate"
        else layout.kline_partition(symbol, interval, file_key)
    )
    atomic_write_parquet(pd.DataFrame(rows), path)
    return {
        "symbol": symbol,
        "interval": interval,
        "kind": kind,
        "partition": key,
        "start_ms": lo,
        "end_ms": hi,
        "rows": len(rows),
        "sources": sources_used,
        "file": str(path.relative_to(root)),
        "file_sha256": digest(path),
        "downloaded_at": utcnow_iso(),
        "validation": result,
        "ok": True,
    }


def load_series(path: Path, symbol: str, interval: str) -> pd.DataFrame:
    """Read only the files explicitly pinned by a validated dataset snapshot."""
    snapshot = json.loads(path.read_text())
    entries = [
        e for e in snapshot["partitions"] if e["symbol"] == symbol and e["interval"] == interval
    ]
    frames = [read_partition(checked_file(path.parent.parent, e)) for e in entries]
    if not frames:
        raise ValueError(f"missing {symbol}/{interval}")
    frame = pd.concat(frames, ignore_index=True)
    key = "funding_time" if interval == "funding" else "open_time"
    frame = frame[(frame[key] >= snapshot["start_ms"]) & (frame[key] < snapshot["end_ms"])]
    if frame[key].duplicated().any():
        raise ValueError("overlapping manifest partitions")
    return frame.sort_values(key).reset_index(drop=True)
