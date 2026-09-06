"""Kline parsing and validation.

Binance publishes USD-M futures klines as 12-column CSV inside monthly/daily
zips. Older archives have no header row, newer ones do; both are handled.

Validation is deliberately strict — a silently corrupt candle is worse than a
missing one, because it produces a backtest that looks fine and is wrong.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from kronos_mt5.marketdata.spec import interval_to_timedelta_seconds

# open_time, open, high, low, close, volume, close_time, quote_volume, count,
# taker_buy_volume, taker_buy_quote_volume, ignore
RAW_COLUMNS = 12
COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time", "trades")
HEADER_TOKENS = {"open_time", "open time", "opentime"}


class KlineFormatError(ValueError):
    """The archive is not parseable as Binance klines."""


@dataclass
class ValidationReport:
    """Everything wrong (or right) with a set of candles."""

    rows: int = 0
    first_open_time: str | None = None
    last_open_time: str | None = None
    duplicates: int = 0
    duplicate_open_times: list[str] = field(default_factory=list)
    out_of_order: int = 0
    missing_intervals: int = 0
    missing_ranges: list[dict] = field(default_factory=list)
    overlaps: int = 0
    invalid_ohlc: int = 0
    negative_volume: int = 0
    invalid_rows: list[dict] = field(default_factory=list)
    incomplete_dropped: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.missing_intervals
            or self.duplicates
            or self.out_of_order
            or self.overlaps
            or self.invalid_ohlc
            or self.negative_volume
        )

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "first_open_time": self.first_open_time,
            "last_open_time": self.last_open_time,
            "duplicates": self.duplicates,
            "duplicate_open_times": self.duplicate_open_times[:20],
            "out_of_order": self.out_of_order,
            "missing_intervals": self.missing_intervals,
            "missing_ranges": self.missing_ranges[:20],
            "overlaps": self.overlaps,
            "invalid_ohlc": self.invalid_ohlc,
            "negative_volume": self.negative_volume,
            "invalid_rows": self.invalid_rows[:20],
            "incomplete_dropped": self.incomplete_dropped,
            "ok": self.ok,
        }


def _is_header(row: list[str]) -> bool:
    return bool(row) and row[0].strip().lower().replace("_", " ") in {
        t.replace("_", " ") for t in HEADER_TOKENS
    }


def parse_kline_csv(payload: bytes | str) -> list[dict]:
    """Parse raw Binance kline CSV into typed rows. Raises on malformed input."""
    text = payload.decode("utf-8", errors="strict") if isinstance(payload, bytes) else payload
    reader = csv.reader(io.StringIO(text))
    rows: list[dict] = []
    for lineno, raw in enumerate(reader, start=1):
        if not raw or all(not cell.strip() for cell in raw):
            continue
        if lineno == 1 and _is_header(raw):
            continue
        if len(raw) != RAW_COLUMNS:
            raise KlineFormatError(
                f"line {lineno}: expected {RAW_COLUMNS} columns, found {len(raw)}"
            )
        try:
            rows.append(
                {
                    "open_time": int(raw[0]),
                    "open": float(raw[1]),
                    "high": float(raw[2]),
                    "low": float(raw[3]),
                    "close": float(raw[4]),
                    "volume": float(raw[5]),
                    "close_time": int(raw[6]),
                    "trades": int(raw[8]) if raw[8].strip() else 0,
                }
            )
        except (TypeError, ValueError) as exc:
            raise KlineFormatError(f"line {lineno}: unparseable value ({exc})") from exc
    if not rows:
        raise KlineFormatError("archive contained no kline rows")
    return rows


def to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def drop_incomplete(rows: list[dict], interval: str, now_ms: int | None = None) -> tuple[list, int]:
    """Remove the still-forming candle.

    A candle is complete only once its close time has passed. `close_time` is the
    last millisecond of the bar, so the bar is finished when `now > close_time`.
    """
    if now_ms is None:
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    step_ms = interval_to_timedelta_seconds(interval) * 1000
    keep, dropped = [], 0
    for row in rows:
        close_time = max(row.get("close_time", 0), row["open_time"] + step_ms - 1)
        if close_time >= now_ms:
            dropped += 1
            continue
        keep.append(row)
    return keep, dropped


def validate(rows: list[dict], interval: str) -> ValidationReport:
    """Full structural and semantic validation of an ordered candle set."""
    report = ValidationReport(rows=len(rows))
    if not rows:
        return report

    step_ms = interval_to_timedelta_seconds(interval) * 1000
    seen: set[int] = set()
    previous: dict | None = None
    duplicates: list[str] = []

    for row in rows:
        open_time = row["open_time"]
        if open_time in seen:
            report.duplicates += 1
            duplicates.append(to_utc(open_time).isoformat())
        seen.add(open_time)

        if previous is not None:
            if open_time < previous["open_time"]:
                report.out_of_order += 1
            elif open_time > previous["open_time"]:
                delta = open_time - previous["open_time"]
                if delta > step_ms:
                    missing = delta // step_ms - 1
                    report.missing_intervals += int(missing)
                    report.missing_ranges.append(
                        {
                            "after": to_utc(previous["open_time"]).isoformat(),
                            "before": to_utc(open_time).isoformat(),
                            "missing_bars": int(missing),
                        }
                    )
            prev_close = previous.get("close_time")
            if (
                prev_close is not None
                and open_time <= prev_close
                and open_time != previous["open_time"]
            ):
                report.overlaps += 1

        high, low = row["high"], row["low"]
        open_, close = row["open"], row["close"]
        if not (
            all(math.isfinite(v) for v in (open_, high, low, close))
            and high >= low
            and high >= open_ >= low
            and high >= close >= low
            and all(v > 0 for v in (open_, high, low, close))
        ):
            report.invalid_ohlc += 1
            report.invalid_rows.append(
                {"open_time": to_utc(open_time).isoformat(), "reason": "ohlc_relationship"}
            )
        if not math.isfinite(row["volume"]) or row["volume"] < 0:
            report.negative_volume += 1
            report.invalid_rows.append(
                {"open_time": to_utc(open_time).isoformat(), "reason": "negative_volume"}
            )
        if open_time % step_ms or row.get("close_time") != open_time + step_ms - 1:
            report.overlaps += 1
            report.invalid_rows.append({"open_time": str(open_time), "reason": "candle_boundary"})
        previous = row

    report.duplicate_open_times = duplicates
    report.first_open_time = to_utc(rows[0]["open_time"]).isoformat()
    report.last_open_time = to_utc(rows[-1]["open_time"]).isoformat()
    return report
