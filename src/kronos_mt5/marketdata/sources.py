"""Binance data sources: bulk archives first, REST API only for the remainder.

No credentials are used or accepted. The bulk archives at data.binance.vision are
public, checksummed and cheap to serve, so they are strongly preferred; the
public REST endpoint fills only the recent tail the archives have not published.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

VISION = "https://data.binance.vision/data/futures/um"
FAPI = "https://fapi.binance.com/fapi/v1/klines"
USER_AGENT = "kronos-mt5-research/1.0 (+offline backtest data pipeline)"

# Bounded exponential backoff. Single-threaded API requests are paced one second
# apart; HTTP 429 Retry-After is honored, with long waits returned as resumable errors.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECS = 1.0
BACKOFF_MAX_SECS = 30.0
API_PAUSE_SECS = 1.0
API_PAGE_LIMIT = 1500

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """A download failed permanently (after retries) or failed verification."""


class ChecksumMismatch(DownloadError):
    """The archive did not match Binance's published SHA-256."""


class NotFound(DownloadError):
    """Binance has not published this partition (normal for the recent tail)."""


@dataclass(frozen=True)
class ArchiveRef:
    """One monthly bulk partition."""

    symbol: str
    interval: str
    year: int
    month: int
    day: int | None = None
    kind: str = "klines"

    @property
    def name(self) -> str:
        token = self.interval if self.kind == "klines" else "fundingRate"
        suffix = f"-{self.day:02d}" if self.day is not None else ""
        return f"{self.symbol}-{token}-{self.year:04d}-{self.month:02d}{suffix}.zip"

    @property
    def url(self) -> str:
        period = "daily" if self.day is not None else "monthly"
        interval = f"/{self.interval}" if self.kind == "klines" else ""
        return f"{VISION}/{period}/{self.kind}/{self.symbol}{interval}/{self.name}"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"

    @property
    def partition(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) covering [start, end]."""
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
    return start, end


def http_get(url: str, *, timeout: float = 60.0, opener=urllib.request.urlopen) -> bytes:
    """GET with bounded exponential backoff on transient failures.

    A 404 raises `NotFound` immediately — for the bulk archives that simply means
    the partition is not published yet, which the caller handles by falling back
    to the API rather than by retrying.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        retry_after = 0.0
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with opener(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotFound(f"not published: {url}") from exc
            last = exc
            try:
                retry_after = float(exc.headers.get("Retry-After", "0"))
            except (ValueError, AttributeError):
                retry_after = 0.0
            if retry_after > BACKOFF_MAX_SECS:
                raise DownloadError(
                    f"rate limited: retry after {retry_after}s; resume later"
                ) from exc
            if exc.code not in RETRYABLE_STATUS:
                raise DownloadError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < MAX_ATTEMPTS:
            time.sleep(
                max(retry_after, min(BACKOFF_BASE_SECS * 2 ** (attempt - 1), BACKOFF_MAX_SECS))
            )
    raise DownloadError(f"giving up on {url} after {MAX_ATTEMPTS} attempts: {last!r}")


def parse_checksum(payload: bytes) -> str:
    """Binance CHECKSUM files hold `<sha256>  <filename>`."""
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        raise ChecksumMismatch("empty checksum file")
    digest = text.split()[0].strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ChecksumMismatch(f"malformed checksum payload: {text[:80]!r}")
    return digest


def extract_single_csv(payload: bytes) -> bytes:
    """Read the single CSV member out of a Binance zip."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(members) != 1:
                raise DownloadError(
                    f"expected exactly one CSV in the archive, found {len(members)}"
                )
            if archive.getinfo(members[0]).file_size > 256 * 1024 * 1024:
                raise DownloadError("archive exceeds 256 MiB uncompressed limit")
            return archive.read(members[0])
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"corrupt archive: {exc}") from exc


def fetch_archive(ref: ArchiveRef, *, verify_checksum: bool = True, opener=None) -> dict:
    """Download one monthly archive, verify it, and return its CSV bytes."""
    kwargs = {"opener": opener} if opener is not None else {}
    payload = http_get(ref.url, **kwargs)
    actual = hashlib.sha256(payload).hexdigest()
    expected = None
    if verify_checksum:
        try:
            expected = parse_checksum(http_get(ref.checksum_url, **kwargs))
        except NotFound:
            expected = None  # some partitions ship without a checksum
        if expected is not None and expected != actual:
            raise ChecksumMismatch(f"{ref.name}: expected sha256 {expected}, downloaded {actual}")
    return {
        "csv": extract_single_csv(payload),
        "sha256": actual,
        "checksum_verified": bool(expected),
        "url": ref.url,
        "bytes": len(payload),
    }


def fetch_api_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    opener=None,
    pause_secs: float = API_PAUSE_SECS,
) -> list[list]:
    """Page the public REST endpoint for `[start_ms, end_ms)`.

    Used only for the recent range the bulk archives have not published yet.
    Pagination advances strictly past the last open time so a stuck cursor cannot
    loop forever.
    """
    kwargs = {"opener": opener} if opener is not None else {}
    out: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": API_PAGE_LIMIT,
            }
        )
        if pause_secs:
            time.sleep(pause_secs)
        payload = http_get(f"{FAPI}?{query}", **kwargs)
        batch = json.loads(payload)
        if not batch:
            break
        if not isinstance(batch, list) or any(
            not isinstance(r, list) or len(r) != 12 for r in batch
        ):
            raise DownloadError("invalid kline API response")
        times = [int(r[0]) for r in batch]
        if (
            times[0] < cursor
            or times[-1] >= end_ms
            or any(a >= b for a, b in itertools.pairwise(times))
        ):
            raise DownloadError("API returned duplicate, unordered, or out-of-range candles")
        out.extend(batch)
        last_open = int(batch[-1][0])
        if last_open + 1 <= cursor:
            raise DownloadError("API pagination made no progress")
        cursor = last_open + 1
    return out


def api_rows_to_csv(rows: list[list]) -> bytes:
    """Render REST rows in the same 12-column shape as the bulk archives."""
    lines = []
    for row in rows:
        cells = list(row[:12]) + [""] * max(0, 12 - len(row))
        lines.append(",".join(str(c) for c in cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def latest_complete_utc_day(now: datetime | None = None) -> date:
    """Yesterday in UTC: the most recent day whose candles are all closed."""
    now = now or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must have an explicit timezone")
    return (now.astimezone(timezone.utc) - timedelta(days=1)).date()
