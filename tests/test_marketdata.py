"""Deterministic fixtures and HTTP mocks; no test depends on Binance/network."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from kronos_mt5.marketdata import sources
from kronos_mt5.marketdata.klines import drop_incomplete, parse_kline_csv, validate
from kronos_mt5.marketdata.pipeline import (
    DAY_MS,
    check_funding,
    check_rows,
    download,
    funding_api,
    ms,
    validate_dataset,
)
from kronos_mt5.marketdata.spec import required_intervals, strategy_warmup_bars
from kronos_mt5.marketdata.store import atomic_write_parquet

START = ms(date(2024, 1, 1))


def raw_rows(count=3, start=START):
    return [
        [start + i * DAY_MS, 100, 110, 90, 105, 2, start + (i + 1) * DAY_MS - 1, 0, 4, 0, 0, 0]
        for i in range(count)
    ]


def csv_bytes(count=3, start=START):
    return sources.api_rows_to_csv(raw_rows(count, start))


def zipped(payload):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.csv", payload)
    return buf.getvalue()


def test_spec_matches_real_live_bar_and_strategy():
    from kronos_mt5.strategies.trend_strategy import TrendStrategyConfig

    source = Path("src/kronos_mt5/live/run_trend_binance.py").read_text()
    assert 'bar_type=f"{iid}-1-DAY-LAST-EXTERNAL"' in source
    c = TrendStrategyConfig(
        instrument_id="BTCUSDT.BINANCE", bar_type="BTCUSDT.BINANCE-1-DAY-LAST-EXTERNAL"
    )
    assert required_intervals() == ("1d",)
    assert strategy_warmup_bars(c.lookbacks, c.vol_window) == 253


@pytest.mark.parametrize(
    "header",
    [b"", b"open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbq,ignore\n"],
)
def test_parse_headers_utc_complete(header):
    rows = parse_kline_csv(header + csv_bytes())
    assert validate(rows, "1d").ok
    kept, dropped = drop_incomplete(rows, "1d", START + 2 * DAY_MS)
    assert len(kept) == 2 and dropped == 1
    assert check_rows(kept, "1d", START, START + 2 * DAY_MS)["ok"]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "gap", "unordered", "nan", "inf", "negative_volume", "boundary", "ohlc"],
)
def test_reject_invalid_rows(mutation):
    rows = parse_kline_csv(csv_bytes())
    if mutation == "duplicate":
        rows.append(rows[-1])
    if mutation == "gap":
        rows.pop(1)
    if mutation == "unordered":
        rows.reverse()
    if mutation == "nan":
        rows[0]["volume"] = float("nan")
    if mutation == "inf":
        rows[0]["high"] = float("inf")
    if mutation == "negative_volume":
        rows[0]["volume"] = -1
    if mutation == "boundary":
        rows[0]["close_time"] += 1
    if mutation == "ohlc":
        rows[0]["low"] = 200
    assert not validate(rows, "1d").ok


def test_boundaries_missing_first_last_and_microseconds_rejected():
    rows = parse_kline_csv(csv_bytes())
    assert not check_rows(rows[1:], "1d", START, START + 3 * DAY_MS)["ok"]
    assert not check_rows(rows[:-1], "1d", START, START + 3 * DAY_MS)["ok"]
    rows[0]["open_time"] *= 1000
    rows[0]["close_time"] *= 1000
    with pytest.raises(ValueError):
        validate(rows, "1d")


def test_incomplete_cannot_lie_with_early_close_time():
    rows = parse_kline_csv(csv_bytes(1))
    rows[0]["close_time"] = START
    assert drop_incomplete(rows, "1d", START + 1234) == ([], 1)


def test_latest_day_timezone():
    assert sources.latest_complete_utc_day(
        datetime.fromisoformat("2024-02-01T00:30:00+02:00")
    ) == date(2024, 1, 30)
    assert sources.month_range(date(2023, 12, 31), date(2024, 2, 1)) == [
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]
    assert sources.month_bounds(2024, 2)[1] == datetime(2024, 3, 1, tzinfo=timezone.utc)


def test_archive_checksum_and_corruption():
    payload = zipped(csv_bytes())
    expected = hashlib.sha256(payload).hexdigest().encode() + b"  data.zip"

    def opener(req, timeout):
        return io.BytesIO(expected if req.full_url.endswith("CHECKSUM") else payload)

    ref = sources.ArchiveRef("BTCUSDT", "1d", 2024, 1)
    assert sources.fetch_archive(ref, opener=opener)["checksum_verified"]
    with pytest.raises(sources.ChecksumMismatch):
        sources.fetch_archive(
            ref,
            opener=lambda req, timeout: io.BytesIO(
                b"0" * 64 if req.full_url.endswith("CHECKSUM") else payload
            ),
        )
    with pytest.raises(sources.DownloadError, match="corrupt"):
        sources.extract_single_csv(b"invalid ZIP")
    with pytest.raises(sources.DownloadError):
        sources.extract_single_csv(zipped(b"x")[:-5])


def test_api_pagination_short_pages_and_progress(monkeypatch):
    monkeypatch.setattr(sources, "API_PAGE_LIMIT", 2)
    calls = []

    def opener(req, timeout):
        calls.append(req.full_url)
        from urllib.parse import parse_qs, urlparse

        cursor = int(parse_qs(urlparse(req.full_url).query)["startTime"][0])
        rows = [r for r in raw_rows(3) if r[0] >= cursor][:1]  # short page is not EOF
        return io.BytesIO(json.dumps(rows).encode())

    result = sources.fetch_api_klines(
        "BTCUSDT", "1d", START, START + 3 * DAY_MS, opener=opener, pause_secs=0
    )
    assert result == raw_rows() and len(calls) == 4
    with pytest.raises(sources.DownloadError):
        sources.fetch_api_klines(
            "BTCUSDT",
            "1d",
            START,
            START + 3 * DAY_MS,
            opener=lambda req, timeout: io.BytesIO(json.dumps(raw_rows(1)).encode()),
            pause_secs=0,
        )


def test_retry_bounded_and_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr(sources.time, "sleep", sleeps.append)
    calls = []

    def opener(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 429, "limited", {"Retry-After": "3"}, None)

    with pytest.raises(sources.DownloadError):
        sources.http_get("https://example.test", opener=opener)
    assert len(calls) == sources.MAX_ATTEMPTS
    assert sleeps == [3, 3, 4, 8]

    def banned(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "limited", {"Retry-After": "120"}, None)

    with pytest.raises(sources.DownloadError, match="resume later"):
        sources.http_get("https://example.test", opener=banned)


def mock_archive(monkeypatch):
    calls = []

    def fetch(ref):
        calls.append(ref)
        return {"csv": csv_bytes(31), "url": ref.url, "sha256": "a" * 64, "checksum_verified": True}

    monkeypatch.setattr(sources, "fetch_archive", fetch)
    return calls


def test_resume_hash_revalidation_and_range_expansion(tmp_path, monkeypatch):
    calls = mock_archive(monkeypatch)
    args = (tmp_path, ["BTCUSDT"], ["1d"], date(2024, 1, 1), date(2024, 1, 3))
    first = download(*args, funding=False)
    assert validate_dataset(first)["rows"] == 3
    download(*args, funding=False)
    assert len(calls) == 1
    second = download(
        tmp_path, ["BTCUSDT"], ["1d"], date(2024, 1, 1), date(2024, 1, 5), funding=False
    )
    assert len(calls) == 2
    assert validate_dataset(first)["rows"] == 3  # older immutable snapshot survives update
    assert validate_dataset(second)["rows"] == 5
    entry = json.loads(second.read_text())["partitions"][0]
    (tmp_path / entry["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(second)
    download(tmp_path, ["BTCUSDT"], ["1d"], date(2024, 1, 1), date(2024, 1, 5), funding=False)
    assert len(calls) == 3


def test_daily_fallback_resumes_interrupted_partition(tmp_path, monkeypatch):
    calls = []
    fail = [True]

    def fetch(ref):
        if ref.day is None:
            raise sources.NotFound("monthly missing")
        calls.append(ref.day)
        if ref.day == 2 and fail[0]:
            raise sources.DownloadError("interrupted")
        return {
            "csv": csv_bytes(1, START + (ref.day - 1) * DAY_MS),
            "url": ref.url,
            "checksum_verified": True,
        }

    monkeypatch.setattr(sources, "fetch_archive", fetch)
    args = (tmp_path, ["BTCUSDT"], ["1d"], date(2024, 1, 1), date(2024, 1, 3))
    with pytest.raises(sources.DownloadError):
        download(*args, funding=False)
    fail[0] = False
    path = download(*args, funding=False)
    assert calls == [1, 2, 2, 3]
    assert validate_dataset(path)["ok"]


def test_cross_partition_overlap_and_metadata_corruption(tmp_path, monkeypatch):
    mock_archive(monkeypatch)
    path = download(
        tmp_path, ["BTCUSDT"], ["1d"], date(2024, 1, 1), date(2024, 1, 3), funding=False
    )
    data = json.loads(path.read_text())
    data["partitions"].append(data["partitions"][0])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap"):
        validate_dataset(path)
    data["partitions"].pop()
    data["partitions"][0]["rows"] = 900
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata"):
        validate_dataset(path)


def test_atomic_write_interruption_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "x.parquet"
    original = pd.DataFrame({"x": [1]})
    atomic_write_parquet(original, path)
    before = path.read_bytes()

    def fail(self, target, **kw):
        Path(target).write_bytes(b"partial")
        raise KeyboardInterrupt()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_parquet(original, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_funding_api_pagination_and_gaps(monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    events = [
        {"fundingTime": START + i * 28_800_000, "fundingRate": "0.0001", "markPrice": "100"}
        for i in range(3)
    ]
    calls = []

    def get(url):
        from urllib.parse import parse_qs, urlparse

        cursor = int(parse_qs(urlparse(url).query)["startTime"][0])
        calls.append(cursor)
        return json.dumps([r for r in events if r["fundingTime"] >= cursor][:1]).encode()

    monkeypatch.setattr(sources, "http_get", get)
    rows, _ = funding_api("BTCUSDT", START, START + DAY_MS)
    assert len(calls) == 4
    assert check_funding(rows, START, START + DAY_MS)["ok"]
    assert not check_funding(rows[:1], START, START + DAY_MS)["ok"]


def test_cli_validation_failure_is_nonzero(tmp_path, capsys):
    from kronos_mt5.marketdata.__main__ import main

    path = tmp_path / "bad.json"
    path.write_text("{}")
    assert main(["validate", "--manifest", str(path)]) == 1
    assert json.loads(capsys.readouterr().err)["ok"] is False


def test_real_funding_timestamp_jitter_preserved():
    rows = [
        {
            "funding_time": START + i * 28_800_000 + (i % 2),
            "funding_rate": 0.0001,
            "mark_price": None,
        }
        for i in range(3)
    ]
    result = check_funding(rows, START, START + DAY_MS)
    assert result["ok"] and rows[1]["funding_time"] == START + 28_800_001
    assert not check_funding([rows[0], rows[2]], START, START + DAY_MS)["ok"]


def test_naive_now_and_zero_intervals_rejected():
    from kronos_mt5.marketdata.spec import interval_to_timedelta_seconds

    with pytest.raises(ValueError):
        sources.latest_complete_utc_day(datetime(2024, 1, 1))  # noqa: DTZ001
    with pytest.raises(ValueError):
        interval_to_timedelta_seconds("0m")


def test_corrupted_archive_never_falls_back_to_api(tmp_path, monkeypatch):
    def bad(ref):
        raise sources.ChecksumMismatch("bad checksum")

    monkeypatch.setattr(sources, "fetch_archive", bad)
    monkeypatch.setattr(
        sources, "fetch_api_klines", lambda *a, **kw: pytest.fail("must not hide corruption")
    )
    with pytest.raises(sources.ChecksumMismatch):
        download(tmp_path, ["BTCUSDT"], ["1d"], date(2024, 1, 1), date(2024, 1, 3), funding=False)
    assert not list(tmp_path.rglob("*.parquet"))
