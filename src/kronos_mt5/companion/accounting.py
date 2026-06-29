"""Authenticated Binance income-ledger polling for performance attribution."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from datetime import datetime, timedelta, timezone
from typing import Callable

from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from kronos_mt5.companion import store


class BinanceIncomeClient:
    """Small signed client for ``GET /fapi/v1/income``.

    The endpoint is the exchange source of truth for REALIZED_PNL, COMMISSION,
    FUNDING_FEE, transfers, rebates, and other wallet adjustments.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        *,
        timeout_secs: float = 10.0,
        opener: Callable = urllib.request.urlopen,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._timeout_secs = timeout_secs
        self._opener = opener
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    def _request_page(self, start_time_ms: int, page: int, limit: int) -> list[dict]:
        params = {
            "startTime": start_time_ms,
            "page": page,
            "limit": limit,
            "recvWindow": 5000,
            "timestamp": self._clock_ms(),
        }
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        request = urllib.request.Request(
            f"{self._base_url}/fapi/v1/income?{query}&signature={signature}",
            headers={"X-MBX-APIKEY": self._api_key},
        )
        try:
            with self._opener(request, timeout=self._timeout_secs) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(
                f"Binance income HTTP {exc.code} {exc.reason}: {body}"
            ) from exc
        if not isinstance(payload, list):
            raise RuntimeError(f"Binance income request failed: {payload!r}")
        return payload

    def fetch_income(self, start_time_ms: int, limit: int = 1000) -> list[dict]:
        rows: list[dict] = []
        for page in range(1, 11):  # safety cap; normal five-minute polls are one page
            batch = self._request_page(start_time_ms, page, limit)
            rows.extend(batch)
            if len(batch) < limit:
                break
        return rows


class IncomePollerConfig(StrategyConfig, frozen=True):
    db_path: str = store.DEFAULT_DB
    base_url: str = "https://demo-fapi.binance.com"
    interval_secs: int = 300


class IncomePoller(Strategy):
    def __init__(self, config: IncomePollerConfig) -> None:
        super().__init__(config)
        # Inject after construction so secrets never appear in StrategyConfig/log output.
        self.api_key = ""
        self.api_secret = ""
        self._client: BinanceIncomeClient | None = None

    def on_start(self) -> None:
        if not self.api_key or not self.api_secret:
            self.log.error("income poller missing API credentials; stopping")
            self.stop()
            return
        self._client = BinanceIncomeClient(
            self.api_key,
            self.api_secret,
            self.config.base_url,
        )
        self._refresh(None)
        self.clock.set_timer(
            name="income_ledger",
            interval=timedelta(seconds=self.config.interval_secs),
            callback=self._refresh,
        )

    def _refresh(self, event) -> None:  # noqa: ANN001
        if self._client is None:
            return
        start_ts = store.get_kv("accounting_start_ts", None, self.config.db_path)
        if not start_ts:
            self.log.warning("income poll skipped: accounting baseline not ready")
            return
        baseline_ms = int(datetime.fromisoformat(start_ts).timestamp() * 1000)
        cursor = int(
            store.get_kv(
                "accounting_income_cursor_ms",
                str(baseline_ms),
                self.config.db_path,
            )
            or baseline_ms
        )
        try:
            rows = self._client.fetch_income(max(baseline_ms, cursor - 1))
        except Exception as exc:  # noqa: BLE001
            store.set_kv("accounting_income_error", repr(exc), self.config.db_path)
            self.log.warning(f"income ledger refresh failed: {exc!r}")
            return

        inserted = 0
        max_time = cursor
        for row in rows:
            time_ms = int(row.get("time") or 0)
            income_type = str(row.get("incomeType") or "UNKNOWN")
            tran_id = str(row.get("tranId") or "")
            trade_id = str(row.get("tradeId") or "")
            asset = str(row.get("asset") or "USDT")
            income_id = f"{income_type}:{tran_id}:{trade_id}:{time_ms}:{asset}"
            inserted += store.record_income(
                income_id=income_id,
                time_ms=time_ms,
                income_type=income_type,
                amount=float(row.get("income") or 0),
                symbol=str(row.get("symbol") or ""),
                asset=asset,
                trade_id=trade_id,
                tran_id=tran_id,
                info=str(row.get("info") or ""),
                db_path=self.config.db_path,
            )
            max_time = max(max_time, time_ms)
        store.set_kv("accounting_income_cursor_ms", str(max_time), self.config.db_path)
        store.set_kv(
            "accounting_income_last_success_ts",
            datetime.now(timezone.utc).isoformat(),
            self.config.db_path,
        )
        store.set_kv("accounting_income_error", "", self.config.db_path)
        if inserted:
            self.log.info(f"income ledger: recorded {inserted} new rows")
