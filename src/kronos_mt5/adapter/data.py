"""LiveMarketDataClient for MT5.

Responsibilities:
  - On subscribe_bars: start a polling loop that calls copy_rates and emits
    Nautilus Bar objects for newly CLOSED bars only.
  - On subscribe_quote_ticks: poll symbol_info_tick -> QuoteTick.
  - Normalize all timestamps to UNIX epoch NANOSECONDS.
  - De-duplicate: never re-emit a bar already published (track last bar time).

VERIFY base class + handler method names against installed NT version.
Docs: https://nautilustrader.io/docs/latest/concepts/adapters/
"""
from __future__ import annotations

# from nautilus_trader.live.data_client import LiveMarketDataClient
from kronos_mt5.adapter.mt5_client import MT5Client


class MT5DataClient:  # TODO: subclass LiveMarketDataClient
    def __init__(self, client: MT5Client, provider, config) -> None:
        self._client = client
        self._provider = provider
        self._config = config
        self._last_bar_ts: dict[str, int] = {}

    async def _connect(self) -> None:
        await self._client.connect()
        await self._provider.load_all_async()

    async def _disconnect(self) -> None:
        await self._client.shutdown()

    def subscribe_bars(self, bar_type) -> None:
        # TODO: launch asyncio task polling copy_rates at poll_interval_ms;
        #       convert MT5 rates -> Bar; self._handle_data(bar) for closed bars.
        raise NotImplementedError

    def subscribe_quote_ticks(self, instrument_id) -> None:
        raise NotImplementedError
