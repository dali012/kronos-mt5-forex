"""Async wrapper around the (blocking, Windows-only) `MetaTrader5` package.

The official `MetaTrader5` Python package is synchronous and talks to a locally
running MT5 terminal. NautilusTrader's live clients are async, so EVERY MT5 call
must be pushed off the event loop. This class is the single choke point that does
that (via run_in_executor); the data/execution clients call only these methods.

Windows-only. On a non-Windows backtest box, this module must never be imported
(backtests bypass the adapter entirely).
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class MT5Client:
    def __init__(self, login: int, password: str, server: str, terminal_path: str | None = None) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")
        self._connected = False

    async def _call(self, fn, *args, **kwargs) -> Any:
        """Run a blocking MT5 SDK call in the dedicated thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    async def connect(self) -> None:
        """initialize() + login(). Raise on failure. Assert demo before returning.

        TODO(claude-code):
          import MetaTrader5 as mt5
          ok = await self._call(mt5.initialize, path=self._terminal_path) if path else
               await self._call(mt5.initialize)
          ok = await self._call(mt5.login, self._login, password=..., server=...)
          info = await self._call(mt5.account_info)
          assert info.trade_mode indicates DEMO  # refuse live
        """
        raise NotImplementedError

    async def shutdown(self) -> None:
        # TODO: await self._call(mt5.shutdown); self._executor.shutdown()
        ...

    # --- reference data ---
    async def symbol_info(self, symbol: str): ...          # -> mt5.SymbolInfo
    async def symbol_info_tick(self, symbol: str): ...      # -> latest bid/ask

    # --- market data ---
    async def copy_rates(self, symbol: str, timeframe: int, count: int): ...
    async def copy_ticks(self, symbol: str, count: int): ...

    # --- trading ---
    async def order_send(self, request: dict): ...          # -> mt5.OrderSendResult
    async def positions_get(self, symbol: str | None = None): ...
    async def history_deals_get(self, *args, **kwargs): ...
