"""LiveExecutionClient for MT5.

Responsibilities:
  - Translate Nautilus order commands -> MT5 order_send request dicts
    (SymbolFilling, deviation/slippage, SL/TP prices, magic number).
  - Poll positions_get / history_deals_get to emit fill, order-status, and
    position events back into the engine.
  - Reconcile on startup: pull current MT5 positions/orders and rebuild engine
    state so we never double-trade after a restart.

This is the riskiest module. Get reconciliation and fill mapping right, and add
a fail-safe: on disconnect or ambiguous state, stop NEW entries.

VERIFY base class + command handler names against installed NT version.
Docs: https://nautilustrader.io/docs/latest/concepts/execution/
"""
from __future__ import annotations

# from nautilus_trader.live.execution_client import LiveExecutionClient
from kronos_mt5.adapter.mt5_client import MT5Client


class MT5ExecutionClient:  # TODO: subclass LiveExecutionClient
    def __init__(self, client: MT5Client, provider, config) -> None:
        self._client = client
        self._provider = provider
        self._config = config

    async def _connect(self) -> None:
        await self._client.connect()
        if self._config.reconcile_on_start:
            await self._reconcile()

    async def _reconcile(self) -> None:
        # TODO: positions_get + open orders -> rebuild engine position state
        raise NotImplementedError

    def submit_order(self, command) -> None:
        # TODO: build mt5 request dict; order_send via client; emit events
        raise NotImplementedError

    def cancel_order(self, command) -> None:
        raise NotImplementedError

    def modify_order(self, command) -> None:
        raise NotImplementedError
