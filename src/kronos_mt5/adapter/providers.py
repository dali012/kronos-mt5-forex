"""InstrumentProvider: build Nautilus CurrencyPair instruments from MT5 symbols.

Map mt5.symbol_info fields -> Nautilus instrument:
  digits/point      -> price_precision / price_increment
  volume_min/step/max-> size_precision / size_increment / limits
  trade_contract_size-> contract multiplier
  currency_base/profit/margin -> the three currencies

Docs: https://nautilustrader.io/docs/latest/concepts/adapters/
Reference: study nautilus_trader.adapters.* InstrumentProvider implementations.
"""
from __future__ import annotations

# from nautilus_trader.common.providers import InstrumentProvider
from kronos_mt5.adapter.mt5_client import MT5Client


class MT5InstrumentProvider:  # TODO: subclass InstrumentProvider
    def __init__(self, client: MT5Client, config) -> None:
        self._client = client
        self._config = config

    async def load_all_async(self, filters=None) -> None:
        # TODO: for each configured symbol -> symbol_info -> build CurrencyPair
        #       -> self.add(instrument)
        raise NotImplementedError

    async def load_async(self, instrument_id, filters=None) -> None:
        raise NotImplementedError
