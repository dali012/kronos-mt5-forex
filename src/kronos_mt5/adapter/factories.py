"""Client factories the TradingNode uses to construct the MT5 adapter clients.

TODO(claude-code): implement LiveDataClientFactory / LiveExecClientFactory
subclasses returning the MT5 clients above, wiring a shared MT5Client instance
and the InstrumentProvider. Verify factory base classes in the installed version.
"""
from __future__ import annotations

# from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory


class MT5LiveDataClientFactory:  # TODO: subclass LiveDataClientFactory
    @staticmethod
    def create(loop, name, config, msgbus, cache, clock):
        raise NotImplementedError


class MT5LiveExecClientFactory:  # TODO: subclass LiveExecClientFactory
    @staticmethod
    def create(loop, name, config, msgbus, cache, clock):
        raise NotImplementedError
