"""Config objects for the MT5 adapter clients.

TODO(claude-code): subclass the NautilusTrader live client config base classes
for the installed version (e.g. LiveDataClientConfig / LiveExecClientConfig).
Verify the exact base class names in your version.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MT5InstrumentProviderConfig:
    symbols: tuple[str, ...] = ("EURUSD",)
    load_all: bool = False


@dataclass
class MT5DataClientConfig:
    login: int
    password: str
    server: str
    terminal_path: str | None = None
    poll_interval_ms: int = 250          # rate/tick polling cadence


@dataclass
class MT5ExecClientConfig:
    login: int
    password: str
    server: str
    terminal_path: str | None = None
    reconcile_on_start: bool = True
