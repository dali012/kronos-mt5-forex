"""Portfolio-level risk controls for the trend book.

RiskManager enforces a drawdown kill-switch: if mark-to-market equity falls more
than `max_drawdown` from its peak it sets mode='kill' (persisted to the store, so
it survives a bot restart). The TrendStrategies read the shared RiskState and
flatten + stop entering. Position-level stop-loss/take-profit live in TrendStrategy
as exchange-native reduce-only orders.
"""
from __future__ import annotations

from datetime import timedelta

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from kronos_mt5.companion import store


class RiskState:
    """Shared control state. The bot-side Companion syncs it from the store
    every few seconds; strategies read `mode` and the one-shot flatten request."""

    def __init__(self) -> None:
        self.mode = "run"            # run | halt | kill
        self.flatten_seq = 0         # bumped for one-shot flatten requests
        self.flatten_target = "all"  # 'all' or a symbol like 'BTCUSDT-PERP'

    @property
    def halted(self) -> bool:
        return self.mode in ("halt", "kill")


class RiskManagerConfig(StrategyConfig, frozen=True):
    venue: str
    instrument_ids: tuple[str, ...]
    max_drawdown: float = 0.20
    check_secs: int = 300
    db_path: str = store.DEFAULT_DB
    persist: bool = True  # write kill mode to the store (live); False in backtests


class RiskManager(Strategy):
    def __init__(self, config: RiskManagerConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue)
        self._iids = [InstrumentId.from_str(i) for i in config.instrument_ids]
        self._peak = 0.0
        self.risk_state: RiskState | None = None

    def on_start(self) -> None:
        self.clock.set_timer(
            name="risk_check",
            interval=timedelta(seconds=self.config.check_secs),
            callback=self._check,
        )

    def _equity(self) -> float:
        acct = self.portfolio.account(self._venue)
        if acct is None:
            return 0.0
        cash = acct.balance_total(USDT)
        equity = cash.as_double() if cash is not None else 0.0
        for iid in self._iids:
            upnl = self.portfolio.unrealized_pnl(iid)
            if upnl is not None:
                equity += upnl.as_double()
        return equity

    def _check(self, event) -> None:  # noqa: ANN001
        equity = self._equity()
        if equity <= 0 or self.risk_state is None or self.risk_state.mode == "kill":
            return
        self._peak = max(self._peak, equity)
        dd = equity / self._peak - 1.0
        if dd <= -self.config.max_drawdown:
            self.risk_state.mode = "kill"
            if self.config.persist:
                try:
                    store.set_mode("kill", self.config.db_path)  # persist (survives restart)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"persist kill mode failed: {exc!r}")
            self.log.error(
                f"DRAWDOWN KILL-SWITCH: equity ${equity:,.0f} (peak ${self._peak:,.0f}, "
                f"dd {dd:.1%}) <= -{self.config.max_drawdown:.0%} -> kill (flatten + halt)"
            )
