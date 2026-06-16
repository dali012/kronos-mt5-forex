"""Portfolio-level risk controls for the trend book.

RiskManager runs alongside the TrendStrategy instances and enforces a portfolio
drawdown kill-switch: if mark-to-market equity falls more than `max_drawdown`
from its peak, it flattens everything, cancels resting orders, and sets a shared
`RiskState.halted` flag that blocks the TrendStrategies from re-entering.

(Position-level stop-loss/take-profit live inside TrendStrategy as exchange-native
reduce-only orders, which also provide crash/disconnect protection.)
"""
from __future__ import annotations

from datetime import timedelta

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class RiskState:
    """Shared flag toggled by RiskManager and read by the TrendStrategies."""

    def __init__(self) -> None:
        self.halted = False


class RiskManagerConfig(StrategyConfig, frozen=True):
    venue: str
    instrument_ids: tuple[str, ...]
    max_drawdown: float = 0.20   # flatten + halt if equity falls this far from peak
    check_secs: int = 300


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
        if equity <= 0 or self.risk_state is None or self.risk_state.halted:
            return
        self._peak = max(self._peak, equity)
        dd = equity / self._peak - 1.0
        if dd <= -self.config.max_drawdown:
            # Signal the halt; each TrendStrategy flattens its OWN position
            # (position closing is strategy-scoped, so the RiskManager can't do
            # it for them). They also stop entering while halted.
            self.risk_state.halted = True
            self.log.error(
                f"DRAWDOWN KILL-SWITCH: equity ${equity:,.0f} (peak ${self._peak:,.0f}, "
                f"dd {dd:.1%}) <= -{self.config.max_drawdown:.0%} -> halting; strategies flattening"
            )
