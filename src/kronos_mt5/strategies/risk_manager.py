"""Portfolio-level risk controls for the trend book.

RiskManager enforces a drawdown kill-switch: if mark-to-market equity falls more
than `max_drawdown` from its peak it sets mode='kill' (persisted to the store, so
it survives a bot restart). The TrendStrategies read the shared RiskState and
flatten + stop entering. Position-level stop-loss/take-profit live in TrendStrategy
as exchange-native reduce-only orders.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from kronos_mt5.companion import store


class RiskState:
    """Shared control state. The bot-side Companion syncs it from the store
    every few seconds; strategies read `mode` and the one-shot flatten request."""

    def __init__(self) -> None:
        self.mode = "run"  # run | halt | kill
        self.flatten_seq = 0  # bumped for one-shot flatten requests
        self.flatten_target = "all"  # 'all' or a symbol like 'BTCUSDT-PERP'
        self.returns_by_symbol: dict[str, np.ndarray] = {}

    @property
    def halted(self) -> bool:
        return self.mode in ("halt", "kill")

    def update_returns(self, symbol: str, returns: np.ndarray) -> None:
        self.returns_by_symbol[symbol] = np.asarray(returns, dtype=float)

    def correlation_scalar(
        self,
        window: int,
        threshold: float,
        min_scalar: float,
    ) -> float:
        series = [r[-window:] for r in self.returns_by_symbol.values() if len(r) >= window]
        if len(series) < 2:
            return 1.0
        arr = np.vstack(series)
        corr = np.corrcoef(arr)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]
        if len(upper) == 0:
            return 1.0
        avg_corr = float(np.mean(np.abs(upper)))
        if avg_corr <= threshold:
            return 1.0
        span = max(1e-9, 1.0 - threshold)
        pressure = min(1.0, (avg_corr - threshold) / span)
        return max(min_scalar, 1.0 - pressure * (1.0 - min_scalar))


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
        if self.config.persist:
            try:
                store.init_db(self.config.db_path)
                self._peak = float(store.get_kv("risk_peak_equity", "0", self.config.db_path) or 0)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"load risk peak failed: {exc!r}")
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
        if equity > self._peak:
            self._peak = equity
            if self.config.persist:
                try:
                    store.set_kv("risk_peak_equity", f"{self._peak:.8f}", self.config.db_path)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"persist risk peak failed: {exc!r}")
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
