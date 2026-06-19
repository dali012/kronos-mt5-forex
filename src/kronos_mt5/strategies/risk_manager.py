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

from nautilus_trader.model.data import MarkPriceUpdate
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
        # Live-only mark stream health. Backtests leave the watchdog disabled.
        self.mark_watchdog_enabled = False
        self.market_data_stale = False
        self.stale_mark_symbols: tuple[str, ...] = ()
        self.mark_prices: dict[str, float] = {}
        self.mark_received_ns: dict[str, int] = {}
        self.mark_age_secs: dict[str, float | None] = {}

    @property
    def halted(self) -> bool:
        return self.mode in ("halt", "kill") or self.market_data_stale

    def enable_mark_watchdog(self, instrument_ids: tuple[InstrumentId, ...]) -> None:
        """Fail closed until every configured instrument has delivered a mark."""
        self.mark_watchdog_enabled = True
        self.market_data_stale = True
        self.stale_mark_symbols = tuple(iid.symbol.value for iid in instrument_ids)
        self.mark_age_secs = {iid.symbol.value: None for iid in instrument_ids}

    def update_mark(self, update: MarkPriceUpdate, received_ns: int) -> None:
        key = str(update.instrument_id)
        self.mark_prices[key] = float(update.value)
        self.mark_received_ns[key] = received_ns

    def evaluate_mark_freshness(
        self,
        instrument_ids: tuple[InstrumentId, ...],
        now_ns: int,
        stale_after_secs: float,
    ) -> tuple[str, ...]:
        """Update health metadata and return missing/stale instrument symbols."""
        stale: list[str] = []
        ages: dict[str, float | None] = {}
        for iid in instrument_ids:
            received_ns = self.mark_received_ns.get(str(iid))
            age = None if received_ns is None else max(0.0, (now_ns - received_ns) / 1e9)
            ages[iid.symbol.value] = age
            if age is None or age > stale_after_secs:
                stale.append(iid.symbol.value)
        self.mark_age_secs = ages
        self.stale_mark_symbols = tuple(stale)
        self.market_data_stale = bool(stale)
        return self.stale_mark_symbols

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


class MarkPriceMonitorConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple[str, ...]
    check_secs: int = 5
    stale_after_secs: int = 15


class MarkPriceMonitor(Strategy):
    """Subscribe to live futures marks and fail closed when their stream is stale.

    Nautilus' Portfolio consumes the same MarkPriceUpdate events when configured
    with ``PortfolioConfig(use_mark_prices=True)``, so all existing portfolio PnL
    consumers become live mark-to-market without duplicating accounting logic.
    """

    def __init__(self, config: MarkPriceMonitorConfig) -> None:
        super().__init__(config)
        self._iids = tuple(InstrumentId.from_str(i) for i in config.instrument_ids)
        self.risk_state: RiskState | None = None

    def on_start(self) -> None:
        if self.risk_state is None:
            self.log.error("mark watchdog has no shared RiskState; stopping")
            self.stop()
            return
        self.risk_state.enable_mark_watchdog(self._iids)
        for iid in self._iids:
            self.subscribe_mark_prices(iid)
        self.log.warning(
            f"MARK WATCHDOG: waiting for {len(self._iids)} mark streams; new entries blocked"
        )
        self.clock.set_timer(
            name="mark_freshness",
            interval=timedelta(seconds=self.config.check_secs),
            callback=self._check_freshness,
        )

    def on_mark_price(self, update: MarkPriceUpdate) -> None:
        if self.risk_state is None or update.instrument_id not in self._iids:
            return
        self.risk_state.update_mark(update, self.clock.timestamp_ns())
        self._evaluate_freshness()

    def _check_freshness(self, event) -> None:  # noqa: ANN001
        self._evaluate_freshness()

    def _evaluate_freshness(self) -> None:
        if self.risk_state is None:
            return
        was_stale = self.risk_state.market_data_stale
        stale = self.risk_state.evaluate_mark_freshness(
            self._iids,
            self.clock.timestamp_ns(),
            self.config.stale_after_secs,
        )
        if stale and not was_stale:
            self.log.error(
                "MARK WATCHDOG: stale streams "
                f"({', '.join(stale)}); new entries halted, positions/stops kept"
            )
        elif not stale and was_stale:
            self.log.warning("MARK WATCHDOG: all mark streams fresh; entry gate released")

    def on_stop(self) -> None:
        for iid in self._iids:
            self.unsubscribe_mark_prices(iid)


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
        if self.risk_state is None or self.risk_state.mode == "kill":
            return
        # Never make a portfolio kill decision from stale marks. New entries are
        # already blocked by the watchdog; exchange-native position stops remain live.
        if self.risk_state.mark_watchdog_enabled and self.risk_state.market_data_stale:
            return
        equity = self._equity()
        if equity <= 0:
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
