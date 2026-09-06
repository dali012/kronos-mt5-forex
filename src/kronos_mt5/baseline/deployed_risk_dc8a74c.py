"""Portfolio-level risk controls for the trend book.

RiskManager enforces a drawdown kill-switch: if mark-to-market equity falls more
than `max_drawdown` from its peak it sets mode='kill' (persisted to the store, so
it survives a bot restart). The TrendStrategies read the shared RiskState and
flatten + stop entering. Position-level stop-loss/take-profit live in TrendStrategy
as exchange-native reduce-only orders.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

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
        self.run_id = uuid4().hex
        # Portfolio allocators publish one common scale for the *next* cycle.
        # That one-cycle lag keeps all strategies on the same scale even though
        # Nautilus dispatches their same-timestamp bars sequentially.
        self.allocation_scales: dict[str, float] = {}
        self.allocation_metrics: dict[str, dict] = {}
        self._allocation_pending: dict[str, dict[str, dict[str, float]]] = {}
        self._allocation_completed: dict[str, set[str]] = {}
        self._shadow_target_queue: dict[tuple[str, str, str], dict] = {}
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

    def publish_portfolio_target(
        self,
        model: str,
        symbol: str,
        raw_weight: float,
        returns: np.ndarray,
        cycle_id: str | int,
        expected_symbols: tuple[str, ...],
        *,
        window: int,
        target_vol: float,
        ppy: int,
        shrinkage: float,
        min_scale: float,
        max_scale: float,
        max_gross: float,
        min_observations: int,
        scale_up_alpha: float,
        scale_deadband: float,
    ) -> float:
        """Publish one target and return the stable scale for this cycle.

        Once every expected symbol has published for ``cycle_id``, a shrunk
        covariance estimate produces the scale activated on the next cycle.
        This avoids order-dependent sizing across sibling strategies.
        """
        current_scale = float(self.allocation_scales.get(model, 1.0))
        self.update_returns(symbol, returns)
        cycle = str(cycle_id)
        completed = self._allocation_completed.setdefault(model, set())
        if cycle in completed:
            return current_scale

        cycles = self._allocation_pending.setdefault(model, {})
        weights = cycles.setdefault(cycle, {})
        weights[symbol] = float(raw_weight)
        expected = tuple(dict.fromkeys(expected_symbols))
        if expected and set(expected).issubset(weights):
            metrics = self._portfolio_allocation_metrics(
                {name: weights[name] for name in expected},
                window=window,
                target_vol=target_vol,
                ppy=ppy,
                shrinkage=shrinkage,
                min_scale=min_scale,
                max_scale=max_scale,
                max_gross=max_gross,
                min_observations=min_observations,
            )
            candidate_scale = float(metrics["scale"])
            previous_scale = float(self.allocation_scales.get(model, 1.0))
            if candidate_scale < previous_scale:
                activated_scale = candidate_scale  # de-risk immediately
            else:
                alpha = float(np.clip(scale_up_alpha, 0.0, 1.0))
                activated_scale = previous_scale + alpha * (candidate_scale - previous_scale)
                relative_change = abs(activated_scale - previous_scale) / max(
                    abs(previous_scale), 1e-9
                )
                if relative_change < max(0.0, float(scale_deadband)):
                    activated_scale = previous_scale
            # Gross is a hard limit even while scale increases are smoothed.
            gross_before = float(metrics["gross_before"])
            if max_gross > 0 and gross_before > 0:
                activated_scale = min(activated_scale, float(max_gross) / gross_before)
            metrics["candidate_scale"] = candidate_scale
            metrics["scale"] = max(0.0, activated_scale)
            metrics["gross_after"] = gross_before * metrics["scale"]
            forecast = metrics["forecast_vol_before"]
            metrics["forecast_vol_after"] = (
                forecast * metrics["scale"] if forecast is not None else None
            )
            metrics.update({"model": model, "source_cycle": cycle})
            self.allocation_scales[model] = float(metrics["scale"])
            self.allocation_metrics[model] = metrics
            completed.add(cycle)
            cycles.pop(cycle, None)
            # Bound state during long-running processes and backtests.
            while len(completed) > 16:
                completed.pop()
            while len(cycles) > 4:
                cycles.pop(next(iter(cycles)))
        return current_scale

    def _portfolio_allocation_metrics(
        self,
        raw_weights: dict[str, float],
        *,
        window: int,
        target_vol: float,
        ppy: int,
        shrinkage: float,
        min_scale: float,
        max_scale: float,
        max_gross: float,
        min_observations: int,
    ) -> dict:
        symbols = tuple(raw_weights)
        n_assets = len(symbols)
        portfolio_weights = np.asarray(
            [raw_weights[symbol] / max(1, n_assets) for symbol in symbols], dtype=float
        )
        gross_before = float(np.abs(portfolio_weights).sum())
        available = [self.returns_by_symbol.get(symbol) for symbol in symbols]
        ready = bool(
            available
            and all(series is not None and len(series) >= min_observations for series in available)
        )
        forecast_vol = None
        vol_scale = 1.0
        if ready:
            length = min(window, *(len(series) for series in available if series is not None))
            matrix = np.vstack([series[-length:] for series in available if series is not None])
            covariance = np.atleast_2d(np.cov(matrix, ddof=1)) * float(ppy)
            shrink = float(np.clip(shrinkage, 0.0, 1.0))
            covariance = (1.0 - shrink) * covariance + shrink * np.diag(
                np.diag(covariance)
            )
            variance = float(portfolio_weights @ covariance @ portfolio_weights)
            forecast_vol = float(np.sqrt(max(0.0, variance)))
            if forecast_vol > 1e-12:
                vol_scale = max(0.0, float(target_vol)) / forecast_vol

        lower = max(0.0, float(min_scale))
        upper = max(lower, float(max_scale))
        scale = float(np.clip(vol_scale, lower, upper)) if ready else min(1.0, upper)
        if max_gross > 0 and gross_before > 0:
            scale = min(scale, float(max_gross) / gross_before)
        scale = max(0.0, scale)
        return {
            "ready": ready,
            "scale": scale,
            "forecast_vol_before": forecast_vol,
            "forecast_vol_after": forecast_vol * scale if forecast_vol is not None else None,
            "target_vol": float(target_vol),
            "gross_before": gross_before,
            "gross_after": gross_before * scale,
            "asset_count": n_assets,
            "window": int(window),
            "shrinkage": float(np.clip(shrinkage, 0.0, 1.0)),
        }

    def queue_shadow_target(self, row: dict) -> None:
        key = (str(row["model"]), str(row["cycle_id"]), str(row["symbol"]))
        self._shadow_target_queue[key] = dict(row)

    def pending_shadow_targets(self) -> list[dict]:
        return list(self._shadow_target_queue.values())

    def acknowledge_shadow_targets(self, rows: list[dict]) -> None:
        for row in rows:
            key = (str(row["model"]), str(row["cycle_id"]), str(row["symbol"]))
            self._shadow_target_queue.pop(key, None)

    def correlation_scalar(
        self,
        window: int,
        threshold: float,
        min_scalar: float,
    ) -> float:
        return float(self.correlation_metrics(window, threshold, min_scalar)["risk_scalar"])

    def correlation_metrics(
        self,
        window: int,
        threshold: float,
        min_scalar: float,
    ) -> dict[str, float | int | str | None]:
        """Describe the book as correlated bets, not a count of symbols.

        ``effective_bets`` uses the standard equal-correlation approximation
        N / (1 + (N - 1) rho). Eight perfectly correlated coins therefore count
        as one effective bet, while genuinely independent series approach eight.
        """
        series = [r[-window:] for r in self.returns_by_symbol.values() if len(r) >= window]
        vols = [float(np.std(r, ddof=1) * np.sqrt(365)) for r in series if len(r) > 1]
        average_vol = float(np.mean(vols)) if vols else None
        if average_vol is None:
            vol_regime = "UNKNOWN"
        elif average_vol < 0.40:
            vol_regime = "LOW_VOL"
        elif average_vol > 0.80:
            vol_regime = "HIGH_VOL"
        else:
            vol_regime = "NORMAL_VOL"
        if len(series) < 2:
            return {
                "series_count": len(series),
                "average_abs_correlation": None,
                "effective_bets": None,
                "risk_scalar": 1.0,
                "average_annualized_volatility": average_vol,
                "volatility_regime": vol_regime,
            }
        arr = np.vstack(series)
        corr = np.corrcoef(arr)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]
        if len(upper) == 0:
            return {
                "series_count": len(series),
                "average_abs_correlation": None,
                "effective_bets": None,
                "risk_scalar": 1.0,
                "average_annualized_volatility": average_vol,
                "volatility_regime": vol_regime,
            }
        avg_corr = float(np.mean(np.abs(upper)))
        effective = len(series) / (1.0 + (len(series) - 1.0) * avg_corr)
        if avg_corr <= threshold:
            scalar = 1.0
        else:
            span = max(1e-9, 1.0 - threshold)
            pressure = min(1.0, (avg_corr - threshold) / span)
            scalar = max(min_scalar, 1.0 - pressure * (1.0 - min_scalar))
        return {
            "series_count": len(series),
            "average_abs_correlation": avg_corr,
            "effective_bets": float(effective),
            "risk_scalar": float(scalar),
            "average_annualized_volatility": average_vol,
            "volatility_regime": vol_regime,
        }


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
    # Only record a MARK_STALE incident once staleness has persisted this long, so
    # transient exchange feed flaps (common on testnet) don't spam the incident log.
    # The entry gate itself still fails closed instantly on the first stale mark.
    incident_min_secs: int = 10
    db_path: str = store.DEFAULT_DB


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
        # Debounce state: the entry gate reacts instantly, but incident/alert
        # logging waits until staleness is confirmed to persist.
        self._stale_since_ns: int | None = None
        self._incident_open = False
        self._warmed_up = False

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
        now_ns = self.clock.timestamp_ns()
        # evaluate_mark_freshness sets risk_state.market_data_stale immediately, so
        # the entry gate fails closed on the very first stale mark (fail-safe). The
        # logging/incident side below is debounced to filter transient feed flaps.
        stale = self.risk_state.evaluate_mark_freshness(
            self._iids,
            now_ns,
            self.config.stale_after_secs,
        )
        if stale:
            if self._stale_since_ns is None:
                self._stale_since_ns = now_ns
            persisted = (now_ns - self._stale_since_ns) / 1e9
            if (
                self._warmed_up
                and not self._incident_open
                and persisted >= self.config.incident_min_secs
            ):
                try:
                    store.start_incident("MARK_STALE", ",".join(stale), self.config.db_path)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"record stale-mark incident failed: {exc!r}")
                self._incident_open = True
                self.log.error(
                    "MARK WATCHDOG: stale streams "
                    f"({', '.join(stale)}); new entries halted, positions/stops kept"
                )
        else:
            if not self._warmed_up:
                self._warmed_up = True
                self.log.warning("MARK WATCHDOG: all mark streams fresh; entry gate released")
            elif self._incident_open:
                try:
                    store.resolve_incident("MARK_STALE", self.config.db_path)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"resolve stale-mark incident failed: {exc!r}")
                self._incident_open = False
                self.log.warning("MARK WATCHDOG: all mark streams fresh; entry gate released")
            self._stale_since_ns = None

    def on_stop(self) -> None:
        # Binance's Nautilus adapter does not implement mark-price unsubscribe
        # in 1.228.0. The client disconnect performed by node disposal releases
        # these subscriptions, so avoid noisy NotImplementedError shutdown logs.
        pass


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
