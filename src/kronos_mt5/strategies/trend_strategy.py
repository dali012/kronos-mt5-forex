"""TrendStrategy — event-driven vol-targeted trend-following (NautilusTrader).

Engine port of the vectorized strategy in backtest/trend.py. One instance trades
one instrument; run N instances for the diversified portfolio.

Per closed bar: ensemble sign-momentum signal in [-1,1], vol-targeted to a per
-instrument risk budget, rebalanced toward the target via a market order.

Risk controls:
  - Exchange-native protective orders per position (reduce-only): a STOP_MARKET
    stop-loss, optional LIMIT take-profit, and optional native volatility trailing
    stop which activates around +1R. These live server-side, so they protect even
    if this process crashes / disconnects.
  - A shared `risk_state.halted` gate (set by the RiskManager drawdown
    kill-switch) blocks new entries.

Verified against NautilusTrader 1.228.0.
"""

from __future__ import annotations

import contextlib
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TrailingOffsetType, TriggerType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class TrendStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    portfolio_instrument_ids: tuple[str, ...] = ()
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    vol_window: int = 33
    target_vol: float = 0.15
    max_leverage: float = 2.0
    n_instruments: int = 1
    ppy: int = 365
    vol_floor: float = 0.02
    rebalance_threshold: float = 0.10
    warmup_request: bool = True
    # --- risk controls ---
    use_stop_loss: bool = True
    stop_pct: float = 0.20  # close if price moves this far against the entry
    use_vol_stop: bool = False
    stop_vol_mult: float = 4.0
    min_stop_pct: float = 0.08
    max_stop_pct: float = 0.30
    use_take_profit: bool = False  # off by default: TP conflicts with trend-following
    tp_pct: float = 0.50
    # Native Binance trailing stop: hard stop stays live; trail arms around +1R.
    use_trailing_stop: bool = False
    trailing_activation_r: float = 1.0
    trailing_vol_mult: float = 3.0
    min_trailing_pct: float = 0.005
    max_trailing_pct: float = 0.10  # Binance callback-rate ceiling
    flatten_on_stop: bool = False
    # --- portfolio/friction controls ---
    use_correlation_scaling: bool = False
    corr_window: int = 90
    corr_threshold: float = 0.65
    corr_min_scalar: float = 0.50
    use_portfolio_allocator: bool = False
    portfolio_target_vol: float = 0.10
    portfolio_cov_window: int = 90
    portfolio_cov_shrinkage: float = 0.25
    portfolio_min_scale: float = 0.25
    portfolio_max_scale: float = 1.25
    portfolio_max_gross: float = 1.50
    portfolio_min_observations: int = 30
    portfolio_scale_up_alpha: float = 0.20
    portfolio_scale_deadband: float = 0.05
    funding_filter_enabled: bool = False
    funding_rate_limit: float = 0.0010
    funding_continuous_sizing: bool = False
    funding_rate_soft_limit: float = 0.0001
    funding_min_scalar: float = 0.0
    cost_aware_rebalance: bool = False
    round_trip_cost_bps: float = 8.0
    slippage_bps: float = 2.0
    cost_threshold_mult: float = 4.0
    min_notional_buffer: float = 1.05
    use_patient_limit: bool = False
    patient_limit_offset_bps: float = 2.0
    patient_limit_timeout_secs: int = 300
    patient_limit_market_fallback: bool = True
    shadow_enabled: bool = False
    shadow_lookbacks: tuple[int, ...] = (7, 21, 63, 126)


class TrendStrategy(Strategy):
    def __init__(self, config: TrendStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.venue = self.instrument_id.venue
        self._closes: deque[float] = deque(maxlen=max(config.lookbacks) + 1)
        self.instrument = None
        self._portfolio_iids = tuple(
            InstrumentId.from_str(i)
            for i in (config.portfolio_instrument_ids or (config.instrument_id,))
        )
        self.risk_state = None  # optional shared control state (injected by the runner)
        self.funding_rates = None  # optional shared FundingRateState (injected by the runner)
        self._dynamic_stop_pct = config.stop_pct
        initial_daily_vol = config.vol_floor / np.sqrt(config.ppy)
        self._dynamic_trailing_pct = self._bounded_trailing_pct(
            config.trailing_vol_mult * initial_daily_vol
        )
        self._pending_entry_order = None
        self._protective_orders = []  # SL/TP/trailing orders we've submitted (may be in-flight)
        self._protection_signature = None  # desired protection currently working/in-flight
        self._protection_position_key = None  # freezes +1R/trail width for this position
        self._protected_once = (
            False  # force a protection refresh on first rebalance (restart cleanup)
        )
        self._last_flatten_seq = 0  # one-shot flatten-request bookkeeping

    def cfg(self) -> TrendStrategyConfig:
        return self.config

    def on_start(self) -> None:
        self._sync_flatten_cursor()
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"No instrument {self.instrument_id}; stopping.")
            self.stop()
            return
        self._start_after_instrument_load()

    def _sync_flatten_cursor(self) -> None:
        # One-shot flatten commands already persisted before this process
        # started have been consumed. Start at the shared cursor so a restart
        # does not replay an old command and skip protection reconciliation.
        if self.risk_state is not None:
            self._last_flatten_seq = self.risk_state.flatten_seq

    def _start_after_instrument_load(self) -> None:
        if self.cfg().warmup_request:
            import pandas as pd

            try:
                start = self.clock.utc_now() - pd.Timedelta(days=max(self.cfg().lookbacks) + 10)
                self.request_bars(self.bar_type, start=start)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"request_bars warm-up skipped: {exc!r}")
            self.clock.set_timer(
                name="init_rebalance", interval=timedelta(seconds=20), callback=self._on_init_timer
            )
            # Live only: react to the kill-switch within a minute (not just on the
            # next daily bar). A 60s timer in backtest sim-time would fire millions
            # of times, so it's gated to live (warmup_request).
            self.clock.set_timer(
                name="control_check",
                interval=timedelta(seconds=60),
                callback=self._on_control_timer,
            )
        self.subscribe_bars(self.bar_type)

    def _on_init_timer(self, event) -> None:  # noqa: ANN001
        self.clock.cancel_timer("init_rebalance")
        self.log.info(f"initial rebalance ({len(self._closes)} bars buffered) {self.instrument_id}")
        run_id = getattr(self.risk_state, "run_id", "standalone")
        self._rebalance(
            cycle_id=f"{run_id}:startup",
            cycle_ts=self.clock.utc_now().isoformat(),
        )

    def on_historical_data(self, data) -> None:
        bars = data if isinstance(data, (list, tuple)) else [data]
        for b in bars:
            close = getattr(b, "close", None)
            if close is not None:
                self._closes.append(close.as_double())

    def on_bar(self, bar: Bar) -> None:
        self._closes.append(bar.close.as_double())
        cycle_ts = datetime.fromtimestamp(bar.ts_event / 1e9, tz=timezone.utc).isoformat()
        self._rebalance(cycle_id=str(bar.ts_event), cycle_ts=cycle_ts)

    def _flatten_self(self) -> None:
        """Flatten this strategy's own position (correctly strategy-scoped)."""
        if self.portfolio.net_position(self.instrument_id) != 0:
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)

    def _apply_control(self) -> bool:
        """Handle remote control (mode + one-shot flatten). Returns True if it
        took over this cycle (kill/halt/just-flattened) — caller should not trade."""
        rs = self.risk_state
        if rs is None:
            return False
        # one-shot flatten request (acts once per new seq, in any mode)
        if rs.flatten_seq > self._last_flatten_seq:
            target_me = rs.flatten_target in ("all", self.instrument_id.symbol.value)
            self._last_flatten_seq = rs.flatten_seq
            if target_me:
                self._flatten_self()
                return True
        if rs.mode == "kill":
            self._flatten_self()  # flatten + stay out
            return True
        if rs.market_data_stale:
            # A patient limit submitted before the stream failed must not fill (or
            # fall back to market) while the watchdog is blocking new exposure.
            order = self._pending_entry_order
            if order is not None and not getattr(order, "is_closed", False):
                with contextlib.suppress(Exception):
                    self.cancel_order(order)
            self._pending_entry_order = None
            return True
        if rs.mode == "halt":
            return True  # freeze: keep positions, no new orders
        return False

    def _on_control_timer(self, event) -> None:  # noqa: ANN001 (live: act on control within ~1 min)
        self._apply_control()

    def _rebalance(
        self,
        cycle_id: str | int | None = None,
        cycle_ts: str | None = None,
    ) -> None:
        if len(self._closes) <= max(self.cfg().lookbacks):
            return
        if self._apply_control():
            return

        closes = np.array(self._closes, dtype=float)
        price = closes[-1]
        sig = float(
            np.mean([np.sign(price / closes[-1 - lb] - 1.0) for lb in self.cfg().lookbacks])
        )
        rets = np.diff(closes) / closes[:-1]
        vw = min(self.cfg().vol_window, len(rets))
        inst_vol = max(float(np.std(rets[-vw:])) * np.sqrt(self.cfg().ppy), self.cfg().vol_floor)
        self._update_dynamic_stop(inst_vol)
        self._update_portfolio_returns(rets)

        equity = self._equity()
        if equity <= 0:
            return
        budget = equity / self.cfg().n_instruments
        raw_weight = float(
            np.clip(
                sig * (self.cfg().target_vol / inst_vol),
                -self.cfg().max_leverage,
                self.cfg().max_leverage,
            )
        )
        cycle_id = cycle_id or f"{getattr(self.risk_state, 'run_id', 'standalone')}:adhoc"
        cycle_ts = cycle_ts or datetime.now(timezone.utc).isoformat()
        if self.cfg().use_portfolio_allocator:
            funded_weight = self._funding_adjusted_weight(raw_weight)
            portfolio_scale = self._portfolio_scale(
                "live_allocator", funded_weight, rets, cycle_id
            )
            w = funded_weight * portfolio_scale
        else:
            portfolio_scale = 1.0
            w = raw_weight * self._correlation_scalar()
            w = self._funding_adjusted_weight(w)
        self._queue_shadow_targets(
            closes=closes,
            returns=rets,
            cycle_id=cycle_id,
            cycle_ts=cycle_ts,
            price=price,
            inst_vol=inst_vol,
            signal=sig,
            raw_weight=raw_weight,
            live_weight=w,
            live_portfolio_scale=portfolio_scale,
        )
        target_units = (budget * w) / price
        current_units = float(self.portfolio.net_position(self.instrument_id))
        delta = target_units - current_units

        traded = False
        threshold_notional = self._rebalance_notional_threshold(budget, price, inst_vol)
        if abs(delta * price) >= threshold_notional:
            qty = self.instrument.make_qty(abs(delta))
            if qty > self.instrument.make_qty(0) and self._order_passes_filters(qty, price):
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                self._submit_entry_order(side, qty, price)
                self.log.info(
                    f"{self.instrument_id.symbol} sig={sig:+.2f} vol={inst_vol:.2f} w={w:+.2f} "
                    f"tgt={target_units:+.4f} cur={current_units:+.4f} {side.name} {qty}"
                )
                traded = True

        # On fills/position events, protection is refreshed from the actual open
        # position. On restart or a quiet rebalance, still reconcile once so stale
        # stops from a prior run get corrected.
        if not traded or not self._protected_once:
            self._refresh_protection(entry_fallback=price)
            self._protected_once = True

    # --- protective orders -------------------------------------------------
    def on_order_filled(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            pending = self._pending_entry_order
            if pending is not None and (
                getattr(event, "client_order_id", None) == pending.client_order_id
            ):
                self._pending_entry_order = None
            # Portfolio state is updated before strategy fill dispatch. Reconcile
            # again here because Binance netting may emit no PositionChanged event
            # for a restart-reconciled position. _set_protection is idempotent and
            # can cancel an in-flight stale stop, so overlapping position callbacks
            # cannot stack duplicate protection.
            self._refresh_protection()

    def on_order_canceled(self, event) -> None:  # noqa: ANN001
        if (
            self._pending_entry_order is not None
            and getattr(event, "client_order_id", None) == self._pending_entry_order.client_order_id
        ):
            self._pending_entry_order = None

    def on_order_rejected(self, event) -> None:  # noqa: ANN001
        client_order_id = getattr(event, "client_order_id", None)
        if any(o.client_order_id == client_order_id for o in self._protective_orders):
            # Keep any accepted hard stop working, but force the next reconciliation
            # to replace an incomplete protection set rather than treating it as valid.
            self._protection_signature = None
            self.log.error(
                f"protective order rejected for {self.instrument_id}: "
                f"{getattr(event, 'reason', 'unknown reason')}"
            )

    def on_position_opened(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            self._refresh_protection()

    def on_position_changed(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            self._refresh_protection()

    def on_position_closed(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            self._cancel_protective_orders()  # drop now-orphaned SL/TP/trail
            self._protection_signature = None
            self._protection_position_key = None

    def _refresh_protection(self, entry_fallback: float | None = None) -> None:
        """Reconcile reduce-only protection to the real open position."""
        units = float(self.portfolio.net_position(self.instrument_id))
        if units == 0:
            if (
                self.cfg().use_stop_loss
                or self.cfg().use_take_profit
                or self.cfg().use_trailing_stop
            ):
                self._cancel_protective_orders()
                self._protection_signature = None
                self._protection_position_key = None
            return
        entry = entry_fallback
        open_pos = self.cache.positions_open(instrument_id=self.instrument_id)
        if open_pos:
            with contextlib.suppress(Exception):
                entry = float(open_pos[0].avg_px_open)
        if entry is None:
            return
        self._set_protection(units, entry)

    def _update_dynamic_stop(self, inst_vol: float) -> None:
        daily_vol = inst_vol / np.sqrt(self.cfg().ppy)
        if not self.cfg().use_vol_stop:
            self._dynamic_stop_pct = self.cfg().stop_pct
        else:
            self._dynamic_stop_pct = float(
                np.clip(
                    self.cfg().stop_vol_mult * daily_vol,
                    self.cfg().min_stop_pct,
                    self.cfg().max_stop_pct,
                )
            )
        self._dynamic_trailing_pct = self._bounded_trailing_pct(
            self.cfg().trailing_vol_mult * daily_vol
        )

    def _bounded_trailing_pct(self, value: float) -> float:
        # Binance Futures callbackRate is 0.1%-10%. Keep configured bounds inside
        # venue limits so a typo cannot leave the trailing protection rejected.
        venue_min, venue_max = 0.001, 0.10
        lower = float(np.clip(self.cfg().min_trailing_pct, venue_min, venue_max))
        upper = float(np.clip(self.cfg().max_trailing_pct, lower, venue_max))
        return float(np.clip(value, lower, upper))

    def _update_portfolio_returns(self, returns: np.ndarray) -> None:
        enabled = (
            self.cfg().use_correlation_scaling
            or self.cfg().use_portfolio_allocator
            or self.cfg().shadow_enabled
        )
        if not (enabled and self.risk_state is not None):
            return
        window = min(
            len(returns),
            max(self.cfg().corr_window, self.cfg().portfolio_cov_window),
        )
        if window > 1 and hasattr(self.risk_state, "update_returns"):
            self.risk_state.update_returns(self.instrument_id.symbol.value, returns[-window:])

    def _portfolio_scale(
        self,
        model: str,
        weight: float,
        returns: np.ndarray,
        cycle_id: str | int,
    ) -> float:
        rs = self.risk_state
        if rs is None or not hasattr(rs, "publish_portfolio_target"):
            return 1.0
        expected = tuple(iid.symbol.value for iid in self._portfolio_iids)
        return float(
            rs.publish_portfolio_target(
                model,
                self.instrument_id.symbol.value,
                weight,
                returns,
                cycle_id,
                expected,
                window=self.cfg().portfolio_cov_window,
                target_vol=self.cfg().portfolio_target_vol,
                ppy=self.cfg().ppy,
                shrinkage=self.cfg().portfolio_cov_shrinkage,
                min_scale=self.cfg().portfolio_min_scale,
                max_scale=self.cfg().portfolio_max_scale,
                max_gross=self.cfg().portfolio_max_gross,
                min_observations=self.cfg().portfolio_min_observations,
                scale_up_alpha=self.cfg().portfolio_scale_up_alpha,
                scale_deadband=self.cfg().portfolio_scale_deadband,
            )
        )

    def _correlation_scalar(self) -> float:
        if not (self.cfg().use_correlation_scaling and self.risk_state is not None):
            return 1.0
        if not hasattr(self.risk_state, "correlation_scalar"):
            return 1.0
        return float(
            self.risk_state.correlation_scalar(
                self.cfg().corr_window,
                self.cfg().corr_threshold,
                self.cfg().corr_min_scalar,
            )
        )

    def _funding_adjusted_weight(
        self,
        weight: float,
        *,
        continuous: bool | None = None,
        log_adjustment: bool = True,
    ) -> float:
        enabled = self.cfg().funding_filter_enabled or continuous is True
        if not enabled or weight == 0 or self.funding_rates is None:
            return weight
        rate = self.funding_rates.get(self.instrument_id.symbol.value)
        if rate is None:
            return weight
        use_continuous = (
            self.cfg().funding_continuous_sizing if continuous is None else continuous
        )
        scalar = self._funding_scalar(weight, rate, continuous=use_continuous)
        if scalar < 1.0 and log_adjustment:
            action = "scaled" if use_continuous and scalar > 0 else "veto"
            self.log.info(
                f"{self.instrument_id.symbol} funding {action}: target_w={weight:+.2f} "
                f"rate={rate:+.4%} scalar={scalar:.2f}"
            )
        return weight * scalar

    def _funding_scalar(self, weight: float, rate: float, *, continuous: bool) -> float:
        if weight == 0:
            return 1.0
        adverse_rate = float(rate) if weight > 0 else -float(rate)
        if adverse_rate <= 0:
            return 1.0  # favorable funding never increases risk
        hard = max(0.0, float(self.cfg().funding_rate_limit))
        if not continuous:
            return 0.0 if adverse_rate > hard else 1.0
        soft = float(np.clip(self.cfg().funding_rate_soft_limit, 0.0, hard))
        floor = float(np.clip(self.cfg().funding_min_scalar, 0.0, 1.0))
        if adverse_rate <= soft:
            return 1.0
        if hard <= soft or adverse_rate >= hard:
            return floor
        pressure = (adverse_rate - soft) / (hard - soft)
        return float(1.0 - pressure * (1.0 - floor))

    def _queue_shadow_targets(
        self,
        *,
        closes: np.ndarray,
        returns: np.ndarray,
        cycle_id: str | int,
        cycle_ts: str,
        price: float,
        inst_vol: float,
        signal: float,
        raw_weight: float,
        live_weight: float,
        live_portfolio_scale: float,
    ) -> None:
        rs = self.risk_state
        if not self.cfg().shadow_enabled or rs is None or not hasattr(rs, "queue_shadow_target"):
            return
        rate = self.funding_rates.get(self.instrument_id.symbol.value) if self.funding_rates else None
        self._queue_shadow_row(
            "live",
            cycle_id,
            cycle_ts,
            price,
            signal,
            raw_weight,
            live_weight,
            live_portfolio_scale,
            rate,
        )

        smooth_weight = self._funding_adjusted_weight(
            raw_weight, continuous=True, log_adjustment=False
        )
        smooth_scale = self._portfolio_scale(
            "shadow_allocator_continuous", smooth_weight, returns, cycle_id
        )
        self._queue_shadow_row(
            "allocator_continuous",
            cycle_id,
            cycle_ts,
            price,
            signal,
            raw_weight,
            smooth_weight * smooth_scale,
            smooth_scale,
            rate,
        )

        lookbacks = tuple(self.cfg().shadow_lookbacks)
        if lookbacks and len(closes) > max(lookbacks):
            fast_signal = float(
                np.mean([np.sign(price / closes[-1 - lb] - 1.0) for lb in lookbacks])
            )
            fast_raw = float(
                np.clip(
                    fast_signal * (self.cfg().target_vol / inst_vol),
                    -self.cfg().max_leverage,
                    self.cfg().max_leverage,
                )
            )
            fast_funded = self._funding_adjusted_weight(
                fast_raw, continuous=True, log_adjustment=False
            )
            fast_scale = self._portfolio_scale(
                "shadow_fast_allocator_continuous", fast_funded, returns, cycle_id
            )
            self._queue_shadow_row(
                "fast_allocator_continuous",
                cycle_id,
                cycle_ts,
                price,
                fast_signal,
                fast_raw,
                fast_funded * fast_scale,
                fast_scale,
                rate,
            )

    def _queue_shadow_row(
        self,
        model: str,
        cycle_id: str | int,
        cycle_ts: str,
        price: float,
        signal: float,
        raw_weight: float,
        target_weight: float,
        portfolio_scale: float,
        funding_rate: float | None,
    ) -> None:
        continuous = model != "live" or self.cfg().funding_continuous_sizing
        funding_enabled = self.cfg().funding_filter_enabled or model != "live"
        funding_scalar = (
            self._funding_scalar(raw_weight, funding_rate, continuous=continuous)
            if funding_enabled and funding_rate is not None
            else 1.0
        )
        self.risk_state.queue_shadow_target(
            {
                "model": model,
                "cycle_id": str(cycle_id),
                "cycle_ts": cycle_ts,
                "symbol": self.instrument_id.symbol.value,
                "price": float(price),
                "signal": float(signal),
                "raw_weight": float(raw_weight),
                "target_weight": float(target_weight),
                "portfolio_weight": float(target_weight) / max(1, self.cfg().n_instruments),
                "portfolio_scale": float(portfolio_scale),
                "funding_rate": float(funding_rate) if funding_rate is not None else None,
                "funding_scalar": float(funding_scalar),
                "cost_bps": float(self.cfg().round_trip_cost_bps + self.cfg().slippage_bps),
            }
        )

    def _rebalance_notional_threshold(self, budget: float, price: float, inst_vol: float) -> float:
        threshold = self.cfg().rebalance_threshold * budget
        min_notional = self._min_order_notional(price)
        if min_notional:
            threshold = max(threshold, min_notional * self.cfg().min_notional_buffer)
        if self.cfg().cost_aware_rebalance:
            total_bps = self.cfg().round_trip_cost_bps + self.cfg().slippage_bps
            threshold = max(threshold, budget * (total_bps / 1e4) * self.cfg().cost_threshold_mult)
            threshold = max(threshold, budget * (inst_vol / np.sqrt(self.cfg().ppy)) * 0.25)
        return threshold

    def _submit_entry_order(self, side: OrderSide, qty, price: float) -> None:  # noqa: ANN001
        reference_tags = [
            f"REF_PX={price:.12g}",
            f"RISK_PCT={self._dynamic_stop_pct:.12g}",
        ]
        if self.cfg().use_patient_limit:
            offset = self.cfg().patient_limit_offset_bps / 1e4
            limit_price = price * (1 - offset) if side == OrderSide.BUY else price * (1 + offset)
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
                price=self.instrument.make_price(limit_price),
                tags=reference_tags,
            )
            self._pending_entry_order = order
            self.submit_order(order)
            if self.cfg().patient_limit_timeout_secs > 0:
                self.clock.set_timer(
                    name=f"patient_timeout_{self.instrument_id.symbol.value}",
                    interval=timedelta(seconds=self.cfg().patient_limit_timeout_secs),
                    callback=self._on_patient_timeout,
                )
            return
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
                tags=reference_tags,
            )
        )

    def _on_patient_timeout(self, event) -> None:  # noqa: ANN001
        with contextlib.suppress(Exception):
            self.clock.cancel_timer(f"patient_timeout_{self.instrument_id.symbol.value}")
        order = self._pending_entry_order
        self._pending_entry_order = None
        if order is None or order not in self.cache.orders_open():
            return
        self.cancel_order(order)
        if self.cfg().patient_limit_market_fallback and not self._apply_control():
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=order.side,
                    quantity=order.quantity,
                    tags=order.tags,
                )
            )

    def _order_passes_filters(self, qty, price: float) -> bool:  # noqa: ANN001
        qty_value = float(qty)
        min_qty = self._as_float(getattr(self.instrument, "min_quantity", None))
        max_qty = self._as_float(getattr(self.instrument, "max_quantity", None))
        if min_qty and qty_value < min_qty:
            self.log.info(
                f"{self.instrument_id.symbol} skip: qty {qty_value:g} < min_qty {min_qty:g}"
            )
            return False
        if max_qty and qty_value > max_qty:
            self.log.warning(
                f"{self.instrument_id.symbol} skip: qty {qty_value:g} > max_qty {max_qty:g}"
            )
            return False
        min_notional = self._min_order_notional(price)
        notional = qty_value * price
        if min_notional and notional < min_notional:
            self.log.info(
                f"{self.instrument_id.symbol} skip: notional ${notional:.2f} < min_notional ${min_notional:.2f}"
            )
            return False
        return True

    def _min_order_notional(self, price: float) -> float:
        min_notional = self._as_float(getattr(self.instrument, "min_notional", None))
        if min_notional:
            return min_notional
        min_qty = self._as_float(getattr(self.instrument, "min_quantity", None))
        return min_qty * price if min_qty else 0.0

    @staticmethod
    def _as_float(value) -> float:  # noqa: ANN001
        if value is None:
            return 0.0
        if hasattr(value, "as_double"):
            return float(value.as_double())
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _has_live_protection(self) -> bool:
        expected = sum(
            (
                bool(self.cfg().use_stop_loss),
                bool(self.cfg().use_take_profit),
                bool(self.cfg().use_trailing_stop),
            )
        )
        live = sum(not getattr(o, "is_closed", False) for o in self._protective_orders)
        return live == expected

    def _cancel_protective_orders(self) -> None:
        """Cancel every protective order we own, INCLUDING in-flight ones, plus any
        stray reduce-only orders in the cache (left by a prior run / reconciliation).

        cancel_order works on submitted-but-not-yet-accepted orders; cancel_all_orders
        only targets already-open ones, which is why a racy cancel-then-resubmit
        stacked duplicates."""
        seen = set()
        for order in list(self._protective_orders):
            if not getattr(order, "is_closed", False):
                with contextlib.suppress(Exception):
                    self.cancel_order(order)
            seen.add(order.client_order_id)
        self._protective_orders.clear()
        for order in self.cache.orders_open(instrument_id=self.instrument_id):
            if order.client_order_id in seen or not getattr(order, "is_reduce_only", False):
                continue
            with contextlib.suppress(Exception):
                self.cancel_order(order)

    def _set_protection(self, position_units: float, entry_price: float) -> None:
        """Idempotently keep a hard SL plus optional TP/native trail on the position."""
        if not (
            self.cfg().use_stop_loss
            or self.cfg().use_take_profit
            or self.cfg().use_trailing_stop
        ):
            return
        if position_units == 0:
            self._cancel_protective_orders()
            self._protection_signature = None
            self._protection_position_key = None
            return
        is_long = position_units > 0
        close_side = OrderSide.SELL if is_long else OrderSide.BUY  # close = opposite side
        qty = self.instrument.make_qty(abs(position_units))
        if qty <= self.instrument.make_qty(0):
            return
        position_key = (is_long, str(qty), str(self.instrument.make_price(entry_price)))
        if (
            self.cfg().use_trailing_stop
            and position_key == self._protection_position_key
            and self._has_live_protection()
        ):
            # Native trailing orders remember their best favorable price. Do not
            # cancel/recreate them on each daily vol update, which would reset and
            # potentially loosen the exchange-side watermark. R and trail width are
            # therefore fixed when this position/size is first protected.
            return
        sl_price = None
        if self.cfg().use_stop_loss:
            # long: stop BELOW; short: stop ABOVE
            sl = (
                entry_price * (1 - self._dynamic_stop_pct)
                if is_long
                else entry_price * (1 + self._dynamic_stop_pct)
            )
            sl_price = self.instrument.make_price(sl)
        tp_price = None
        if self.cfg().use_take_profit:
            tp = (
                entry_price * (1 + self.cfg().tp_pct)
                if is_long
                else entry_price * (1 - self.cfg().tp_pct)
            )
            tp_price = self.instrument.make_price(tp)

        trail_activation = None
        trail_offset_bps = None
        if self.cfg().use_trailing_stop:
            activation_distance = max(0.0, self.cfg().trailing_activation_r) * (
                entry_price * self._dynamic_stop_pct
            )
            activation = (
                entry_price + activation_distance if is_long else entry_price - activation_distance
            )
            trail_activation = self.instrument.make_price(activation)

            # If price already passed +1R, omitting activation_price tells Binance
            # to arm from the current mark and avoids "Order would immediately trigger"
            # on restart or reconciliation.
            mark = None
            if self.risk_state is not None:
                mark = getattr(self.risk_state, "mark_prices", {}).get(str(self.instrument_id))
            already_activated = mark is not None and (
                (is_long and mark >= activation) or (not is_long and mark <= activation)
            )
            if already_activated:
                trail_activation = None
            # Binance Futures accepts callbackRate in percent with one decimal
            # place. Nautilus represents this as basis points and divides by
            # 100, so round to 10-bps increments before submission.
            callback_rate_pct = round(self._dynamic_trailing_pct * 100, 1)
            trail_offset_bps = Decimal(str(callback_rate_pct * 100))

        signature = (
            close_side.value,
            str(qty),
            str(sl_price) if sl_price is not None else None,
            str(tp_price) if tp_price is not None else None,
            (
                str(trail_activation)
                if trail_activation is not None
                else ("IMMEDIATE" if trail_offset_bps is not None else None)
            ),
            str(trail_offset_bps) if trail_offset_bps is not None else None,
        )
        if signature == self._protection_signature and self._has_live_protection():
            return  # correct protection already working/in-flight — don't duplicate

        self._cancel_protective_orders()
        try:
            if sl_price is not None:
                order = self.order_factory.stop_market(
                    instrument_id=self.instrument_id,
                    order_side=close_side,
                    quantity=qty,
                    trigger_price=sl_price,
                    reduce_only=True,
                )
                self._protective_orders.append(order)
                self.submit_order(order)
            if tp_price is not None:
                order = self.order_factory.limit(
                    instrument_id=self.instrument_id,
                    order_side=close_side,
                    quantity=qty,
                    price=tp_price,
                    reduce_only=True,
                )
                self._protective_orders.append(order)
                self.submit_order(order)
            if trail_offset_bps is not None:
                order = self.order_factory.trailing_stop_market(
                    instrument_id=self.instrument_id,
                    order_side=close_side,
                    quantity=qty,
                    trailing_offset=trail_offset_bps,
                    trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
                    activation_price=trail_activation,
                    trigger_type=TriggerType.MARK_PRICE,
                    reduce_only=True,
                    tags=["VOL_TRAIL"],
                )
                self._protective_orders.append(order)
                self.submit_order(order)
            self._protection_signature = signature
            self._protection_position_key = position_key
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"protection placement failed for {self.instrument_id}: {exc!r}")
            self._protection_signature = None
            self._protection_position_key = None

    def _equity(self) -> float:
        account = self.portfolio.account(self.venue)
        if account is None:
            return 0.0
        ccy = getattr(account, "base_currency", None) or self.instrument.quote_currency
        bal = account.balance_total(ccy)
        equity = bal.as_double() if bal is not None else 0.0
        for iid in self._portfolio_iids:
            upnl = self.portfolio.unrealized_pnl(iid)
            if upnl is not None:
                equity += upnl.as_double()
        return equity

    def on_stop(self) -> None:
        if self.cfg().flatten_on_stop:
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)
