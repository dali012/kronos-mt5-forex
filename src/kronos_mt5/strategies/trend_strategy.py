"""TrendStrategy — event-driven vol-targeted trend-following (NautilusTrader).

Engine port of the vectorized strategy in backtest/trend.py. One instance trades
one instrument; run N instances for the diversified portfolio.

Per closed bar: ensemble sign-momentum signal in [-1,1], vol-targeted to a per
-instrument risk budget, rebalanced toward the target via a market order.

Risk controls:
  - Exchange-native protective orders per position (reduce-only): a STOP_MARKET
    stop-loss and (optional) LIMIT take-profit, re-anchored on each position
    change. These live server-side, so they protect even if this process
    crashes / disconnects.
  - A shared `risk_state.halted` gate (set by the RiskManager drawdown
    kill-switch) blocks new entries.

Verified against NautilusTrader 1.228.0.
"""

from __future__ import annotations

import contextlib
from collections import deque
from datetime import timedelta

import numpy as np

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
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
    flatten_on_stop: bool = False
    # --- portfolio/friction controls ---
    use_correlation_scaling: bool = False
    corr_window: int = 90
    corr_threshold: float = 0.65
    corr_min_scalar: float = 0.50
    funding_filter_enabled: bool = False
    funding_rate_limit: float = 0.0010
    cost_aware_rebalance: bool = False
    round_trip_cost_bps: float = 8.0
    slippage_bps: float = 2.0
    cost_threshold_mult: float = 4.0
    min_notional_buffer: float = 1.05
    use_patient_limit: bool = False
    patient_limit_offset_bps: float = 2.0
    patient_limit_timeout_secs: int = 300
    patient_limit_market_fallback: bool = True


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
        self._pending_entry_order = None
        self._protected_once = (
            False  # force a protection refresh on first rebalance (restart cleanup)
        )
        self._last_flatten_seq = 0  # one-shot flatten-request bookkeeping

    def cfg(self) -> TrendStrategyConfig:
        return self.config

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"No instrument {self.instrument_id}; stopping.")
            self.stop()
            return
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
        self._rebalance()

    def on_historical_data(self, data) -> None:
        bars = data if isinstance(data, (list, tuple)) else [data]
        for b in bars:
            close = getattr(b, "close", None)
            if close is not None:
                self._closes.append(close.as_double())

    def on_bar(self, bar: Bar) -> None:
        self._closes.append(bar.close.as_double())
        self._rebalance()

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
        if rs.mode == "halt":
            return True  # freeze: keep positions, no new orders
        return False

    def _on_control_timer(self, event) -> None:  # noqa: ANN001 (live: act on control within ~1 min)
        self._apply_control()

    def _rebalance(self) -> None:
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
        w = float(
            np.clip(
                sig * (self.cfg().target_vol / inst_vol),
                -self.cfg().max_leverage,
                self.cfg().max_leverage,
            )
        )
        w *= self._correlation_scalar()
        w = self._funding_adjusted_weight(w)
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
            self._pending_entry_order = None
            self._refresh_protection()

    def on_order_canceled(self, event) -> None:  # noqa: ANN001
        if (
            self._pending_entry_order is not None
            and getattr(event, "client_order_id", None) == self._pending_entry_order.client_order_id
        ):
            self._pending_entry_order = None

    def on_position_opened(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            self._refresh_protection()

    def on_position_changed(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            self._refresh_protection()

    def on_position_closed(self, event) -> None:  # noqa: ANN001
        if getattr(event, "instrument_id", None) == self.instrument_id:
            self.cancel_all_orders(self.instrument_id)  # drop now-orphaned SL/TP

    def _refresh_protection(self, entry_fallback: float | None = None) -> None:
        """Reconcile reduce-only protection to the real open position."""
        units = float(self.portfolio.net_position(self.instrument_id))
        if units == 0:
            if self.cfg().use_stop_loss or self.cfg().use_take_profit:
                self.cancel_all_orders(self.instrument_id)
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
        if not self.cfg().use_vol_stop:
            self._dynamic_stop_pct = self.cfg().stop_pct
            return
        daily_vol = inst_vol / np.sqrt(self.cfg().ppy)
        self._dynamic_stop_pct = float(
            np.clip(
                self.cfg().stop_vol_mult * daily_vol,
                self.cfg().min_stop_pct,
                self.cfg().max_stop_pct,
            )
        )

    def _update_portfolio_returns(self, returns: np.ndarray) -> None:
        if not (self.cfg().use_correlation_scaling and self.risk_state is not None):
            return
        window = min(len(returns), self.cfg().corr_window)
        if window > 1 and hasattr(self.risk_state, "update_returns"):
            self.risk_state.update_returns(self.instrument_id.symbol.value, returns[-window:])

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

    def _funding_adjusted_weight(self, weight: float) -> float:
        if not self.cfg().funding_filter_enabled or weight == 0 or self.funding_rates is None:
            return weight
        rate = self.funding_rates.get(self.instrument_id.symbol.value)
        if rate is None:
            return weight
        adverse_long = weight > 0 and rate > self.cfg().funding_rate_limit
        adverse_short = weight < 0 and rate < -self.cfg().funding_rate_limit
        if adverse_long or adverse_short:
            self.log.info(
                f"{self.instrument_id.symbol} funding veto: target_w={weight:+.2f} "
                f"rate={rate:+.4%} limit={self.cfg().funding_rate_limit:.4%}"
            )
            return 0.0
        return weight

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
        if self.cfg().use_patient_limit:
            offset = self.cfg().patient_limit_offset_bps / 1e4
            limit_price = price * (1 - offset) if side == OrderSide.BUY else price * (1 + offset)
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
                price=self.instrument.make_price(limit_price),
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
                instrument_id=self.instrument_id, order_side=side, quantity=qty
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
        if self.cfg().patient_limit_market_fallback:
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=order.side,
                    quantity=order.quantity,
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

    def _set_protection(self, position_units: float, entry_price: float) -> None:
        """(Re)place reduce-only SL/TP once per rebalance, sized & sided to the
        actual open position. Driven by signed net position (not pos.quantity,
        which is absolute), so stops don't accumulate or mismatch after fills."""
        if not (self.cfg().use_stop_loss or self.cfg().use_take_profit):
            return
        self.cancel_all_orders(self.instrument_id)  # clear previous protection
        if position_units == 0:
            return
        is_long = position_units > 0
        close_side = OrderSide.SELL if is_long else OrderSide.BUY  # close = opposite side
        qty = self.instrument.make_qty(abs(position_units))
        if qty <= self.instrument.make_qty(0):
            return
        try:
            if self.cfg().use_stop_loss:
                # long: stop BELOW; short: stop ABOVE
                sl = (
                    entry_price * (1 - self._dynamic_stop_pct)
                    if is_long
                    else entry_price * (1 + self._dynamic_stop_pct)
                )
                self.submit_order(
                    self.order_factory.stop_market(
                        instrument_id=self.instrument_id,
                        order_side=close_side,
                        quantity=qty,
                        trigger_price=self.instrument.make_price(sl),
                        reduce_only=True,
                    )
                )
            if self.cfg().use_take_profit:
                tp = (
                    entry_price * (1 + self.cfg().tp_pct)
                    if is_long
                    else entry_price * (1 - self.cfg().tp_pct)
                )
                self.submit_order(
                    self.order_factory.limit(
                        instrument_id=self.instrument_id,
                        order_side=close_side,
                        quantity=qty,
                        price=self.instrument.make_price(tp),
                        reduce_only=True,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"protection placement failed for {self.instrument_id}: {exc!r}")

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
