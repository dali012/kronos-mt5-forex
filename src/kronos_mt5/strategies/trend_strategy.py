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

from kronos_mt5.execution import (
    ROLE_HARD_STOP,
    ROLE_PATIENT_FALLBACK,
    ROLE_PATIENT_LIMIT,
    ROLE_TAKE_PROFIT,
    ROLE_TRAILING_STOP,
    ROLE_TREND_MARKET,
    STATE_WORKING,
    TAG_DECISION_TS,
    TAG_EXEC_ROLE,
    TAG_FALLBACK_DRIFT,
    TAG_FALLBACK_REASON,
    TAG_LIMIT_PX,
    TAG_REF_PX,
    TAG_RISK_PCT,
    PatientOrder,
    evaluate_fallback,
)

# A fallback must price off something current. Marks arrive ~1/s live, so a
# reference older than this is treated as missing rather than executable — the
# bar close that produced the decision is never reused as a fill price.
FRESH_PRICE_MAX_AGE_SECS = 60.0


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
    # Post-only rejects a patient limit that would cross and take liquidity.
    # Default False keeps existing deployments byte-identical; `true` is the
    # recommended rollout value once maker fills are confirmed.
    patient_limit_post_only: bool = False
    # Skip the market fallback when the market has already run this far against
    # the decision price (bps). <= 0 disables the cap.
    patient_limit_max_adverse_bps: float = 15.0
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
        # Explicit patient-limit lifecycle: `_patient` is the state, `_patient_order`
        # the engine object we may still need to cancel. Both are cleared together.
        self._patient: PatientOrder | None = None
        self._patient_order = None
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
                self._abort_patient("control_flatten")
                self._flatten_self()
                return True
        if rs.mode == "kill":
            self._abort_patient("control_kill")
            self._flatten_self()  # flatten + stay out
            return True
        if rs.market_data_stale:
            # A patient limit submitted before the stream failed must not fill (or
            # fall back to market) while the watchdog is blocking new exposure.
            self._abort_patient("market_data_stale")
            return True
        if rs.mode == "halt":
            # Freeze: keep positions and any resting limit, but never fall back to
            # market — `_control_block_reason` blocks that at resolution time.
            return True
        return False

    def _control_block_reason(self) -> str | None:
        """Side-effect-free view of the control gate, for the fallback decision."""
        rs = self.risk_state
        if rs is None:
            return None
        if getattr(rs, "flatten_seq", 0) > self._last_flatten_seq:
            return "control_flatten"
        if rs.mode == "kill":
            return "control_kill"
        if rs.mode == "halt":
            return "control_halt"
        if getattr(rs, "market_data_stale", False):
            return "market_data_stale"
        return None

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
                self._submit_entry_order(side, qty, price, target_units=target_units)
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
            patient = self._patient
            if patient is not None and (
                getattr(event, "client_order_id", None) == patient.client_order_id
            ):
                patient.register_fill(self._as_float(getattr(event, "last_qty", 0.0)))
                order = self._patient_order
                closed = True if order is None else bool(getattr(order, "is_closed", False))
                if closed:
                    # Terminal by fill. If a cancel was pending, the fallback is
                    # re-evaluated against the *new* position, so a completed fill
                    # simply leaves nothing to do.
                    self._patient_terminal("filled")
                else:
                    self.log.info(
                        f"patient partial fill {patient.symbol} "
                        f"coid={patient.client_order_id} filled={patient.filled_units:.8f}/"
                        f"{patient.submitted_units:.8f} state={patient.state}"
                    )
            # Portfolio state is updated before strategy fill dispatch. Reconcile
            # again here because Binance netting may emit no PositionChanged event
            # for a restart-reconciled position. _set_protection is idempotent and
            # can cancel an in-flight stale stop, so overlapping position callbacks
            # cannot stack duplicate protection.
            self._refresh_protection()

    def on_order_canceled(self, event) -> None:  # noqa: ANN001
        self._patient_terminal("canceled", client_order_id=getattr(event, "client_order_id", None))

    def on_order_expired(self, event) -> None:
        self._patient_terminal("expired", client_order_id=getattr(event, "client_order_id", None))

    def on_order_denied(self, event) -> None:
        self._patient_terminal("denied", client_order_id=getattr(event, "client_order_id", None))

    def on_order_cancel_rejected(self, event) -> None:
        """The venue refused the cancel — the limit may still fill, so never
        market-fallback for it. Retry the cancel on the next timeout instead."""
        patient = self._patient
        client_order_id = getattr(event, "client_order_id", None)
        if patient is None or client_order_id != patient.client_order_id:
            return
        order = self._patient_order
        if order is None or getattr(order, "is_closed", False):
            self._patient_terminal("cancel_rejected_closed")
            return
        patient.fallback_forbidden = True
        patient.fallback_pending = False
        patient.state = STATE_WORKING
        patient.note("cancel_rejected")
        self.log.error(
            f"patient cancel rejected {patient.symbol} coid={patient.client_order_id} "
            f"attempts={patient.cancel_attempts} reason="
            f"{getattr(event, 'reason', 'unknown reason')} — fallback disabled, retrying cancel"
        )
        self._set_patient_timer()

    def on_order_rejected(self, event) -> None:  # noqa: ANN001
        client_order_id = getattr(event, "client_order_id", None)
        patient = self._patient
        if patient is not None and client_order_id == patient.client_order_id:
            # Includes a post-only limit rejected for crossing the book. Clear the
            # state and let the next rebalance cycle decide — never resubmit here.
            self.log.warning(
                f"patient limit rejected {patient.symbol} coid={patient.client_order_id} "
                f"post_only={patient.post_only} "
                f"reason={getattr(event, 'reason', 'unknown reason')}"
            )
            self._patient_terminal("rejected")
            return
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

    def _submit_entry_order(
        self,
        side: OrderSide,
        qty,
        price: float,
        target_units: float,
    ) -> None:
        decision_ts_ns = self._now_ns()
        reference_tags = [
            f"{TAG_REF_PX}{price:.12g}",
            f"{TAG_RISK_PCT}{self._dynamic_stop_pct:.12g}",
            f"{TAG_DECISION_TS}{decision_ts_ns}",
        ]
        if self.cfg().use_patient_limit:
            # Only ever one patient order per instrument; cancel any predecessor so
            # a stale resting limit cannot fill on top of the new one.
            # Sweep first (so the order we still track is excluded), then cancel it.
            self._cancel_orphan_entry_limits()
            self._abort_patient("superseded")
            offset = self.cfg().patient_limit_offset_bps / 1e4
            raw_limit = price * (1 - offset) if side == OrderSide.BUY else price * (1 + offset)
            limit_price = self.instrument.make_price(raw_limit)
            post_only = bool(self.cfg().patient_limit_post_only)
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
                price=limit_price,
                post_only=post_only,
                tags=[
                    *reference_tags,
                    f"{TAG_EXEC_ROLE}{ROLE_PATIENT_LIMIT}",
                    f"{TAG_LIMIT_PX}{float(limit_price):.12g}",
                ],
            )
            self._patient = PatientOrder(
                client_order_id=order.client_order_id,
                symbol=self.instrument_id.symbol.value,
                side=side.name,
                target_units=float(target_units),
                decision_price=float(price),
                decision_ts_ns=decision_ts_ns,
                limit_price=float(limit_price),
                submitted_units=self._as_float(qty),
                post_only=post_only,
            )
            self._patient_order = order
            self.submit_order(order)
            self.log.info(
                f"patient limit submitted {self._patient.symbol} "
                f"coid={order.client_order_id} {side.name} {qty} @ {limit_price} "
                f"post_only={post_only} ref={price:.12g} "
                f"target={target_units:+.8f} timeout={self.cfg().patient_limit_timeout_secs}s"
            )
            self._set_patient_timer()
            return
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=qty,
                tags=[*reference_tags, f"{TAG_EXEC_ROLE}{ROLE_TREND_MARKET}"],
            )
        )

    # --- patient-limit lifecycle -------------------------------------------
    def _now_ns(self) -> int:
        with contextlib.suppress(Exception):
            return int(self.clock.timestamp_ns())
        return 0

    def _patient_timer_name(self) -> str:
        return f"patient_timeout_{self.instrument_id.symbol.value}"

    def _set_patient_timer(self) -> None:
        if self.cfg().patient_limit_timeout_secs <= 0:
            return
        self._cancel_patient_timer()
        with contextlib.suppress(Exception):
            self.clock.set_timer(
                name=self._patient_timer_name(),
                interval=timedelta(seconds=self.cfg().patient_limit_timeout_secs),
                callback=self._on_patient_timeout,
            )

    def _cancel_patient_timer(self) -> None:
        with contextlib.suppress(Exception):
            self.clock.cancel_timer(self._patient_timer_name())

    def _on_patient_timeout(self, event=None) -> None:
        """Timeout only *requests* cancellation. The fallback is decided later,
        once the venue has confirmed the limit is really closed."""
        self._cancel_patient_timer()
        patient = self._patient
        if patient is None:
            return
        order = self._patient_order
        if order is None or getattr(order, "is_closed", False):
            self._clear_patient("timeout_after_close")
            return
        if not patient.begin_cancel(
            fallback_enabled=bool(self.cfg().patient_limit_market_fallback)
        ):
            return  # already cancelling — a repeated timer is a no-op
        self.log.info(
            f"patient timeout {patient.symbol} coid={patient.client_order_id} "
            f"target={patient.target_units:+.8f} filled={patient.filled_units:.8f} "
            f"attempt={patient.cancel_attempts} -> cancel requested "
            f"(fallback_pending={patient.fallback_pending})"
        )
        with contextlib.suppress(Exception):
            self.cancel_order(order)

    def _patient_terminal(self, reason: str, *, client_order_id=None) -> None:
        """Authoritative end of the patient order. Idempotent: duplicate cancel/
        fill callbacks after the first are no-ops."""
        patient = self._patient
        if patient is None:
            return
        if client_order_id is not None and client_order_id != patient.client_order_id:
            return
        if not patient.mark_terminal(reason):
            return
        self._cancel_patient_timer()
        if patient.fallback_pending and not patient.fallback_resolved:
            self._resolve_fallback(patient, reason)
        self._clear_patient(reason)

    def _resolve_fallback(self, patient: PatientOrder, reason: str) -> None:
        """Recompute the remaining delta against the REAL position and, if every
        guard passes, take the remainder with a market order."""
        if patient.fallback_resolved:
            return
        current_units = float(self.portfolio.net_position(self.instrument_id))
        fresh_price = self._fresh_price()
        decision = evaluate_fallback(
            patient,
            current_units=current_units,
            fresh_price=fresh_price,
            fallback_enabled=bool(self.cfg().patient_limit_market_fallback),
            control_reason=self._control_block_reason(),
            max_adverse_bps=float(self.cfg().patient_limit_max_adverse_bps),
            min_units=self._min_units(),
        )
        patient.fallback_resolved = True
        patient.fallback_pending = False
        drift = "n/a" if decision.drift_bps is None else f"{decision.drift_bps:+.2f}bps"
        context = (
            f"{patient.symbol} coid={patient.client_order_id} after={reason} "
            f"target={patient.target_units:+.8f} current={current_units:+.8f} "
            f"remaining={decision.remaining_units:+.8f} ref={patient.decision_price:.12g} "
            f"fresh={'n/a' if fresh_price is None else format(fresh_price, '.12g')} drift={drift}"
        )
        if not decision.submit:
            self.log.info(f"patient fallback skipped: reason={decision.reason} {context}")
            return

        qty = self.instrument.make_qty(decision.units)
        if self._as_float(qty) <= 0 or not self._order_passes_filters(qty, decision.fresh_price):
            self.log.info(f"patient fallback skipped: reason=instrument_filters {context}")
            return
        side = OrderSide.BUY if decision.side == "BUY" else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            tags=[
                f"{TAG_REF_PX}{patient.decision_price:.12g}",
                f"{TAG_RISK_PCT}{self._dynamic_stop_pct:.12g}",
                f"{TAG_DECISION_TS}{patient.decision_ts_ns}",
                f"{TAG_EXEC_ROLE}{ROLE_PATIENT_FALLBACK}",
                f"{TAG_FALLBACK_REASON}{decision.reason}",
                f"{TAG_FALLBACK_DRIFT}{decision.drift_bps:.4f}",
            ],
        )
        self.submit_order(order)
        self.log.info(
            f"patient fallback submitted coid={order.client_order_id} {side.name} {qty} {context}"
        )

    def _cancel_orphan_entry_limits(self) -> None:
        """Cancel resting non-reduce-only limits this process no longer tracks.

        In-memory patient state does not survive a restart, so a limit left by a
        previous run would otherwise still be working at the venue and could fill
        on top of a freshly sized entry.
        """
        tracked = None
        if self._patient_order is not None:
            tracked = self._patient_order.client_order_id
        try:
            resting = list(self.cache.orders_open(instrument_id=self.instrument_id))
        except Exception:  # noqa: BLE001
            return
        for order in resting:
            if getattr(order, "is_reduce_only", False):
                continue
            if getattr(order.order_type, "name", None) != "LIMIT":
                continue
            if tracked is not None and order.client_order_id == tracked:
                continue
            self.log.warning(
                f"cancelling untracked resting entry limit {order.client_order_id} "
                f"on {self.instrument_id} (likely left by a previous run)"
            )
            with contextlib.suppress(Exception):
                self.cancel_order(order)

    def _abort_patient(self, reason: str) -> None:
        """Control/stop path: cancel the resting limit and never fall back."""
        patient = self._patient
        if patient is None:
            return
        patient.fallback_forbidden = True
        patient.fallback_pending = False
        order = self._patient_order
        if order is not None and not getattr(order, "is_closed", False):
            with contextlib.suppress(Exception):
                self.cancel_order(order)
        self._clear_patient(reason)

    def _clear_patient(self, reason: str) -> None:
        patient = self._patient
        self._patient = None
        self._patient_order = None
        self._cancel_patient_timer()
        if patient is not None:
            self.log.info(
                f"patient state cleared {patient.symbol} coid={patient.client_order_id} "
                f"reason={reason} filled={patient.filled_units:.8f}/"
                f"{patient.submitted_units:.8f}"
            )

    def _min_units(self) -> float:
        return self._as_float(getattr(self.instrument, "min_quantity", None))

    def _fresh_price(self) -> float | None:
        """Freshest executable reference available, or None.

        Never falls back to the bar close that produced the decision: reusing a
        stale daily close as an executable price is exactly what produced the
        ~412 bps fallback observed on testnet.
        """
        rs = self.risk_state
        key = str(self.instrument_id)
        if rs is not None:
            mark = getattr(rs, "mark_prices", {}).get(key)
            received_ns = getattr(rs, "mark_received_ns", {}).get(key)
            if mark and received_ns:
                age = max(0.0, (self._now_ns() - int(received_ns)) / 1e9)
                if age <= FRESH_PRICE_MAX_AGE_SECS:
                    return float(mark)
        tick_price = self._cached_tick_price()
        if tick_price is not None:
            return tick_price
        if rs is None or not getattr(rs, "mark_watchdog_enabled", False):
            # Backtest / no live mark stream: the latest bar close IS the freshest
            # price that exists, and fills are priced off bars anyway. Live runs
            # always have the watchdog on, so this branch cannot reuse the stale
            # daily close that produced the ~412 bps fallback on testnet.
            return float(self._closes[-1]) if self._closes else None
        return None

    def _cached_tick_price(self) -> float | None:
        cache = self.cache
        if cache is None:
            return None
        max_age_ns = FRESH_PRICE_MAX_AGE_SECS * 1e9
        now_ns = self._now_ns()
        with contextlib.suppress(Exception):
            quote = cache.quote_tick(self.instrument_id)
            if quote is not None and (now_ns - int(quote.ts_event)) <= max_age_ns:
                mid = (quote.bid_price.as_double() + quote.ask_price.as_double()) / 2.0
                if mid > 0:
                    return mid
        with contextlib.suppress(Exception):
            trade = cache.trade_tick(self.instrument_id)
            if trade is not None and (now_ns - int(trade.ts_event)) <= max_age_ns:
                price = float(trade.price)
                if price > 0:
                    return price
        return None

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
            trail_offset_bps = Decimal(f"{callback_rate_pct:.1f}") * Decimal("100")

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
                    tags=[f"{TAG_EXEC_ROLE}{ROLE_HARD_STOP}"],
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
                    tags=[f"{TAG_EXEC_ROLE}{ROLE_TAKE_PROFIT}"],
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
                    tags=["VOL_TRAIL", f"{TAG_EXEC_ROLE}{ROLE_TRAILING_STOP}"],
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
        self._abort_patient("strategy_stop")
        if self.cfg().flatten_on_stop:
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)
