"""KronosStrategy — the NautilusTrader strategy.

Flow per closed bar (on a cadence):
  buffer -> Kronos.forecast() -> make_signal() (cost filter) -> position_size()
  -> submit bracket order (entry + SL + TP).

Verified against NautilusTrader 1.228.0:
  - `Strategy` / `StrategyConfig` from nautilus_trader.trading.strategy
  - `self.order_factory.bracket(...)` -> OrderList -> `self.submit_order_list(...)`
  - ATR: nautilus_trader.indicators.volatility.AverageTrueRange (`.value`, `.initialized`)
  - equity via `self.portfolio.account(venue).balance_total(ccy)`

Critical correctness notes:
  - Only buffer CLOSED bars (no look-ahead). NT delivers a bar on `on_bar` only
    once it is closed, so appending here is safe.
  - Kronos inference is slow + blocking. In LIVE, run it off the event-loop
    thread (loop.run_in_executor or a sidecar). In BACKTEST, synchronous is fine
    and is what we do here.
  - Don't forecast every bar if it's expensive — respect `forecast_every_n_bars`.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from nautilus_trader.indicators.volatility import AverageTrueRange
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from kronos_mt5.brain.kronos_predictor import KronosBrain
from kronos_mt5.risk.sizing import Side, atr_stop_distance, make_signal, position_size


class KronosStrategyConfig(StrategyConfig, frozen=True):
    """Config for KronosStrategy. Only serializable fields (no brain here — the
    brain is injected into __init__ since it holds a torch model)."""

    instrument_id: str
    bar_type: str
    lookback: int = 400
    pred_len: int = 12
    forecast_every_n_bars: int = 1
    sample_count: int = 1
    kronos_t: float = 1.0
    kronos_top_p: float = 0.9
    use_volume: bool = False
    signal_threshold_bps: float = 8.0
    risk_per_trade: float = 0.005
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    take_profit_r: float = 1.5
    max_open_positions: int = 1
    # Fallback half-spread (in price) when no live quote is available (e.g. the
    # very first bars). Keeps the cost filter honest.
    default_spread: float = 0.00010
    # Minimum stop distance (price) so ATR≈0 never yields an oversized position.
    min_stop_distance: float = 0.00050


class KronosStrategy(Strategy):
    """Forecast-driven FX strategy. See module docstring for the flow."""

    def __init__(self, config: KronosStrategyConfig, brain: KronosBrain) -> None:
        super().__init__(config)
        self.brain = brain

        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.venue = self.instrument_id.venue

        self.atr = AverageTrueRange(config.atr_period)
        self._bars: deque[Bar] = deque(maxlen=config.lookback)
        self._bars_since_forecast = 0
        self.instrument = None

    # --- lifecycle ---------------------------------------------------------
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found in cache; stopping.")
            self.stop()
            return

        self.register_indicator_for_bars(self.bar_type, self.atr)
        # Historical warm-up (matters in live; in backtest the buffer simply
        # fills from the streamed bars, guarded by the lookback check in on_bar).
        # Wrapped defensively so a data client that can't serve history never
        # aborts the run.
        try:
            start = self.clock.utc_now() - pd.Timedelta(minutes=15) * (self.cfg().lookback + 1)
            self.request_bars(self.bar_type, start=start)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"request_bars warm-up skipped: {exc!r}")
        self.subscribe_bars(self.bar_type)
        self.log.info(f"KronosStrategy started on {self.bar_type}")

    def on_bar(self, bar: Bar) -> None:
        """Append the closed bar; forecast on cadence; act on the signal."""
        self._bars.append(bar)
        self._bars_since_forecast += 1

        if len(self._bars) < self.cfg().lookback:
            return
        if not self.atr.initialized:
            return
        if self._bars_since_forecast < self.cfg().forecast_every_n_bars:
            return
        self._bars_since_forecast = 0

        df, x_ts, y_ts = self._build_frames()
        forecast = self._run_forecast(df, x_ts, y_ts)

        spread = self._current_spread()
        signal = make_signal(forecast, spread, self.cfg().signal_threshold_bps)
        self.log.info(
            f"forecast er={forecast.expected_return:.6f} prob_up={forecast.prob_up:.2f} "
            f"spread={spread:.5f} -> {signal.side.value} ({signal.reason})"
        )

        if signal.side is Side.FLAT:
            return

        # max_open_positions / no-flip-without-exit: only enter when flat.
        if not self.portfolio.is_flat(self.instrument_id):
            self.log.info("Position already open; skipping new entry (bracket manages exit).")
            return

        stop_dist = max(
            atr_stop_distance(self.atr.value, self.cfg().atr_stop_mult),
            self.cfg().min_stop_distance,
        )
        equity = self._equity()
        ref = self._reference_price()
        pip_value_per_unit = self._value_per_unit_per_price(ref)
        raw_units = position_size(equity, self.cfg().risk_per_trade, stop_dist, pip_value_per_unit)
        self._submit_bracket(signal.side, raw_units, stop_dist)

    def on_stop(self) -> None:
        # Fail safe: cancel resting orders and flatten on stop.
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

    # --- helpers -----------------------------------------------------------
    def cfg(self) -> KronosStrategyConfig:
        return self.config  # typed accessor

    def _build_frames(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Build the OHLCV DataFrame + x/y timestamps from the closed-bar buffer."""
        rows = [
            (
                pd.Timestamp(b.ts_event, unit="ns"),
                b.open.as_double(),
                b.high.as_double(),
                b.low.as_double(),
                b.close.as_double(),
                b.volume.as_double(),
            )
            for b in self._bars
        ]
        df = pd.DataFrame(rows, columns=["timestamps", "open", "high", "low", "close", "volume"])
        x_ts = df["timestamps"]

        interval = x_ts.iloc[-1] - x_ts.iloc[-2] if len(x_ts) >= 2 else pd.Timedelta(minutes=15)
        y_ts = pd.Series(
            pd.date_range(x_ts.iloc[-1] + interval, periods=self.cfg().pred_len, freq=interval)
        )
        return df[["open", "high", "low", "close", "volume"]], x_ts, y_ts

    def _run_forecast(self, df, x_ts, y_ts):
        """In backtest call directly; in live, dispatch to an executor so Kronos
        inference never blocks the engine's event loop."""
        return self.brain.forecast(
            df=df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=self.cfg().pred_len,
            sample_count=self.cfg().sample_count,
            temperature=self.cfg().kronos_t,
            top_p=self.cfg().kronos_top_p,
            use_volume=self.cfg().use_volume,
        )

    def _current_spread(self) -> float:
        quote = self.cache.quote_tick(self.instrument_id)
        if quote is not None:
            return float(quote.ask_price - quote.bid_price)
        return self.cfg().default_spread

    def _account_currency(self):
        account = self.portfolio.account(self.venue)
        if account is None:
            return None
        # Margin/cash FX accounts carry a single base currency (e.g. USD).
        return getattr(account, "base_currency", None) or self.instrument.quote_currency

    def _equity(self) -> float:
        account = self.portfolio.account(self.venue)
        if account is None:
            return 0.0
        balance = account.balance_total(self._account_currency())
        return balance.as_double() if balance is not None else 0.0

    def _value_per_unit_per_price(self, ref_price: float) -> float:
        """Account-currency value of a 1.0 price move per 1 unit of base ccy.

        - quote == account ccy (EUR/USD, GBP/USD on a USD account): 1.0
        - base  == account ccy (USD/JPY on a USD account): 1/price, since PnL is
          earned in the quote ccy (JPY) and converted back at the pair's rate.
        - otherwise (a true cross vs the account ccy): fall back to 1.0 and warn;
          a proper conversion needs another instrument's rate from the cache.
        """
        acct = self._account_currency()
        if acct is None or self.instrument.quote_currency == acct:
            return 1.0
        if self.instrument.base_currency == acct and ref_price > 0:
            return 1.0 / ref_price
        self.log.warning(
            f"No conversion rate from {self.instrument.quote_currency} to {acct}; "
            "sizing with value_per_unit=1.0 (may be inaccurate for cross pairs)."
        )
        return 1.0

    def _submit_bracket(self, side: Side, units: float, stop_dist: float) -> None:
        """Build + submit a market entry with attached SL/TP via OrderFactory."""
        qty = self.instrument.make_qty(units)
        if qty <= self.instrument.make_qty(0):
            self.log.info(f"Sized qty {units:.0f} rounds to zero; skipping.")
            return
        if qty < self.instrument.min_quantity:
            self.log.info(f"Sized qty {qty} < min {self.instrument.min_quantity}; skipping.")
            return

        ref = self._reference_price()
        tp_dist = stop_dist * self.cfg().take_profit_r
        if side is Side.LONG:
            order_side = OrderSide.BUY
            sl = ref - stop_dist
            tp = ref + tp_dist
        else:
            order_side = OrderSide.SELL
            sl = ref + stop_dist
            tp = ref - tp_dist

        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=qty,
            sl_trigger_price=self.instrument.make_price(sl),
            tp_price=self.instrument.make_price(tp),
        )
        self.submit_order_list(bracket)
        self.log.info(
            f"BRACKET {side.value} qty={qty} ref={ref:.5f} sl={sl:.5f} tp={tp:.5f} "
            f"ids={[o.client_order_id.value for o in bracket.orders]}"
        )

    def _reference_price(self) -> float:
        quote = self.cache.quote_tick(self.instrument_id)
        if quote is not None:
            return float((quote.ask_price + quote.bid_price) / 2)
        return self._bars[-1].close.as_double()
