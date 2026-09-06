"""Replay the production TrendStrategy in Nautilus with explicit event timing.

The subclass changes transport/execution only: closed-bar decisions are queued
until the following open. Native protective orders use synthetic OHLC quotes.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Quantity

from kronos_mt5.strategies.risk_manager import RiskState
from kronos_mt5.strategies.trend_strategy import TrendStrategy, TrendStrategyConfig

from .config import STRATEGY, BaselineConfig
from .instruments import instrument, reject_reason, rounded

DAY_NS = 86_400_000_000_000


def funding_payment(units: float, mark_price: float, rate: float) -> float:
    """Signed account cash flow: a positive rate debits longs and credits shorts."""
    return -units * mark_price * rate


def friction(
    notional: float, commission_bps: float, half_spread_bps: float, slippage_bps: float
) -> dict:
    return {
        "commission": abs(notional) * commission_bps / 1e4,
        "spread": abs(notional) * half_spread_bps / 1e4,
        "slippage": abs(notional) * slippage_bps / 1e4,
    }


class FundingModule(SimulationModule):
    def __init__(self, events: list[dict], rates: dict, start_ns: int, quotes: dict):
        super().__init__(SimulationModuleConfig())
        self.events = sorted(events, key=lambda r: (r["ts_ns"], r["symbol"]))
        self.rates = rates
        self.quotes = quotes
        self.start_ns = start_ns
        self.cursor = 0
        self.last_mid: dict[str, float] = {}
        self.payments: list[dict] = []

    def pre_process(self, data):
        # Settle positions held BEFORE the event, including midnight before new entries.
        while self.cursor < len(self.events) and self.events[self.cursor]["ts_ns"] <= data.ts_init:
            event = self.events[self.cursor]
            self.cursor += 1
            symbol = event["symbol"]
            self.rates[symbol] = event["funding_rate"]  # only settled rates, never future funding
            units = sum(
                float(p.signed_qty)
                for p in self.exchange.cache.positions_open()
                if p.instrument_id.symbol.value == symbol
            )
            price = event["mark_price"]
            approximate = price is None or not np.isfinite(price)
            if approximate:
                price = self.last_mid.get(symbol)
            if units and (price is None or price <= 0):
                raise ValueError("no causal mark available to value funding")
            amount = funding_payment(units, price or 0, event["funding_rate"])
            if event["ts_ns"] >= self.start_ns:
                if amount:
                    self.exchange.adjust_account(Money(amount, USDT))
                self.payments.append(
                    {
                        **event,
                        "units": units,
                        "valuation_price": price,
                        "approximate_mark": approximate,
                        "cash_flow": amount,
                    }
                )
        if isinstance(data, QuoteTick):
            self.quotes[data.instrument_id] = data
            self.last_mid[data.instrument_id.symbol.value] = (
                float(data.bid_price) + float(data.ask_price)
            ) / 2

    def process(self, ts_now):
        pass

    def log_diagnostics(self, logger):
        pass

    def reset(self):
        self.cursor = 0
        self.last_mid.clear()
        self.quotes.clear()
        self.payments.clear()
        self.rates.clear()


class ReplayStrategy(TrendStrategy):
    def __init__(
        self,
        config,
        replay: BaselineConfig,
        filters: dict,
        start_ns: int,
        end_ns: int,
        shared: dict,
    ):
        super().__init__(config)
        self.replay = replay
        self.filters = filters
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.shared = shared
        self.pending = None
        self.decisions = []
        self.fills = []
        self.rejections = []
        self.signal_history = []

    def on_start(self):
        # Offline startup: avoid all live history requests and wall-clock timers.
        self.instrument = self.cache.instrument(self.instrument_id)
        self.subscribe_bars(self.bar_type)
        self.subscribe_quote_ticks(self.instrument_id)

    def on_bar(self, bar):
        if bar.ts_event < self.start_ns - 1:
            self._closes.append(float(bar.close))
            return
        if bar.ts_event >= self.end_ns - 1:
            self.pending = None
            return
        super().on_bar(bar)
        self.signal_history.append({"ts_ns": bar.ts_event, "closes": list(self._closes)})
        self.snapshot(bar.ts_event)

    def _submit_entry_order(self, side, qty, price, *, target_units, threshold_notional):
        decision = {
            "ts_ns": self.clock.timestamp_ns(),
            "side": side.name,
            "qty": float(qty),
            "reference_price": price,
            "target_units": target_units,
        }
        self.decisions.append(decision)
        self.pending = decision

    def _order_passes_filters(self, qty, price):
        ok = super()._order_passes_filters(qty, price)
        if not ok:
            self.rejections.append(
                {"ts_ns": self.clock.timestamp_ns(), "reason": "production_filters"}
            )
        return ok

    def on_quote_tick(self, tick):
        ts = tick.ts_event
        if ts < self.start_ns:
            return
        self.shared["mid"][self.instrument_id] = (float(tick.bid_price) + float(tick.ask_price)) / 2
        equity = self._equity()
        self.shared["peak"] = max(self.shared["peak"], equity)
        if equity <= self.shared["peak"] * (1 - self.replay.max_drawdown):
            self.risk_state.mode = "kill"
        if ts >= self.end_ns - 2 or self.risk_state.mode == "kill":
            self.pending = None
            self._flatten_self()
            self.snapshot(ts)
            return
        # Only the open quote can execute a pending closed-bar decision.
        if ts % DAY_NS == 0 and self.pending is not None:
            pending, self.pending = self.pending, None
            if pending["ts_ns"] >= ts:
                raise ValueError("look-ahead: decision must precede executable open")
            side = OrderSide.BUY if pending["side"] == "BUY" else OrderSide.SELL
            price = float(tick.ask_price if side == OrderSide.BUY else tick.bid_price)
            qty = rounded(pending["qty"], self.filters["step_size"])
            reason = reject_reason(qty, price, self.filters)
            current = float(self.portfolio.net_position(self.instrument_id))
            delta = qty if side == OrderSide.BUY else -qty
            gross = sum(
                abs(float(self.portfolio.net_position(iid))) * mid
                for iid, mid in self.shared["mid"].items()
            )
            future_gross = (
                gross
                - abs(current) * self.shared["mid"][self.instrument_id]
                + abs(current + delta) * price
            )
            if future_gross > max(0, equity) * self.replay.leverage and abs(current + delta) > abs(
                current
            ):
                reason = "leverage_limit"
            if reason:
                self.rejections.append({"ts_ns": ts, "reason": reason})
            else:
                order = self.order_factory.market(
                    self.instrument_id,
                    side,
                    self.instrument.make_qty(qty),
                    tags=[f"baseline_decision_ns={pending['ts_ns']}"],
                )
                self.submit_order(order)
        self.snapshot(ts)

    def on_order_filled(self, event):
        if event.instrument_id == self.instrument_id:
            # Native stops fill in the matching engine BEFORE DataEngine updates
            # cache.quote_tick. Use the quote captured in pre_process, otherwise
            # a market gap is incorrectly attributed entirely to slippage.
            quote = self.shared["quotes"][self.instrument_id]
            mid = (float(quote.bid_price) + float(quote.ask_price)) / 2
            price, qty = float(event.last_px), float(event.last_qty)
            spread_slippage = qty * abs(price - mid)
            total_bps = self.replay.half_spread_bps + self.replay.slippage_bps
            self.fills.append(
                {
                    "ts_ns": event.ts_event,
                    "symbol": self.instrument_id.symbol.value,
                    "side": event.order_side.name,
                    "qty": qty,
                    "price": price,
                    "mid": mid,
                    "commission": float(event.commission.as_double()),
                    "spread": spread_slippage * self.replay.half_spread_bps / total_bps,
                    "slippage": spread_slippage * self.replay.slippage_bps / total_bps,
                    "notional": qty * price,
                }
            )
        super().on_order_filled(event)
        if event.ts_event >= self.start_ns:
            self.snapshot(event.ts_event)

    def on_order_rejected(self, event):
        self.rejections.append({"ts_ns": event.ts_event, "reason": str(event.reason)})
        super().on_order_rejected(event)

    def snapshot(self, ts):
        if ts % DAY_NS == 0 and ts < self.end_ns:
            ts += 1  # preserve initial capital and attribute open fills to the new day
        positions = self.cache.positions_open()
        equity = self._equity()
        gross = sum(
            abs(float(p.signed_qty)) * self.shared["mid"].get(p.instrument_id, float(p.avg_px_open))
            for p in positions
        )
        self.shared["observations"][ts] = {
            "ts_ns": ts,
            "equity": equity,
            "n_open": len(positions),
            "gross_exposure": gross / equity if equity > 0 else None,
            "regime": self.risk_state.correlation_metrics(90, 0.65, 0.5)["volatility_regime"],
        }


def events_for_frame(frame: pd.DataFrame, inst, config: BaselineConfig, funding_times: list[int]):
    """Closed bars at end-1ns; executable opens at start. No synthetic future close quote."""
    bt = BarType.from_str(f"{inst.id}-1-DAY-LAST-EXTERNAL")
    # Use instrument increments, not a nominal number of decimal places.
    tick = str(inst.price_increment)
    half = (config.half_spread_bps + config.slippage_bps) / 1e4
    result = []
    for r in frame.to_dict("records"):
        start = int(r["open_time"]) * 1_000_000
        prices = {
            start: r["open"],
            start + DAY_NS // 3 + 1: r["high" if config.bar_path == "OHLC" else "low"],
            start + 2 * DAY_NS // 3 + 1: r["low" if config.bar_path == "OHLC" else "high"],
            start + DAY_NS - 2: r["close"],
        }
        for ts in funding_times:
            if start <= ts < start + DAY_NS and ts not in prices:
                prices[ts] = prices[max(t for t in prices if t <= ts)]
        for ts, mid in sorted(prices.items()):
            bid = inst.make_price(rounded(mid * (1 - half), tick))
            ask = inst.make_price(rounded(mid * (1 + half), tick, up=True))
            result.append(
                QuoteTick(
                    inst.id,
                    bid,
                    ask,
                    Quantity(1_000_000, inst.size_precision),
                    Quantity(1_000_000, inst.size_precision),
                    ts,
                    ts,
                )
            )
        close_ts = start + DAY_NS - 1
        result.append(
            Bar(
                bt,
                inst.make_price(r["open"]),
                inst.make_price(r["high"]),
                inst.make_price(r["low"]),
                inst.make_price(r["close"]),
                inst.make_qty(r["volume"]),
                close_ts,
                close_ts,
            )
        )
    return result


def run_engine(
    frames: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
    filters: dict,
    config: BaselineConfig,
    start_ns: int,
    end_ns: int,
) -> dict:
    if start_ns >= end_ns or start_ns % DAY_NS or end_ns % DAY_NS:
        raise ValueError("window must be an ordered half-open UTC day range")
    venue = Venue("BINANCE")
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(bypass_logging=True)))
    rates = {}
    events = [
        {
            "symbol": s,
            "ts_ns": int(r["funding_time"]) * 1_000_000,
            "funding_rate": float(r["funding_rate"]),
            "mark_price": r["mark_price"],
        }
        for s, f in funding.items()
        for r in f.to_dict("records")
        if start_ns - 253 * DAY_NS <= int(r["funding_time"]) * 1_000_000 < end_ns
    ]
    quotes = {}
    module = FundingModule(events, rates, start_ns, quotes)
    engine.add_venue(
        venue,
        OmsType.NETTING,
        AccountType.MARGIN,
        starting_balances=[Money(config.starting_equity, USDT)],
        base_currency=USDT,
        default_leverage=Decimal(str(config.leverage)),
        modules=[module],
        bar_execution=False,
        fill_model=FillModel(prob_slippage=0.0, random_seed=42),
        use_message_queue=False,
    )
    risk = RiskState()
    risk.run_id = "offline-baseline"
    shared = {
        "quotes": quotes,
        "mid": {},
        "peak": config.starting_equity,
        "observations": {
            start_ns: {
                "ts_ns": start_ns,
                "equity": config.starting_equity,
                "n_open": 0,
                "gross_exposure": 0.0,
                "regime": "UNKNOWN",
            }
        },
    }
    strategies = []
    try:
        iids = tuple(f"{s}.BINANCE" for s in config.symbols)
        for symbol in config.symbols:
            inst = instrument(symbol, filters[symbol], config.commission_bps, config.leverage)
            engine.add_instrument(inst)
            frame = frames[symbol]
            times = frame["open_time"] * 1_000_000
            frame = frame[(times >= start_ns - 253 * DAY_NS) & (times < end_ns)]
            if sum(frame["open_time"] * 1_000_000 < start_ns) < 253:
                raise ValueError(f"insufficient 253-bar warm-up for {symbol}")
            expected = pd.Series(range(start_ns - 253 * DAY_NS, end_ns, DAY_NS))
            if frame["open_time"].tolist() != (expected // 1_000_000).tolist():
                raise ValueError(f"non-contiguous daily candles for {symbol}")
            data = events_for_frame(
                frame, inst, config, [e["ts_ns"] for e in events if e["symbol"] == symbol]
            )
            engine.add_data(data)
            strategy = ReplayStrategy(
                TrendStrategyConfig(
                    instrument_id=str(inst.id),
                    bar_type=f"{inst.id}-1-DAY-LAST-EXTERNAL",
                    portfolio_instrument_ids=iids,
                    n_instruments=len(config.symbols),
                    warmup_request=False,
                    **STRATEGY,
                ),
                config,
                filters[symbol],
                start_ns,
                end_ns,
                shared,
            )
            strategy.risk_state = risk
            strategy.funding_rates = SimpleNamespace(get=lambda s: rates.get(s))
            strategies.append(strategy)
            engine.add_strategy(strategy)
        engine.run()
        for strategy in strategies:
            strategy.snapshot(end_ns)
        if engine.cache.positions_open():
            raise ValueError("window did not liquidate all open positions")
        result = {
            "observations": [
                v for k, v in sorted(shared["observations"].items()) if start_ns <= k <= end_ns
            ],
            "fills": sorted(
                [f for s in strategies for f in s.fills], key=lambda r: (r["ts_ns"], r["symbol"])
            ),
            "funding": module.payments,
            "decisions": [d for s in strategies for d in s.decisions],
            "rejections": [r for s in strategies for r in s.rejections],
            "signal_history": {s.instrument_id.symbol.value: s.signal_history for s in strategies},
            "start_ns": start_ns,
            "end_ns": end_ns,
        }
        return result
    finally:
        engine.dispose()
