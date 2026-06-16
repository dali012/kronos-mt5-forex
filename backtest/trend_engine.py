"""Event-driven (NautilusTrader) backtest of the crypto trend portfolio.

Execution-realistic validation of backtest/trend.py: runs one TrendStrategy per
coin in a single engine with realistic fees + slippage, on daily bars. This is
the gate the vectorized research backtest must clear before any live demo.

Run:  python backtest/trend_engine.py --symbols BTCUSDT ETHUSDT ... --start 2018-01-01
Verified against NautilusTrader 1.228.0.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, CurrencyType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider  # noqa: F401 (kept for reference)

from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from backtest.trend import stats
from kronos_mt5.strategies.risk_manager import RiskManager, RiskManagerConfig, RiskState
from kronos_mt5.strategies.trend_strategy import TrendStrategy, TrendStrategyConfig


class EquityRecorderConfig(StrategyConfig, frozen=True):
    bar_types: tuple[str, ...]
    venue: str


class EquityRecorder(Strategy):
    """Records daily mark-to-market equity (cash + unrealized PnL) — the account
    report's 'total' is cash-only and misses open-position MTM."""

    def __init__(self, config: EquityRecorderConfig) -> None:
        super().__init__(config)
        self._bts = [BarType.from_str(b) for b in config.bar_types]
        self._venue = Venue(config.venue)
        self.records: dict[pd.Timestamp, float] = {}

    def on_start(self) -> None:
        for bt in self._bts:
            self.subscribe_bars(bt)

    def on_bar(self, bar) -> None:
        acct = self.portfolio.account(self._venue)
        if acct is None:
            return
        cash = acct.balance_total(USDT)
        equity = cash.as_double() if cash is not None else 0.0
        for bt in self._bts:
            iid = bt.instrument_id
            upnl = self.portfolio.unrealized_pnl(iid)
            if upnl is not None:
                equity += upnl.as_double()
        day = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").normalize()
        self.records[day] = equity  # last write per day wins

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
REPORTS = REPO / "reports"
VENUE = "BINANCE"
PX_PREC = 6
SZ_PREC = 6


def _currency(code: str) -> Currency:
    try:
        return Currency.from_str(code)
    except Exception:  # noqa: BLE001
        cur = Currency(code, precision=8, iso4217=0, name=code, currency_type=CurrencyType.CRYPTO)
        Currency.register(cur)
        return cur


def crypto_instrument(symbol: str, taker_fee: float) -> CurrencyPair:
    """Generic Binance-spot-like CurrencyPair for any COINUSDT symbol."""
    assert symbol.endswith("USDT"), symbol
    base = _currency(symbol[:-4])
    return CurrencyPair(
        instrument_id=InstrumentId(symbol=Symbol(symbol), venue=Venue(VENUE)),
        raw_symbol=Symbol(symbol),
        base_currency=base,
        quote_currency=USDT,
        price_precision=PX_PREC,
        size_precision=SZ_PREC,
        price_increment=Price(10 ** -PX_PREC, precision=PX_PREC),
        size_increment=Quantity(10 ** -SZ_PREC, precision=SZ_PREC),
        lot_size=None,
        max_quantity=Quantity(1e9, precision=SZ_PREC),
        min_quantity=Quantity(10 ** -SZ_PREC, precision=SZ_PREC),
        max_notional=None,
        min_notional=Money(1.0, USDT),
        max_price=Price(1e9, precision=PX_PREC),
        min_price=Price(10 ** -PX_PREC, precision=PX_PREC),
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.05"),
        maker_fee=Decimal(str(taker_fee)),
        taker_fee=Decimal(str(taker_fee)),
        ts_event=0,
        ts_init=0,
    )


def load_daily(symbol: str, start: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{symbol}_1D.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
    df = df.set_index("timestamps").sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    return df[["open", "high", "low", "close", "volume"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="NautilusTrader crypto trend backtest")
    ap.add_argument("--symbols", nargs="+",
                    default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT",
                             "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "TRXUSDT"])
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--balance", type=float, default=100_000.0)
    ap.add_argument("--leverage", type=int, default=5)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--taker-fee", type=float, default=0.0004, help="per-side fee (0.0004 = 8bps round trip)")
    ap.add_argument("--prob-slippage", type=float, default=0.5)
    ap.add_argument("--stop-pct", type=float, default=0.0, help="per-position stop-loss (0 = off)")
    ap.add_argument("--tp-pct", type=float, default=0.0, help="per-position take-profit (0 = off)")
    ap.add_argument("--max-drawdown", type=float, default=0.0, help="portfolio kill-switch DD (0 = off)")
    args = ap.parse_args()

    venue = Venue(VENUE)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="TREND-001",
            logging=LoggingConfig(log_level="ERROR", bypass_logging=True),
        )
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(args.balance, USDT)],
        default_leverage=Decimal(args.leverage),
        fill_model=FillModel(prob_slippage=args.prob_slippage, random_seed=42),
    )

    n = len(args.symbols)
    loaded, bar_types, iids = [], [], []
    risk = RiskState()
    for sym in args.symbols:
        df = load_daily(sym, args.start)
        if len(df) < 300:
            print(f"  skip {sym}: only {len(df)} bars")
            continue
        instrument = crypto_instrument(sym, args.taker_fee)
        engine.add_instrument(instrument)
        bar_type = BarType.from_str(f"{sym}.{VENUE}-1-DAY-LAST-EXTERNAL")
        bars = BarDataWrangler(bar_type, instrument).process(df)
        engine.add_data(bars)
        strat = TrendStrategy(
            TrendStrategyConfig(
                instrument_id=instrument.id.value,
                bar_type=str(bar_type),
                n_instruments=n,
                target_vol=args.target_vol,
                ppy=365,
                warmup_request=False,  # backtest streams all bars via on_bar
                use_stop_loss=args.stop_pct > 0,
                stop_pct=args.stop_pct or 0.20,
                use_take_profit=args.tp_pct > 0,
                tp_pct=args.tp_pct or 0.50,
            )
        )
        strat.risk_state = risk
        engine.add_strategy(strat)
        loaded.append(sym)
        bar_types.append(str(bar_type))
        iids.append(instrument.id.value)

    if args.max_drawdown > 0:
        rm = RiskManager(
            RiskManagerConfig(
                venue=VENUE, instrument_ids=tuple(iids),
                max_drawdown=args.max_drawdown, check_secs=86400,  # daily check in sim time
            )
        )
        rm.risk_state = risk
        engine.add_strategy(rm)

    recorder = EquityRecorder(EquityRecorderConfig(bar_types=tuple(bar_types), venue=VENUE))
    engine.add_strategy(recorder)

    print(f"Universe: {loaded} | start {args.start} | balance ${args.balance:,.0f} "
          f"lev {args.leverage} | fee {args.taker_fee*2*1e4:.0f}bps round-trip")
    print("Running engine...")
    engine.run()

    # daily mark-to-market equity curve (cash + unrealized PnL)
    curve = pd.Series(recorder.records).sort_index()
    rets = curve.pct_change().dropna()
    st = stats(rets, ppy=365)  # crypto trades 365 days/yr
    start_bal, end_bal = float(curve.iloc[0]), float(curve.iloc[-1])
    n_fills = len(engine.trader.generate_order_fills_report())

    print("\n=== NAUTILUSTRADER CRYPTO TREND — RESULT ===")
    print(f"  start ${start_bal:,.0f} -> end ${end_bal:,.0f}  ({end_bal/start_bal-1:+.1%} total)")
    print(f"  Sharpe {st['sharpe']:+.2f} | ann_ret {st['ann_return']:+.1%} | vol {st['ann_vol']:.1%} | "
          f"maxDD {st['max_dd']:.1%} | Calmar {st['calmar']:.2f}")
    print(f"  fills: {n_fills}")

    try:
        import quantstats as qs

        REPORTS.mkdir(exist_ok=True)
        out = REPORTS / "tearsheet_trend_engine.html"
        qs.reports.html(rets, output=str(out), title="Crypto trend (NautilusTrader)")
        print(f"  [tearsheet] {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [tearsheet] skipped ({exc!r})")

    engine.dispose()


if __name__ == "__main__":
    main()
