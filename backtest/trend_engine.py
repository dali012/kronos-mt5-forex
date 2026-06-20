"""Event-driven (NautilusTrader) backtest of the crypto trend portfolio.

Execution-realistic validation of backtest/trend.py: runs one TrendStrategy per
coin in a single engine with realistic fees + slippage, on daily bars. This is
the gate the vectorized research backtest must clear before any live demo.

Run:  python backtest/trend_engine.py --symbols BTCUSDT ETHUSDT ... --start 2018-01-01
Verified against NautilusTrader 1.228.0.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

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

from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from backtest.trend import load_funding, stats
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
        self.notionals: dict[pd.Timestamp, dict[str, float]] = {}

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
        self.notionals.setdefault(day, {})[bar.bar_type.instrument_id.symbol.value] = (
            float(self.portfolio.net_position(bar.bar_type.instrument_id)) * bar.close.as_double()
        )


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
        price_increment=Price(10**-PX_PREC, precision=PX_PREC),
        size_increment=Quantity(10**-SZ_PREC, precision=SZ_PREC),
        lot_size=None,
        max_quantity=Quantity(1e9, precision=SZ_PREC),
        min_quantity=Quantity(10**-SZ_PREC, precision=SZ_PREC),
        max_notional=None,
        min_notional=Money(1.0, USDT),
        max_price=Price(1e9, precision=PX_PREC),
        min_price=Price(10**-PX_PREC, precision=PX_PREC),
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


def live_symbols() -> list[str]:
    from config.settings import settings

    return [s.strip() for s in settings.binance_symbols.split(",") if s.strip()]


def apply_funding_to_curve(
    curve: pd.Series,
    notionals: dict[pd.Timestamp, dict[str, float]],
    funding_dir: Path | None,
) -> pd.Series:
    if funding_dir is None:
        return curve
    funding_by_symbol = {}
    for symbol in sorted({s for day in notionals.values() for s in day}):
        funding = load_funding(symbol, funding_dir)
        if funding is None:
            print(f"  funding missing for {symbol}: engine funding cost ignored")
            continue
        funding_by_symbol[symbol] = funding
    if not funding_by_symbol:
        return curve

    daily_pnl = pd.Series(0.0, index=curve.index)
    for day, by_symbol in notionals.items():
        if day not in daily_pnl.index:
            continue
        pnl = 0.0
        for symbol, notional in by_symbol.items():
            funding = funding_by_symbol.get(symbol)
            if funding is None:
                continue
            pnl += -notional * float(funding.get(day, 0.0))
        daily_pnl.loc[day] = pnl
    return curve + daily_pnl.cumsum()


def main() -> None:
    ap = argparse.ArgumentParser(description="NautilusTrader crypto trend backtest")
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "DOGEUSDT",
            "LTCUSDT",
            "LINKUSDT",
            "DOTUSDT",
            "AVAXUSDT",
            "TRXUSDT",
        ],
    )
    ap.add_argument(
        "--live-universe", action="store_true", help="use BINANCE_SYMBOLS from config/.env"
    )
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--balance", type=float, default=100_000.0)
    ap.add_argument("--leverage", type=int, default=5)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument(
        "--taker-fee", type=float, default=0.0004, help="per-side fee (0.0004 = 8bps round trip)"
    )
    ap.add_argument("--prob-slippage", type=float, default=0.5)
    ap.add_argument("--stop-pct", type=float, default=0.0, help="per-position stop-loss (0 = off)")
    ap.add_argument("--tp-pct", type=float, default=0.0, help="per-position take-profit (0 = off)")
    ap.add_argument(
        "--max-drawdown", type=float, default=0.0, help="portfolio kill-switch DD (0 = off)"
    )
    ap.add_argument(
        "--funding-dir", type=Path, default=None, help="directory of SYMBOL_funding.csv files"
    )
    ap.add_argument("--use-vol-stop", action="store_true")
    ap.add_argument("--stop-vol-mult", type=float, default=4.0)
    ap.add_argument("--min-stop-pct", type=float, default=0.08)
    ap.add_argument("--max-stop-pct", type=float, default=0.30)
    ap.add_argument("--use-correlation-scaling", action="store_true")
    ap.add_argument("--corr-window", type=int, default=90)
    ap.add_argument("--corr-threshold", type=float, default=0.65)
    ap.add_argument("--corr-min-scalar", type=float, default=0.50)
    ap.add_argument("--use-portfolio-allocator", action="store_true")
    ap.add_argument("--portfolio-target-vol", type=float, default=0.10)
    ap.add_argument("--portfolio-cov-window", type=int, default=90)
    ap.add_argument("--portfolio-cov-shrinkage", type=float, default=0.25)
    ap.add_argument("--portfolio-min-scale", type=float, default=0.25)
    ap.add_argument("--portfolio-max-scale", type=float, default=1.25)
    ap.add_argument("--portfolio-max-gross", type=float, default=1.50)
    ap.add_argument("--portfolio-scale-up-alpha", type=float, default=0.20)
    ap.add_argument("--portfolio-scale-deadband", type=float, default=0.05)
    ap.add_argument("--cost-aware-rebalance", action="store_true")
    ap.add_argument("--round-trip-cost-bps", type=float, default=8.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--use-patient-limit", action="store_true")
    ap.add_argument("--patient-limit-offset-bps", type=float, default=2.0)
    ap.add_argument("--patient-limit-timeout-secs", type=int, default=300)
    ap.add_argument("--no-patient-limit-market-fallback", action="store_true")
    args = ap.parse_args()
    if args.live_universe:
        args.symbols = live_symbols()

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
        loaded.append(sym)
        bar_types.append(str(bar_type))
        iids.append(instrument.id.value)

    n = len(loaded)
    for iid, bar_type in zip(iids, bar_types, strict=True):
        strat = TrendStrategy(
            TrendStrategyConfig(
                instrument_id=iid,
                bar_type=bar_type,
                portfolio_instrument_ids=tuple(iids),
                n_instruments=n,
                target_vol=args.target_vol,
                ppy=365,
                warmup_request=False,  # backtest streams all bars via on_bar
                use_stop_loss=args.stop_pct > 0,
                stop_pct=args.stop_pct or 0.20,
                use_vol_stop=args.use_vol_stop,
                stop_vol_mult=args.stop_vol_mult,
                min_stop_pct=args.min_stop_pct,
                max_stop_pct=args.max_stop_pct,
                use_take_profit=args.tp_pct > 0,
                tp_pct=args.tp_pct or 0.50,
                use_correlation_scaling=args.use_correlation_scaling,
                corr_window=args.corr_window,
                corr_threshold=args.corr_threshold,
                corr_min_scalar=args.corr_min_scalar,
                use_portfolio_allocator=args.use_portfolio_allocator,
                portfolio_target_vol=args.portfolio_target_vol,
                portfolio_cov_window=args.portfolio_cov_window,
                portfolio_cov_shrinkage=args.portfolio_cov_shrinkage,
                portfolio_min_scale=args.portfolio_min_scale,
                portfolio_max_scale=args.portfolio_max_scale,
                portfolio_max_gross=args.portfolio_max_gross,
                portfolio_scale_up_alpha=args.portfolio_scale_up_alpha,
                portfolio_scale_deadband=args.portfolio_scale_deadband,
                cost_aware_rebalance=args.cost_aware_rebalance,
                round_trip_cost_bps=args.round_trip_cost_bps,
                slippage_bps=args.slippage_bps,
                use_patient_limit=args.use_patient_limit,
                patient_limit_offset_bps=args.patient_limit_offset_bps,
                patient_limit_timeout_secs=args.patient_limit_timeout_secs,
                patient_limit_market_fallback=not args.no_patient_limit_market_fallback,
            )
        )
        strat.risk_state = risk
        engine.add_strategy(strat)

    if args.max_drawdown > 0:
        rm = RiskManager(
            RiskManagerConfig(
                venue=VENUE,
                instrument_ids=tuple(iids),
                max_drawdown=args.max_drawdown,
                check_secs=86400,  # daily check in sim time
                persist=False,  # don't touch the live bot's companion.db
            )
        )
        rm.risk_state = risk
        engine.add_strategy(rm)

    recorder = EquityRecorder(EquityRecorderConfig(bar_types=tuple(bar_types), venue=VENUE))
    engine.add_strategy(recorder)

    print(
        f"Universe: {loaded} | start {args.start} | balance ${args.balance:,.0f} "
        f"lev {args.leverage} | fee {args.taker_fee*2*1e4:.0f}bps round-trip"
    )
    print("Running engine...")
    engine.run()

    # daily mark-to-market equity curve (cash + unrealized PnL)
    curve = pd.Series(recorder.records).sort_index()
    curve = apply_funding_to_curve(curve, recorder.notionals, args.funding_dir)
    rets = curve.pct_change().dropna()
    st = stats(rets, ppy=365)  # crypto trades 365 days/yr
    start_bal, end_bal = float(curve.iloc[0]), float(curve.iloc[-1])
    n_fills = len(engine.trader.generate_order_fills_report())

    print("\n=== NAUTILUSTRADER CRYPTO TREND — RESULT ===")
    print(f"  start ${start_bal:,.0f} -> end ${end_bal:,.0f}  ({end_bal/start_bal-1:+.1%} total)")
    print(
        f"  Sharpe {st['sharpe']:+.2f} | ann_ret {st['ann_return']:+.1%} | vol {st['ann_vol']:.1%} | "
        f"maxDD {st['max_dd']:.1%} | Calmar {st['calmar']:.2f}"
    )
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
