"""Shared backtest harness — used by run_backtest.py and walk_forward.py.

Builds a NautilusTrader `BacktestEngine` from a CSV of OHLCV bars, with
MANDATORY realistic costs:
  - spread:   quote ticks are synthesized from bars with a configurable
              half-spread, so every fill pays the bid/ask.
  - slippage: a `FillModel(prob_slippage=...)` slips fills by a tick.
  - commission: a `FixedFeeModel` charges per fill.

A backtest without these is treated as a bug (see CLAUDE.md). Execution runs off
the synthesized quotes (`bar_execution=False`); the strategy consumes the
external LAST bars for its rolling forecast buffer.

Verified against NautilusTrader 1.228.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, FixedFeeModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler, QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from kronos_mt5.brain.kronos_predictor import KronosBrain
from kronos_mt5.strategies.kronos_strategy import KronosStrategy, KronosStrategyConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass
class CostModel:
    """Realistic, mandatory cost parameters."""

    half_spread: float = 0.00005  # 0.5 pip half-spread -> 1.0 pip round-trip
    commission_usd: float = 0.50  # per-fill commission
    prob_slippage: float = 0.30  # probability a fill slips one tick
    random_seed: int = 42


@dataclass
class BacktestSetup:
    symbol: str = "EUR/USD"
    venue: str = "SIM"
    starting_balance: float = 10_000.0
    leverage: int = 30
    # strategy / brain params
    lookback: int = 256
    pred_len: int = 12
    forecast_every_n_bars: int = 8
    sample_count: int = 1
    signal_threshold_bps: float = 8.0
    risk_per_trade: float = 0.005
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    take_profit_r: float = 1.5
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_model: str = "NeoQuasar/Kronos-small"
    kronos_device: str = "cpu"
    kronos_max_context: int = 512
    seed: int | None = 42


def load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV with columns timestamps, open, high, low, close, volume."""
    df = pd.read_csv(csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
    df = df.set_index("timestamps").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _bar_type(setup: BacktestSetup) -> str:
    return f"{setup.symbol}.{setup.venue}-15-MINUTE-LAST-EXTERNAL"


def build_engine(
    ohlcv: pd.DataFrame,
    setup: BacktestSetup,
    costs: CostModel,
    brain: KronosBrain,
    *,
    trader_id: str = "BACKTESTER-001",
    log_level: str = "ERROR",
) -> tuple[BacktestEngine, Venue]:
    """Assemble a fully-configured engine (instrument, data, venue, strategy)."""
    venue = Venue(setup.venue)
    instrument = TestInstrumentProvider.default_fx_ccy(setup.symbol, venue=venue)

    # bypass_logging=True: the Rust logging subsystem can only be initialized
    # once per process, so re-using normal logging across multiple engines (as
    # walk-forward does) panics. Bypassing it keeps multi-window runs in-process.
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=trader_id,
            logging=LoggingConfig(log_level=log_level, bypass_logging=True),
        )
    )

    fill_model = FillModel(
        prob_fill_on_limit=1.0,
        prob_slippage=costs.prob_slippage,
        random_seed=costs.random_seed,
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(setup.starting_balance, USD)],
        default_leverage=Decimal(setup.leverage),
        fill_model=fill_model,
        fee_model=FixedFeeModel(Money(costs.commission_usd, USD)),
        bar_execution=False,  # execution comes from the synthesized quotes
    )
    engine.add_instrument(instrument)

    # --- data: external LAST bars (for the strategy) + spread-bearing quotes ---
    bar_type = BarType.from_str(_bar_type(setup))
    bars = BarDataWrangler(bar_type, instrument).process(ohlcv.copy())

    bid = ohlcv[["open", "high", "low", "close"]].copy()
    ask = bid.copy()
    for col in ["open", "high", "low", "close"]:
        bid[col] = ohlcv[col] - costs.half_spread
        ask[col] = ohlcv[col] + costs.half_spread
    quotes = QuoteTickDataWrangler(instrument).process_bar_data(bid_data=bid, ask_data=ask)

    engine.add_data(quotes)
    engine.add_data(bars)

    # --- strategy ---
    config = KronosStrategyConfig(
        instrument_id=instrument.id.value,
        bar_type=str(bar_type),
        lookback=setup.lookback,
        pred_len=setup.pred_len,
        forecast_every_n_bars=setup.forecast_every_n_bars,
        sample_count=setup.sample_count,
        signal_threshold_bps=setup.signal_threshold_bps,
        risk_per_trade=setup.risk_per_trade,
        atr_period=setup.atr_period,
        atr_stop_mult=setup.atr_stop_mult,
        take_profit_r=setup.take_profit_r,
    )
    engine.add_strategy(KronosStrategy(config=config, brain=brain))
    return engine, venue


def load_brain(setup: BacktestSetup) -> KronosBrain:
    return KronosBrain(
        tokenizer_name=setup.kronos_tokenizer,
        model_name=setup.kronos_model,
        device=setup.kronos_device,
        max_context=setup.kronos_max_context,
        seed=setup.seed,
    )


def equity_curve(engine: BacktestEngine, venue: Venue) -> pd.Series:
    """Extract an equity curve (account total balance over time) for analytics."""
    report = engine.trader.generate_account_report(venue)
    if report.empty:
        return pd.Series(dtype=float)
    # account report is indexed by timestamp; 'total' holds the balance string.
    col = "total" if "total" in report.columns else report.columns[-1]
    series = report[col].astype(float)
    series.index = pd.to_datetime(report.index, utc=True)
    return series[~series.index.duplicated(keep="last")].sort_index()


def returns_from_equity(curve: pd.Series, freq: str = "D") -> pd.Series:
    """Resample an equity curve to periodic returns for QuantStats."""
    if curve.empty:
        return pd.Series(dtype=float)
    resampled = curve.resample(freq).last().ffill()
    return resampled.pct_change().dropna()


def _money_to_float(value) -> float:
    """Parse NautilusTrader money strings like '72.88 USD' -> 72.88."""
    if value is None:
        return 0.0
    text = str(value).split()[0] if str(value).strip() else "0"
    try:
        return float(text)
    except ValueError:
        return 0.0


def summarize(engine: BacktestEngine, venue: Venue) -> dict:
    """Compact, cost-adjusted summary metrics from the run."""
    curve = equity_curve(engine, venue)
    rets = returns_from_equity(curve)
    positions = engine.trader.generate_positions_report()
    # Count every closed round-trip. In NETTING mode each completed round-trip is
    # recorded as a position snapshot, and only the final position is non-snapshot
    # — so we must KEEP snapshots and count rows that have actually closed.
    if "ts_closed" in positions.columns:
        positions = positions[positions["ts_closed"].notna()]

    start_bal = float(curve.iloc[0]) if not curve.empty else 0.0
    end_bal = float(curve.iloc[-1]) if not curve.empty else 0.0
    total_return = (end_bal / start_bal - 1.0) if start_bal else 0.0

    n_trades = len(positions)
    wins = 0
    gross_win = gross_loss = 0.0
    if n_trades and "realized_pnl" in positions.columns:
        pnls = positions["realized_pnl"].map(_money_to_float)
        wins = int((pnls > 0).sum())
        gross_win = float(pnls[pnls > 0].sum())
        gross_loss = float(-pnls[pnls < 0].sum())

    profit_factor = (gross_win / gross_loss) if gross_loss else float("inf") if gross_win else 0.0
    win_rate = (wins / n_trades) if n_trades else 0.0
    expectancy = ((gross_win - gross_loss) / n_trades) if n_trades else 0.0

    sharpe = 0.0
    max_dd = 0.0
    if len(rets) > 1 and rets.std() != 0:
        sharpe = float((rets.mean() / rets.std()) * (252**0.5))
    if not curve.empty:
        running_max = curve.cummax()
        max_dd = float(((curve - running_max) / running_max).min())

    return {
        "start_balance": start_bal,
        "end_balance": end_bal,
        "total_return": total_return,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_usd": expectancy,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def print_reports(engine: BacktestEngine, venue: Venue) -> None:
    """Print Nautilus' built-in reports + the compact summary."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== ACCOUNT REPORT ===")
    print(engine.trader.generate_account_report(venue).tail(10))
    print("\n=== ORDER FILLS REPORT ===")
    print(engine.trader.generate_order_fills_report())
    print("\n=== POSITIONS REPORT ===")
    print(engine.trader.generate_positions_report())
    print("\n=== SUMMARY (cost-adjusted) ===")
    for k, v in summarize(engine, venue).items():
        print(f"  {k:16s}: {v}")


def quantstats_tearsheet(
    engine: BacktestEngine, venue: Venue, out_html: str | Path, title: str = "Kronos FX"
) -> bool:
    """Write a QuantStats HTML tearsheet from the equity curve. Returns success."""
    rets = returns_from_equity(equity_curve(engine, venue))
    if len(rets) < 3:
        print(f"[tearsheet] only {len(rets)} return points — skipping QuantStats HTML.")
        return False
    try:
        import quantstats as qs

        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        qs.reports.html(rets, output=str(out_html), title=title)
        print(f"[tearsheet] wrote {out_html}")
        return True
    except Exception as exc:  # quantstats is finicky on short/degenerate series
        print(f"[tearsheet] QuantStats failed ({exc!r}); summary metrics still valid.")
        return False
