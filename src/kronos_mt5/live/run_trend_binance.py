"""Live-paper runner: crypto trend-following on the Binance USDT-M FUTURES TESTNET.

This is the realistic finish line for the validated strategy (backtest/trend.py
-> backtest/trend_engine.py). It uses NautilusTrader's maintained Binance adapter
on the **testnet** (paper money) — no real funds, no custom MT5 adapter needed.

Guardrails:
  - TESTNET only. Refuses to start on LIVE (see settings.assert_binance_testnet).
  - Reads keys from .env (BINANCE_API_KEY / BINANCE_API_SECRET), never committed.

Setup:
  1. Create FUTURES testnet API keys at https://testnet.binancefuture.com
     (log in with GitHub/Google -> API Key panel at the bottom).
  2. Put them in .env:
       BINANCE_API_KEY=...
       BINANCE_API_SECRET=...
  3. Run:  python -m kronos_mt5.live.run_trend_binance
"""
from __future__ import annotations

from datetime import timedelta

from nautilus_trader.adapters.binance import (
    BinanceAccountType,
    BinanceDataClientConfig,
    BinanceExecClientConfig,
    BinanceLiveDataClientFactory,
    BinanceLiveExecClientFactory,
)
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from config.settings import settings
from kronos_mt5.companion.recorder import Companion, CompanionConfig
from kronos_mt5.strategies.risk_manager import RiskManager, RiskManagerConfig, RiskState
from kronos_mt5.strategies.trend_strategy import TrendStrategy, TrendStrategyConfig

VENUE = "BINANCE"


class HeartbeatConfig(StrategyConfig, frozen=True):
    venue: str
    instrument_ids: tuple[str, ...]
    interval_secs: int = 3600
    equity_csv: str = "logs/live_equity.csv"  # appended each beat for later review


class Heartbeat(Strategy):
    """Periodic equity / position / PnL log + equity CSV, so a multi-month
    forward test leaves a reviewable record between the daily rebalances."""

    def __init__(self, config: HeartbeatConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue)
        self._iids = [InstrumentId.from_str(i) for i in config.instrument_ids]

    def on_start(self) -> None:
        self.clock.set_timer(
            name="heartbeat",
            interval=timedelta(seconds=self.config.interval_secs),
            callback=self._beat,
        )

    def _beat(self, event) -> None:  # noqa: ANN001 (TimeEvent)
        acct = self.portfolio.account(self._venue)
        if acct is None:
            self.log.warning("heartbeat: no account yet")
            return
        cash_money = acct.balance_total(USDT)
        cash = cash_money.as_double() if cash_money is not None else 0.0
        upnl_total = 0.0
        n_open = 0
        open_lines = []
        for iid in self._iids:
            pos = self.portfolio.net_position(iid)
            if pos == 0:
                continue
            n_open += 1
            upnl = self.portfolio.unrealized_pnl(iid)
            u = upnl.as_double() if upnl is not None else 0.0
            upnl_total += u
            open_lines.append(f"{iid.symbol}={float(pos):+.4f}({u:+.0f})")
        equity = cash + upnl_total
        self.log.info(
            f"HEARTBEAT equity=${equity:,.0f} cash=${cash:,.0f} uPnL=${upnl_total:+,.0f} "
            f"| open[{n_open}]: {', '.join(open_lines) or 'flat'}"
        )
        self._append_equity_csv(equity, cash, upnl_total, n_open)

    def _append_equity_csv(self, equity: float, cash: float, upnl: float, n_open: int) -> None:
        try:
            from datetime import datetime, timezone
            from pathlib import Path

            path = Path(self.config.equity_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a") as f:
                if new:
                    f.write("timestamp,equity,cash,unrealized_pnl,n_open\n")
                ts = datetime.now(timezone.utc).isoformat()
                f.write(f"{ts},{equity:.2f},{cash:.2f},{upnl:.2f},{n_open}\n")
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"equity csv append failed: {exc!r}")


def _instrument_ids(symbols: list[str]) -> list[str]:
    # USDT-M perpetual futures instrument ids, e.g. "BTCUSDT-PERP.BINANCE"
    return [f"{s}-PERP.{VENUE}" for s in symbols]


def build_node() -> TradingNode:
    settings.assert_binance_testnet()  # refuse non-testnet / missing keys
    symbols = [s.strip() for s in settings.binance_symbols.split(",") if s.strip()]
    iids = _instrument_ids(symbols)
    provider = InstrumentProviderConfig(load_ids=frozenset(iids))

    common = dict(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        account_type=BinanceAccountType.USDT_FUTURES,
        environment=BinanceEnvironment.TESTNET,
        instrument_provider=provider,
    )
    config = TradingNodeConfig(
        trader_id=TraderId("TREND-TESTNET-001"),
        logging=LoggingConfig(
            log_level="INFO",                # stdout (captured by systemd/nohup)
            log_level_file="INFO",
            log_directory="logs",            # persistent, rotating engine logs
            log_file_name="kronos_trend",
            log_file_max_size=100_000_000,   # 100 MB per file
            log_file_max_backup_count=10,
        ),
        data_clients={VENUE: BinanceDataClientConfig(**common)},
        exec_clients={VENUE: BinanceExecClientConfig(**common)},
    )

    node = TradingNode(config=config)
    node.add_data_client_factory(VENUE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(VENUE, BinanceLiveExecClientFactory)
    node.build()

    n = len(symbols)
    risk = RiskState()  # shared halt flag (drawdown kill-switch -> blocks entries)
    for iid in iids:
        strat = TrendStrategy(
            TrendStrategyConfig(
                instrument_id=iid,
                bar_type=f"{iid}-1-DAY-LAST-EXTERNAL",
                n_instruments=n,
                target_vol=settings.binance_target_vol,
                ppy=365,
                use_stop_loss=True,
                stop_pct=settings.binance_stop_pct,
                use_take_profit=settings.binance_use_take_profit,
                tp_pct=settings.binance_tp_pct,
            )
        )
        strat.risk_state = risk
        node.trader.add_strategy(strat)

    rm = RiskManager(
        RiskManagerConfig(
            venue=VENUE,
            instrument_ids=tuple(iids),
            max_drawdown=settings.binance_max_drawdown,
            check_secs=settings.binance_risk_check_secs,
        )
    )
    rm.risk_state = risk
    node.trader.add_strategy(rm)

    # companion bridge: writes state/equity/fills to SQLite + polls remote kill
    companion = Companion(
        CompanionConfig(
            venue=VENUE,
            instrument_ids=tuple(iids),
            db_path=settings.companion_db,
            snapshot_secs=settings.companion_snapshot_secs,
            control_secs=settings.companion_control_secs,
        )
    )
    companion.risk_state = risk
    node.trader.add_strategy(companion)

    node.trader.add_strategy(
        Heartbeat(
            HeartbeatConfig(
                venue=VENUE,
                instrument_ids=tuple(iids),
                interval_secs=settings.binance_heartbeat_secs,
            )
        )
    )
    return node


def main() -> None:
    print("Starting crypto trend on Binance FUTURES TESTNET (paper)...")
    node = build_node()
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()
