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

import json
import urllib.parse
import urllib.request
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
from nautilus_trader.portfolio.config import PortfolioConfig
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from config.settings import settings
from kronos_mt5.companion import store
from kronos_mt5.companion.accounting import IncomePoller, IncomePollerConfig
from kronos_mt5.companion.recorder import Companion, CompanionConfig
from kronos_mt5.strategies.risk_manager import (
    MarkPriceMonitor,
    MarkPriceMonitorConfig,
    RiskManager,
    RiskManagerConfig,
    RiskState,
)
from kronos_mt5.strategies.trend_strategy import TrendStrategy, TrendStrategyConfig

VENUE = "BINANCE"
PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
TESTNET_FUTURES_REST_URL = "https://demo-fapi.binance.com"


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


class FundingRateState:
    def __init__(self) -> None:
        self.rates: dict[str, float] = {}

    def get(self, symbol: str) -> float | None:
        return self.rates.get(symbol)


class FundingRateUpdaterConfig(StrategyConfig, frozen=True):
    symbols: tuple[str, ...]
    interval_secs: int = 3600


class FundingRateUpdater(Strategy):
    def __init__(self, config: FundingRateUpdaterConfig, state: FundingRateState) -> None:
        super().__init__(config)
        # NB: not `self.state` — Component has a read-only `.state` (lifecycle) property.
        self.funding_state = state

    def on_start(self) -> None:
        self._refresh(None)
        self.clock.set_timer(
            name="funding_rates",
            interval=timedelta(seconds=self.config.interval_secs),
            callback=self._refresh,
        )

    def _refresh(self, event) -> None:  # noqa: ANN001
        for symbol in self.config.symbols:
            try:
                params = urllib.parse.urlencode({"symbol": symbol})
                with urllib.request.urlopen(
                    f"{PREMIUM_INDEX_URL}?{params}", timeout=10
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.funding_state.rates[symbol] = float(payload["lastFundingRate"])
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"funding refresh failed for {symbol}: {exc!r}")


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
            log_level="INFO",  # stdout (captured by systemd/nohup)
            log_level_file="INFO",
            log_directory="logs",  # persistent, rotating engine logs
            log_file_name="kronos_trend",
            log_file_max_size=100_000_000,  # 100 MB per file
            log_file_max_backup_count=10,
        ),
        # MarkPriceUpdate events now drive Portfolio.unrealized_pnl/net exposure.
        # Without this, the daily bars leave live PnL and drawdown stale all day.
        portfolio=PortfolioConfig(use_mark_prices=True),
        data_clients={VENUE: BinanceDataClientConfig(**common)},
        exec_clients={VENUE: BinanceExecClientConfig(**common)},
    )

    node = TradingNode(config=config)
    node.add_data_client_factory(VENUE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(VENUE, BinanceLiveExecClientFactory)
    node.build()

    n = len(symbols)
    risk = RiskState()  # shared halt flag (drawdown kill-switch -> blocks entries)
    if settings.binance_use_portfolio_allocator:
        try:
            store.init_db(settings.companion_db)
            allocator_context = "|".join(
                (
                    ",".join(symbols),
                    f"target={settings.binance_portfolio_target_vol:.12g}",
                    f"window={settings.binance_portfolio_cov_window}",
                    f"shrink={settings.binance_portfolio_cov_shrinkage:.12g}",
                    f"gross={settings.binance_portfolio_max_gross:.12g}",
                )
            )
            stored_context = store.get_kv(
                "portfolio_allocator_context", "", settings.companion_db
            )
            restored_scale = (
                float(
                    store.get_kv(
                        "portfolio_allocator_scale", "1.0", settings.companion_db
                    )
                    or 1.0
                )
                if stored_context == allocator_context
                else 1.0
            )
            risk.allocation_scales["live_allocator"] = max(0.0, restored_scale)
            store.set_kv(
                "portfolio_allocator_context", allocator_context, settings.companion_db
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Allocator scale restore skipped: {exc!r}")
    funding = FundingRateState()
    mark_monitor = MarkPriceMonitor(
        MarkPriceMonitorConfig(
            instrument_ids=tuple(iids),
            check_secs=settings.binance_mark_check_secs,
            stale_after_secs=settings.binance_mark_stale_secs,
            db_path=settings.companion_db,
        )
    )
    mark_monitor.risk_state = risk
    node.trader.add_strategy(mark_monitor)
    if settings.binance_funding_filter_enabled or settings.binance_shadow_enabled:
        node.trader.add_strategy(
            FundingRateUpdater(
                FundingRateUpdaterConfig(
                    symbols=tuple(symbols),
                    interval_secs=settings.binance_funding_refresh_secs,
                ),
                funding,
            )
        )
    for iid in iids:
        strat = TrendStrategy(
            TrendStrategyConfig(
                instrument_id=iid,
                bar_type=f"{iid}-1-DAY-LAST-EXTERNAL",
                portfolio_instrument_ids=tuple(iids),
                n_instruments=n,
                target_vol=settings.binance_target_vol,
                ppy=365,
                use_stop_loss=True,
                stop_pct=settings.binance_stop_pct,
                use_vol_stop=settings.binance_use_vol_stop,
                stop_vol_mult=settings.binance_stop_vol_mult,
                min_stop_pct=settings.binance_min_stop_pct,
                max_stop_pct=settings.binance_max_stop_pct,
                use_take_profit=settings.binance_use_take_profit,
                tp_pct=settings.binance_tp_pct,
                use_trailing_stop=settings.binance_use_trailing_stop,
                trailing_activation_r=settings.binance_trailing_activation_r,
                trailing_vol_mult=settings.binance_trailing_vol_mult,
                min_trailing_pct=settings.binance_min_trailing_pct,
                max_trailing_pct=settings.binance_max_trailing_pct,
                flatten_on_stop=settings.binance_flatten_on_stop,
                use_correlation_scaling=settings.binance_use_correlation_scaling,
                corr_window=settings.binance_corr_window,
                corr_threshold=settings.binance_corr_threshold,
                corr_min_scalar=settings.binance_corr_min_scalar,
                use_portfolio_allocator=settings.binance_use_portfolio_allocator,
                portfolio_target_vol=settings.binance_portfolio_target_vol,
                portfolio_cov_window=settings.binance_portfolio_cov_window,
                portfolio_cov_shrinkage=settings.binance_portfolio_cov_shrinkage,
                portfolio_min_scale=settings.binance_portfolio_min_scale,
                portfolio_max_scale=settings.binance_portfolio_max_scale,
                portfolio_max_gross=settings.binance_portfolio_max_gross,
                portfolio_min_observations=settings.binance_portfolio_min_observations,
                portfolio_scale_up_alpha=settings.binance_portfolio_scale_up_alpha,
                portfolio_scale_deadband=settings.binance_portfolio_scale_deadband,
                funding_filter_enabled=settings.binance_funding_filter_enabled,
                funding_rate_limit=settings.binance_funding_rate_limit,
                funding_continuous_sizing=settings.binance_funding_continuous_sizing,
                funding_rate_soft_limit=settings.binance_funding_rate_soft_limit,
                funding_min_scalar=settings.binance_funding_min_scalar,
                cost_aware_rebalance=settings.binance_cost_aware_rebalance,
                round_trip_cost_bps=settings.binance_round_trip_cost_bps,
                slippage_bps=settings.binance_slippage_bps,
                use_patient_limit=settings.binance_use_patient_limit,
                patient_limit_offset_bps=settings.binance_patient_limit_offset_bps,
                patient_limit_timeout_secs=settings.binance_patient_limit_timeout_secs,
                patient_limit_market_fallback=settings.binance_patient_limit_market_fallback,
                shadow_enabled=settings.binance_shadow_enabled,
                shadow_lookbacks=tuple(
                    int(value.strip())
                    for value in settings.binance_shadow_lookbacks.split(",")
                    if value.strip()
                ),
            )
        )
        strat.risk_state = risk
        strat.funding_rates = funding
        node.trader.add_strategy(strat)

    rm = RiskManager(
        RiskManagerConfig(
            venue=VENUE,
            instrument_ids=tuple(iids),
            max_drawdown=settings.binance_max_drawdown,
            check_secs=settings.binance_risk_check_secs,
            db_path=settings.companion_db,
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
            corr_window=settings.binance_corr_window,
            corr_threshold=settings.binance_corr_threshold,
            corr_min_scalar=settings.binance_corr_min_scalar,
        )
    )
    companion.risk_state = risk
    node.trader.add_strategy(companion)

    income_poller = IncomePoller(
        IncomePollerConfig(
            db_path=settings.companion_db,
            base_url=TESTNET_FUTURES_REST_URL,
            interval_secs=settings.binance_income_poll_secs,
        )
    )
    income_poller.api_key = settings.binance_api_key
    income_poller.api_secret = settings.binance_api_secret
    node.trader.add_strategy(income_poller)

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
