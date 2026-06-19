"""Central configuration. All tunables and secrets load from the environment
(.env). Nothing here should be hardcoded in strategy/adapter code.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("demo_only", mode="before")
    @classmethod
    def _lenient_bool(cls, v):
        """Tolerate stray chars/whitespace in .env booleans (e.g. 'true ²')."""
        if isinstance(v, str):
            return v.strip().lower().split()[0] in {"1", "true", "yes", "on"} if v.strip() else True
        return v

    # --- MT5 demo account ---
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_terminal_path: str | None = None

    # --- Instrument / timeframe ---
    trading_symbol: str = "EURUSD"
    bar_spec: str = "15-MINUTE-LAST"

    # --- Kronos brain ---
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_model: str = "NeoQuasar/Kronos-small"
    kronos_device: str = "cpu"
    kronos_max_context: int = 512
    lookback: int = 400
    pred_len: int = 12
    sample_count: int = 10
    kronos_t: float = 1.0
    kronos_top_p: float = 0.9

    # --- Signal / risk ---
    signal_threshold_bps: float = 8.0
    risk_per_trade: float = 0.005
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    take_profit_r: float = 1.5
    max_open_positions: int = 1

    # --- Binance testnet (crypto trend live-paper) ---
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_environment: str = "TESTNET"  # TESTNET only for this scaffold
    binance_symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT,LTCUSDT,LINKUSDT"
    binance_target_vol: float = 0.15
    binance_heartbeat_secs: int = 3600  # periodic equity/PnL log between daily rebalances
    binance_mark_check_secs: int = 5  # watchdog cadence for the 1s Binance mark stream
    binance_mark_stale_secs: int = 15  # fail closed when any configured mark is older
    # --- risk controls ---
    binance_stop_pct: float = 0.20  # per-position stop-loss (exchange-native, reduce-only)
    binance_use_vol_stop: bool = False
    binance_stop_vol_mult: float = 4.0
    binance_min_stop_pct: float = 0.08
    binance_max_stop_pct: float = 0.30
    binance_use_take_profit: bool = False
    binance_tp_pct: float = 0.50
    binance_use_trailing_stop: bool = False
    binance_trailing_activation_r: float = 1.0
    binance_trailing_vol_mult: float = 3.0
    binance_min_trailing_pct: float = 0.005
    binance_max_trailing_pct: float = 0.10
    binance_flatten_on_stop: bool = False  # graceful process stop should not force-exit by default
    binance_max_drawdown: float = 0.20  # portfolio kill-switch: flatten+halt below this DD
    binance_risk_check_secs: int = 5  # live mark-to-market drawdown cadence
    binance_use_correlation_scaling: bool = False
    binance_corr_window: int = 90
    binance_corr_threshold: float = 0.65
    binance_corr_min_scalar: float = 0.50
    binance_funding_filter_enabled: bool = False
    binance_funding_rate_limit: float = 0.0010
    binance_funding_refresh_secs: int = 3600
    binance_cost_aware_rebalance: bool = False
    binance_round_trip_cost_bps: float = 8.0
    binance_slippage_bps: float = 2.0
    binance_use_patient_limit: bool = False
    binance_patient_limit_offset_bps: float = 2.0
    binance_patient_limit_timeout_secs: int = 300
    binance_patient_limit_market_fallback: bool = True

    # --- companion dashboard / API / alerts ---
    companion_db: str = "logs/companion.db"
    companion_snapshot_secs: int = 60
    companion_control_secs: int = 15
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_token: str = ""  # if set, required for /kill and /resume
    api_poll_secs: int = 30  # alerter loop interval
    api_dd_warn_pct: float = 0.10  # Telegram warn when drawdown crosses this
    api_bot_down_secs: int = 600  # alert if no bot heartbeat for this long
    api_daily_summary_hour: int = 0  # UTC hour for the daily Telegram digest (<0 = off)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    def assert_binance_testnet(self) -> None:
        """Guardrail: refuse anything but the Binance TESTNET in this scaffold."""
        if self.binance_environment.upper() != "TESTNET":
            raise RuntimeError(
                f"binance_environment='{self.binance_environment}' — only TESTNET is permitted here."
            )
        if not self.binance_api_key or not self.binance_api_secret:
            raise RuntimeError(
                "Missing BINANCE_API_KEY / BINANCE_API_SECRET. Create FUTURES testnet keys at "
                "https://testnet.binancefuture.com and put them in .env."
            )

    # --- Safety ---
    demo_only: bool = True

    def assert_demo(self) -> None:
        """Guardrail: refuse to proceed unless this looks like a demo account.

        TODO(claude-code): tighten this. Most demo servers contain 'Demo' in the
        server name; some brokers expose account type via mt5.account_info().
        Fail loudly if anything looks like a live account.
        """
        if not self.demo_only:
            raise RuntimeError(
                "DEMO_ONLY is not set; live trading is not permitted by this scaffold."
            )
        if "demo" not in self.mt5_server.lower():
            raise RuntimeError(
                f"Server '{self.mt5_server}' does not look like a demo server. Aborting for safety."
            )


settings = Settings()
