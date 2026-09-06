"""Non-secret production snapshot observed read-only on 2026-09-06."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from kronos_mt5.marketdata.spec import PRODUCTION_UNIVERSE

DEPLOYED_COMMIT = "dc8a74c2a9afda9f28a0d8b2f7a1bf6044df35b2"
DEPLOYED_STRATEGY_SHA256 = "33a8821d843d2b8c472c6f1a832b6d66fd297bd90ef9ff81fe8293c70eab085d"
DEPLOYED_RISK_SHA256 = "17330fd3eda6a622d29ef17d4a597ed1deb7198049346b286b3dffcab0f2e31a"

AUDIT_PROVENANCE = {
    "summary_sha256": "5116d73288b41a311875e5d0edba029b111d94d023960cd698c4530786d383bb",
    "export_collected_utc": "2026-09-05T19:16:21Z",
    "source_commit": DEPLOYED_COMMIT,
    "environment": "TESTNET",
    "execution_telemetry_available": False,
    "missing_commission_rows": 29,
    "fills": 460,
    "cost_decision": "Incomplete testnet commission/liquidity telemetry cannot identify a live fee tier; use explicit conservative 5bps per side, no discounts.",
}

# Effective non-instrument strategy configuration observed on the deployed bot.
# This includes defaults which the live runner did not explicitly override. The
# research adapter changes only ``warmup_request`` for offline startup.
DEPLOYED_STRATEGY_PARAMETERS = {
    "lookbacks": (21, 63, 126, 252),
    "vol_window": 33,
    "target_vol": 0.15,
    "max_leverage": 2.0,
    "ppy": 365,
    "vol_floor": 0.02,
    "rebalance_threshold": 0.10,
    "warmup_request": True,
    "use_stop_loss": True,
    "stop_pct": 0.20,
    "use_vol_stop": True,
    "stop_vol_mult": 4.0,
    "min_stop_pct": 0.08,
    "max_stop_pct": 0.30,
    "use_take_profit": False,
    "tp_pct": 0.50,
    "use_trailing_stop": False,
    "trailing_activation_r": 1.0,
    "trailing_vol_mult": 3.0,
    "min_trailing_pct": 0.005,
    "max_trailing_pct": 0.10,
    "flatten_on_stop": False,
    "use_correlation_scaling": False,
    "corr_window": 90,
    "corr_threshold": 0.65,
    "corr_min_scalar": 0.50,
    "use_portfolio_allocator": True,
    "portfolio_target_vol": 0.10,
    "portfolio_cov_window": 90,
    "portfolio_cov_shrinkage": 0.25,
    "portfolio_min_scale": 0.25,
    "portfolio_max_scale": 1.25,
    "portfolio_max_gross": 1.50,
    "portfolio_min_observations": 30,
    "portfolio_scale_up_alpha": 0.20,
    "portfolio_scale_deadband": 0.05,
    "funding_filter_enabled": True,
    "funding_rate_limit": 0.001,
    "funding_continuous_sizing": False,
    "funding_rate_soft_limit": 0.0001,
    "funding_min_scalar": 0.0,
    "cost_aware_rebalance": True,
    "round_trip_cost_bps": 8.0,
    "slippage_bps": 2.0,
    "cost_threshold_mult": 4.0,
    "min_notional_buffer": 1.05,
    "use_patient_limit": True,
    "patient_limit_offset_bps": 2.0,
    "patient_limit_timeout_secs": 300,
    "patient_limit_market_fallback": True,
    "shadow_enabled": True,
    "shadow_lookbacks": (7, 21, 63, 126),
}


def research_strategy_parameters() -> dict:
    """Return deployed parameters with the single offline-startup override."""
    parameters = deepcopy(DEPLOYED_STRATEGY_PARAMETERS)
    parameters["warmup_request"] = False
    return parameters


@dataclass(frozen=True)
class BaselineConfig:
    symbols: tuple[str, ...] = PRODUCTION_UNIVERSE
    starting_equity: float = 100_000.0
    leverage: float = 2.0
    commission_bps: float = 5.0  # conservative per-side taker approximation, no discounts
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    max_drawdown: float = 0.20
    holdout_start: str = "2026-01-01"
    window_days: int = 90
    funding_mode: str = "historical"  # or explicitly incomplete smoke/sensitivity mode "omit"
    execution: str = "next_open_taker"
    bar_path: str = "OHLC"

    def __post_init__(self):
        if (
            not self.symbols
            or len(set(self.symbols)) != len(self.symbols)
            or any(s not in PRODUCTION_UNIVERSE for s in self.symbols)
        ):
            raise ValueError("baseline symbols must be a unique subset of the deployed universe")
        for name in (
            "starting_equity",
            "leverage",
            "commission_bps",
            "half_spread_bps",
            "slippage_bps",
            "max_drawdown",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.leverage > 2 or self.max_drawdown != 0.2 or self.holdout_start != "2026-01-01":
            raise ValueError("baseline risk cap and final holdout boundary are frozen")
        if self.window_days < 1 or self.funding_mode not in {"historical", "omit"}:
            raise ValueError("invalid windows or funding mode")
        if self.execution != "next_open_taker" or self.bar_path not in {"OHLC", "OLHC"}:
            raise ValueError("unsupported execution model")

    def payload(self) -> dict:
        return {
            "schema_version": 1,
            "replay": asdict(self),
            "strategy": deepcopy(DEPLOYED_STRATEGY_PARAMETERS),
            "audit_provenance": deepcopy(AUDIT_PROVENANCE),
            "deployed_commit": DEPLOYED_COMMIT,
            "deployed_strategy_sha256": DEPLOYED_STRATEGY_SHA256,
            "deployed_risk_sha256": DEPLOYED_RISK_SHA256,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def read(cls, path: Path) -> BaselineConfig:
        payload = json.loads(path.read_text())
        config = dict(payload["replay"])
        config["symbols"] = tuple(config["symbols"])
        result = cls(**config)
        if json.loads(json.dumps(result.payload())) != payload:
            raise ValueError("configuration does not match frozen deployed strategy snapshot")
        return result
