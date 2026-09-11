"""Phase 4A experiment registry.

Every experiment is declared here before it is run, with its hypothesis, its
eligibility, its exact parameters and a deterministic fingerprint. Nothing in
this module tunes a parameter: moving-average length, volatility threshold,
entry scale, stops and the symbol universe are all fixed constants.

Eligibility:

``candidate``
    Predeclared hypothesis that may be scored against the acceptance gate.
``diagnostic``
    Explanatory only. Never eligible for strategy selection, typically because
    it was constructed after observing results and is therefore post-hoc.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal

from kronos_mt5.baseline.config import BaselineConfig
from kronos_mt5.marketdata.spec import PRODUCTION_UNIVERSE

from .overlays import (
    CompositeOverlay,
    LongOnlyOverlay,
    NormalVolEntryGateOverlay,
    Overlay,
    ShortBearishRegimeOverlay,
)

CANDIDATE = "candidate"
DIAGNOSTIC = "diagnostic"

#: Cost multipliers applied to commission, half-spread and slippage together.
#: Funding is historical and is never reduced or scaled away.
COST_MULTIPLIERS = ("1.0", "1.5", "2.0")
#: Assumed intraday path variants.
BAR_PATHS = ("OHLC", "OLHC")


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    description: str
    hypothesis: str
    eligibility: str
    rules: tuple[str, ...]
    overlay_factory: object = None
    symbols: tuple[str, ...] = PRODUCTION_UNIVERSE
    post_hoc: bool = False
    post_hoc_note: str = ""
    parameters: dict = field(default_factory=dict)

    def overlay(self) -> Overlay | None:
        return None if self.overlay_factory is None else self.overlay_factory()

    def config(self, *, bar_path: str = "OHLC", cost_multiplier: str = "1.0") -> BaselineConfig:
        """Frozen baseline configuration with only path and cost stress varied."""

        base = BaselineConfig()
        multiplier = Decimal(cost_multiplier)
        if multiplier <= 0:
            raise ValueError("cost multiplier must be positive")
        return BaselineConfig(
            symbols=self.symbols,
            starting_equity=base.starting_equity,
            leverage=base.leverage,
            commission_bps=float(Decimal(str(base.commission_bps)) * multiplier),
            half_spread_bps=float(Decimal(str(base.half_spread_bps)) * multiplier),
            slippage_bps=float(Decimal(str(base.slippage_bps)) * multiplier),
            max_drawdown=base.max_drawdown,
            holdout_start=base.holdout_start,
            window_days=base.window_days,
            funding_mode=base.funding_mode,
            execution=base.execution,
            bar_path=bar_path,
        )

    def identity(self) -> dict:
        overlay = self.overlay()
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "eligibility": self.eligibility,
            "rules": list(self.rules),
            "parameters": dict(self.parameters),
            "symbols": list(self.symbols),
            "post_hoc": self.post_hoc,
            "post_hoc_note": self.post_hoc_note,
            "overlay": None if overlay is None else overlay.identity(),
            "cost_multipliers": list(COST_MULTIPLIERS),
            "bar_paths": list(BAR_PATHS),
        }

    def fingerprint(self) -> str:
        """Deterministic identity of everything that defines this experiment."""

        return hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


CONTROL = Experiment(
    experiment_id="control-corrected-baseline",
    description="The corrected fixed-parameter deployed baseline, unmodified.",
    hypothesis="Control. No hypothesis is tested; this is the immutable reference.",
    eligibility=DIAGNOSTIC,
    rules=(
        (
            "No overlay is installed; the pinned dc8a74c decision, sizing, stop and "
            "allocator logic executes exactly as in the corrected baseline."
        ),
    ),
)

LONG_ONLY = Experiment(
    experiment_id="long-only",
    description="Remove all short exposure while leaving long behaviour untouched.",
    hypothesis=(
        "The deployed long trend component contains the usable edge, while shorts "
        "destroy most performance."
    ),
    eligibility=CANDIDATE,
    rules=(
        "Existing long signals, sizing, stops and allocator are preserved.",
        "No order may open or increase short exposure.",
        "Existing short exposure may only be reduced or closed.",
        "A negative signal is never reinterpreted as a long signal.",
    ),
    overlay_factory=LongOnlyOverlay,
)

SHORT_BEARISH_REGIME = Experiment(
    experiment_id="short-bearish-regime",
    description="Allow shorts only under a confirmed slow bearish trend.",
    hypothesis="Shorts work only when the asset has a confirmed slow bearish trend.",
    eligibility=CANDIDATE,
    rules=(
        "Long behaviour is unchanged.",
        (
            "A short is permitted only when the previous fully closed daily candle is "
            "below its causal 200-day moving average."
        ),
        "No current-day close or future information is used.",
        "The 200-day window is fixed and is not optimised in this PR.",
        "Existing shorts can always be reduced or closed.",
    ),
    overlay_factory=ShortBearishRegimeOverlay,
    parameters={"moving_average_days": 200, "optimised": False},
)

NORMAL_VOL_ENTRY_GATE = Experiment(
    experiment_id="normal-vol-entry-gate",
    description="Block new or increasing exposure while the pinned regime is HIGH_VOL.",
    hypothesis="New exposure during the existing HIGH_VOL regime creates negative expectancy.",
    eligibility=CANDIDATE,
    rules=(
        "Only new or increasing exposure is blocked during HIGH_VOL.",
        "Reductions, exits, stops and liquidation are always permitted.",
        "The existing regime classifier is neither redefined nor tuned.",
    ),
    overlay_factory=lambda: NormalVolEntryGateOverlay("0"),
    parameters={"new_exposure_scale": "0", "optimised": False},
)

NORMAL_VOL_ENTRY_SCALED = Experiment(
    experiment_id="normal-vol-entry-gate-scaled",
    description="Scale new or increasing exposure to 50% while the regime is HIGH_VOL.",
    hypothesis=(
        "Halving rather than blocking new HIGH_VOL exposure retains participation "
        "while reducing the negative expectancy."
    ),
    eligibility=CANDIDATE,
    rules=(
        "Only new or increasing exposure is scaled during HIGH_VOL.",
        "The 50% scale is a fixed predeclared value and is not optimised.",
        "Reductions, exits, stops and liquidation are always permitted.",
        "The existing regime classifier is neither redefined nor tuned.",
    ),
    overlay_factory=lambda: NormalVolEntryGateOverlay("0.5"),
    parameters={"new_exposure_scale": "0.5", "optimised": False},
)

LONG_ONLY_NORMAL_VOL = Experiment(
    experiment_id="long-only-normal-vol",
    description="Predeclared combination of the long-only rule and the HIGH_VOL entry gate.",
    hypothesis=(
        "Removing shorts and refusing new HIGH_VOL exposure are complementary; "
        "combining them improves risk-adjusted development performance."
    ),
    eligibility=CANDIDATE,
    rules=(
        "Applies the long-only rule and the HIGH_VOL entry gate together.",
        (
            "Declared before any experiment result was inspected; it is not a "
            "combination chosen after seeing outcomes."
        ),
        "Reductions, exits, stops and liquidation are always permitted.",
    ),
    overlay_factory=lambda: CompositeOverlay(
        "long-only-normal-vol",
        "long-only short block composed with the HIGH_VOL new-exposure block",
        LongOnlyOverlay(),
        NormalVolEntryGateOverlay("0"),
    ),
)

EXCLUDE_LTC_DIAGNOSTIC = Experiment(
    experiment_id="exclude-ltc-diagnostic",
    description="Remove LTCUSDT from the replay universe. DIAGNOSTIC ONLY.",
    hypothesis=(
        "How much of the baseline's loss is attributable to LTCUSDT alone? "
        "This is an attribution question, not a deployable strategy rule."
    ),
    eligibility=DIAGNOSTIC,
    rules=(
        "LTCUSDT is removed from the replay universe.",
        "No other behaviour changes.",
    ),
    symbols=tuple(s for s in PRODUCTION_UNIVERSE if s != "LTCUSDT"),
    post_hoc=True,
    post_hoc_note=(
        "POST-HOC AND POTENTIALLY OVERFIT. LTCUSDT was selected for removal only "
        "after observing its poor historical performance in the corrected "
        "baseline. Dropping the worst historical symbol is guaranteed to improve "
        "an in-sample result and carries no evidence about future performance. "
        "This experiment is INELIGIBLE for strategy selection and must not be "
        "used to justify deployment."
    ),
)

EXPERIMENTS: tuple[Experiment, ...] = (
    CONTROL,
    LONG_ONLY,
    SHORT_BEARISH_REGIME,
    NORMAL_VOL_ENTRY_GATE,
    NORMAL_VOL_ENTRY_SCALED,
    LONG_ONLY_NORMAL_VOL,
    EXCLUDE_LTC_DIAGNOSTIC,
)

BY_ID = {e.experiment_id: e for e in EXPERIMENTS}

#: Every hypothesis tested in this PR, counted for multiple-comparison honesty.
HYPOTHESIS_COUNT = sum(1 for e in EXPERIMENTS if e.experiment_id != CONTROL.experiment_id)


def get(experiment_id: str) -> Experiment:
    if experiment_id not in BY_ID:
        raise KeyError(f"unknown experiment: {experiment_id}")
    return BY_ID[experiment_id]
