"""Phase 4A controlled strategy-experiment harness (research only).

Nothing in this package changes a production file, a deployed parameter or a
live service, and no code path here can evaluate the final 2026 holdout.
"""

from .gates import FAILED, INELIGIBLE, PASSED
from .overlays import (
    CompositeOverlay,
    LongOnlyOverlay,
    NormalVolEntryGateOverlay,
    Overlay,
    ShortBearishRegimeOverlay,
)
from .registry import CANDIDATE, DIAGNOSTIC, EXPERIMENTS, Experiment, get

__all__ = [
    "CANDIDATE",
    "DIAGNOSTIC",
    "EXPERIMENTS",
    "FAILED",
    "INELIGIBLE",
    "PASSED",
    "CompositeOverlay",
    "Experiment",
    "LongOnlyOverlay",
    "NormalVolEntryGateOverlay",
    "Overlay",
    "ShortBearishRegimeOverlay",
    "get",
]
