"""Research-only decision overlays for Phase 4A strategy experiments.

An overlay may only **veto or reduce** a pinned deployed decision at the
executable next open. It never invents a decision, never flips a side, never
resizes a permitted order upward, and never touches signal, sizing, stop or
allocator logic. The frozen ``dc8a74c`` strategy and risk snapshots are
untouched; the pinned baseline installs no overlay at all.

Position convention: ``current`` is the signed net position and ``quantity`` is
the unsigned order size. ``target = current + (quantity if BUY else -quantity)``.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import ClassVar

from nautilus_trader.model.enums import OrderSide

HIGH_VOL = "HIGH_VOL"
BEARISH_MA_WINDOW = 200


def _decimal(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class Overlay:
    """Base overlay: passes every pinned decision through unchanged."""

    overlay_id = "control"
    #: Human-readable statement of what the overlay changes.
    changed_behavior = "none: the pinned deployed decision is executed unchanged"
    parameters: ClassVar[dict] = {}

    def adjust(self, strategy, side, quantity, current):
        """Return ``(side, quantity, veto_reason)`` for an executable decision."""

        return side, quantity, None

    def identity(self) -> dict:
        return {
            "overlay_id": self.overlay_id,
            "changed_behavior": self.changed_behavior,
            "parameters": dict(self.parameters),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _target(side, quantity: Decimal, current: Decimal) -> Decimal:
    return current + (quantity if side == OrderSide.BUY else -quantity)


def _reduces_short(current: Decimal, target: Decimal) -> bool:
    """True when an order makes an existing short position less negative."""

    return current < 0 and target > current


class LongOnlyOverlay(Overlay):
    """Never hold short exposure; long behaviour is untouched.

    A reversal order that would carry a long straight through flat into a short
    is reduced to exactly the size that closes the long, so the deployed exit
    still happens. A negative signal is never reinterpreted as a long: the
    overlay only ever removes short exposure, never adds long exposure.
    """

    overlay_id = "long-only"
    changed_behavior = (
        "block any order that opens or increases short exposure; a long-to-short "
        "reversal is reduced to exactly close the long; short reductions and exits "
        "are always allowed"
    )

    veto_reason = "long_only_short_blocked"

    def adjust(self, strategy, side, quantity, current):
        quantity, current = _decimal(quantity), _decimal(current)
        target = _target(side, quantity, current)
        if target >= 0:
            return side, quantity, None
        if _reduces_short(current, target):
            return side, quantity, None
        if current > 0:
            # Close the long exactly; never flip through flat into a short.
            return side, current, None
        return side, quantity, self.veto_reason


class ShortBearishRegimeOverlay(LongOnlyOverlay):
    """Permit shorts only under a confirmed slow bearish trend.

    The gate is the pinned deployed strategy's own closed-bar buffer: the last
    fully closed daily candle must sit below the mean of the last
    ``BEARISH_MA_WINDOW`` fully closed daily closes. The buffer is appended on
    bar close and read at the following executable open, so no current-day close
    or future observation is used. The window is fixed and is not optimised.
    """

    overlay_id = "short-bearish-regime"
    changed_behavior = (
        "shorts are permitted only when the previous fully closed daily candle is "
        "below its causal 200-day moving average; long behaviour is unchanged; "
        "short reductions and exits are always allowed"
    )
    parameters: ClassVar[dict] = {
        "moving_average_days": BEARISH_MA_WINDOW,
        "optimised": False,
    }

    veto_reason = "short_blocked_not_bearish"

    @staticmethod
    def is_bearish(strategy) -> bool:
        """True when the last closed candle is below its causal 200-day mean."""

        closes = list(strategy._closes)
        if len(closes) < BEARISH_MA_WINDOW:
            return False  # unconfirmed trend is never treated as bearish
        window = closes[-BEARISH_MA_WINDOW:]
        return closes[-1] < sum(window) / BEARISH_MA_WINDOW

    def adjust(self, strategy, side, quantity, current):
        quantity, current = _decimal(quantity), _decimal(current)
        target = _target(side, quantity, current)
        if target >= 0 or _reduces_short(current, target):
            return side, quantity, None
        if self.is_bearish(strategy):
            return side, quantity, None
        return super().adjust(strategy, side, quantity, current)


class NormalVolEntryGateOverlay(Overlay):
    """Restrict only *new or increasing* exposure while the regime is HIGH_VOL.

    Reductions, exits, stops and liquidation are always permitted, in either
    direction. The regime comes from the pinned classifier via
    ``strategy.current_regime()``; it is neither redefined nor tuned here.
    With ``scale`` at 0 an increase is blocked outright; with ``scale`` at 0.5
    only half of the requested increase is executed. Both values are fixed.
    """

    overlay_id = "normal-vol-entry-gate"
    veto_reason = "high_vol_entry_blocked"

    def __init__(self, scale: str | Decimal = "0"):
        self.scale = _decimal(scale)
        if not (0 <= self.scale < 1):
            raise ValueError("high-volatility entry scale must be in [0, 1)")
        self.parameters = {
            "regime_blocked": HIGH_VOL,
            "new_exposure_scale": str(self.scale),
            "classifier": "pinned deployed correlation_metrics(90, 0.65, 0.5)",
            "optimised": False,
        }
        self.changed_behavior = (
            f"while the pinned classifier reports {HIGH_VOL}, new or increasing "
            f"exposure is scaled to {self.scale} of the requested increase; "
            "reductions, exits, stops and liquidation are always allowed"
        )

    def adjust(self, strategy, side, quantity, current):
        quantity, current = _decimal(quantity), _decimal(current)
        target = _target(side, quantity, current)
        if strategy.current_regime() != HIGH_VOL:
            return side, quantity, None

        if target == 0:
            # Exact flattening never creates exposure.
            return side, quantity, None
        if current != 0 and (current > 0) == (target > 0):
            if abs(target) <= abs(current):
                # Same-side reduction (including an order that remains open).
                return side, quantity, None
            # Scale only the same-side increase above the current exposure.
            permitted_exposure = abs(current) + (abs(target) - abs(current)) * self.scale
            permitted_target = permitted_exposure if target > 0 else -permitted_exposure
        elif current != 0:
            # A cross-zero order first closes the current position. That portion
            # is always permitted; only the requested exposure beyond flat is
            # new exposure and therefore scaled.
            permitted_target = target * self.scale
        else:
            # From flat, the entire requested target is new exposure.
            permitted_target = target * self.scale

        delta = permitted_target - current
        if delta == 0:
            return side, quantity, self.veto_reason
        if (delta > 0) != (side == OrderSide.BUY):
            # Scaling must never reverse the pinned decision's direction.
            return side, quantity, self.veto_reason
        permitted_quantity = abs(delta)
        if permitted_quantity > quantity:
            raise ValueError("HIGH_VOL gate must never increase the requested quantity")
        return side, permitted_quantity, None


class CompositeOverlay(Overlay):
    """Apply overlays in a fixed, predeclared order; any veto stops the chain."""

    def __init__(self, overlay_id: str, changed_behavior: str, *overlays: Overlay):
        self.overlay_id = overlay_id
        self.changed_behavior = changed_behavior
        self.overlays = overlays
        self.parameters = {"composed_of": [o.overlay_id for o in overlays]}

    def adjust(self, strategy, side, quantity, current):
        quantity = _decimal(quantity)
        for overlay in self.overlays:
            side, quantity, veto = overlay.adjust(strategy, side, quantity, current)
            if veto:
                return side, quantity, veto
        return side, quantity, None

    def identity(self) -> dict:
        return {**super().identity(), "components": [o.identity() for o in self.overlays]}
