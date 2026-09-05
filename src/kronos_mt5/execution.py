"""Execution primitives shared by the strategy (writer) and the companion (recorder).

Two concerns live here, both deliberately engine-free so they can be unit-tested
without NautilusTrader, a broker or a GPU:

1. The **patient-limit lifecycle** (`PatientOrder` + `evaluate_fallback`). The
   strategy owns the engine calls; every *decision* is made by the pure functions
   below so the awkward orderings (late fill, cancel rejection, duplicate
   callbacks) are testable.

2. **Execution telemetry** (`implementation_shortfall_*`, `classify_exec_role`,
   `liquidity_label`) plus the order-tag vocabulary that carries decision
   metadata from order creation through to the companion recorder.

Sign convention — used identically for implementation shortfall and for the
pre-fallback adverse-drift check:

    POSITIVE  => execution was WORSE than the decision price (a cost)
    NEGATIVE  => price improvement

    BUY :  (execution / decision - 1) * 10_000
    SELL:  (decision / execution - 1) * 10_000

Shortfall is a *measurement* of the price actually paid. It is already embedded
in the execution price and must never be booked as a separate cash expense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- order-tag vocabulary ----------------------------------------------------
# Tags are bounded, non-secret key=value strings carried by the order through the
# engine and read back by the companion recorder. Never put credentials or
# unbounded payloads here.
TAG_REF_PX = "REF_PX="  # decision / reference price (pre-existing)
TAG_RISK_PCT = "RISK_PCT="  # stop distance at decision time (pre-existing)
TAG_EXEC_ROLE = "EXEC_ROLE="  # why this order exists (see ROLE_* below)
TAG_DECISION_TS = "DEC_TS="  # decision timestamp, UNIX epoch nanoseconds
TAG_LIMIT_PX = "LIMIT_PX="  # submitted patient-limit price
TAG_FALLBACK_REASON = "FB_REASON="  # why a fallback was submitted
TAG_FALLBACK_DRIFT = "FB_DRIFT_BPS="  # adverse drift measured at fallback time

# --- execution roles ---------------------------------------------------------
ROLE_PATIENT_LIMIT = "PATIENT_LIMIT"
ROLE_PATIENT_FALLBACK = "PATIENT_FALLBACK"
ROLE_TREND_MARKET = "TREND_MARKET"
ROLE_HARD_STOP = "HARD_STOP"
ROLE_TAKE_PROFIT = "TAKE_PROFIT"
ROLE_TRAILING_STOP = "TRAILING_STOP"
ROLE_OTHER = "OTHER"

# --- patient-order lifecycle states -----------------------------------------
STATE_WORKING = "WORKING"  # limit resting, timeout not reached
STATE_CANCEL_PENDING = "CANCEL_PENDING"  # cancel requested, awaiting a terminal event
STATE_TERMINAL = "TERMINAL"  # order closed; state is about to be cleared

# --- fallback decisions ------------------------------------------------------
ACTION_SUBMIT = "SUBMIT"
ACTION_SKIP = "SKIP"

REASON_TIMEOUT = "timeout"
SKIP_ALREADY_RESOLVED = "already_resolved"
SKIP_FALLBACK_DISABLED = "fallback_disabled"
SKIP_CONTROL = "control_active"
SKIP_NO_FRESH_PRICE = "no_fresh_price"
SKIP_TARGET_REACHED = "target_reached"
SKIP_ADVERSE_DRIFT = "adverse_drift"

SIDE_BUY = "BUY"
SIDE_SELL = "SELL"


def implementation_shortfall_bps(side: str, decision_price: float, execution_price: float) -> float:
    """Signed execution cost in basis points. Positive = worse than the decision."""
    if not decision_price or not execution_price or decision_price <= 0 or execution_price <= 0:
        raise ValueError("decision_price and execution_price must be positive")
    if side == SIDE_BUY:
        return (execution_price / decision_price - 1.0) * 1e4
    return (decision_price / execution_price - 1.0) * 1e4


def implementation_shortfall_quote(
    side: str,
    decision_price: float,
    execution_price: float,
    qty: float,
) -> float:
    """Signed execution cost in quote currency. Positive = worse than the decision."""
    if side == SIDE_BUY:
        return (execution_price - decision_price) * qty
    return (decision_price - execution_price) * qty


def adverse_drift_bps(side: str, decision_price: float, fresh_price: float) -> float:
    """How far the market has moved AGAINST `side` since the decision price.

    Same sign convention as `implementation_shortfall_bps`: positive is adverse,
    negative is favourable. Favourable drift must never trip the adverse cap.
    """
    return implementation_shortfall_bps(side, decision_price, fresh_price)


def liquidity_label(liquidity_side_name: str | None) -> str:
    """Normalise a NautilusTrader LiquiditySide name to MAKER / TAKER / UNKNOWN."""
    if liquidity_side_name in ("MAKER", "TAKER"):
        return liquidity_side_name
    return "UNKNOWN"


def classify_exec_role(
    order_type_name: str | None,
    *,
    reduce_only: bool = False,
    tags: object = None,
) -> str:
    """Best-effort execution role for a fill.

    Prefers the explicit `EXEC_ROLE=` tag; falls back to the order shape so fills
    recorded before this release (and venue-side reconciled orders, which carry
    no tags) still classify sensibly.
    """
    tagged = tag_value(tags, TAG_EXEC_ROLE)
    if tagged:
        return tagged
    if order_type_name == "STOP_MARKET":
        return ROLE_HARD_STOP
    if order_type_name == "TRAILING_STOP_MARKET":
        return ROLE_TRAILING_STOP
    if order_type_name == "LIMIT" and reduce_only:
        return ROLE_TAKE_PROFIT
    if order_type_name == "LIMIT":
        return ROLE_PATIENT_LIMIT
    if order_type_name == "MARKET":
        return ROLE_TREND_MARKET
    return ROLE_OTHER


def tag_value(tags: object, prefix: str) -> str | None:
    """Read one `prefix<value>` order tag. Tolerates None/odd tag containers."""
    if not tags:
        return None
    if isinstance(tags, str):
        tags = [tags]
    try:
        iterator = iter(tags)
    except TypeError:
        return None
    for tag in iterator:
        text = str(tag)
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def tag_float(tags: object, prefix: str) -> float | None:
    raw = tag_value(tags, prefix)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def tag_int(tags: object, prefix: str) -> int | None:
    raw = tag_value(tags, prefix)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


@dataclass
class PatientOrder:
    """Explicit state for one working patient-limit entry.

    `target_units` is the *intended net position* the decision aimed at, not the
    order quantity. The fallback delta is always recomputed as
    ``target_units - current_units`` so a partial fill can never be re-sent.
    """

    client_order_id: str
    symbol: str
    side: str  # side of the ORIGINAL limit order
    target_units: float
    decision_price: float
    decision_ts_ns: int
    limit_price: float
    submitted_units: float
    post_only: bool = False
    state: str = STATE_WORKING
    fallback_pending: bool = False  # a fallback is waiting for cancel confirmation
    fallback_resolved: bool = False  # a fallback was already submitted or skipped
    fallback_forbidden: bool = False  # a control event / failed cancel banned the fallback
    filled_units: float = 0.0
    cancel_attempts: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state == STATE_TERMINAL

    def note(self, event: str) -> None:
        """Small bounded audit trail, useful in logs and tests."""
        self.events.append(event)
        del self.events[:-16]

    def register_fill(self, units: float) -> None:
        self.filled_units += abs(float(units))

    def begin_cancel(self, *, fallback_enabled: bool) -> bool:
        """Move WORKING -> CANCEL_PENDING. Returns False if already past that."""
        if self.state != STATE_WORKING:
            return False
        self.state = STATE_CANCEL_PENDING
        self.fallback_pending = bool(fallback_enabled) and not self.fallback_forbidden
        self.cancel_attempts += 1
        self.note("cancel_requested")
        return True

    def mark_terminal(self, reason: str) -> bool:
        """Move to TERMINAL exactly once. Returns False for duplicate callbacks."""
        if self.state == STATE_TERMINAL:
            return False
        self.state = STATE_TERMINAL
        self.note(f"terminal:{reason}")
        return True


@dataclass(frozen=True)
class FallbackDecision:
    action: str
    reason: str
    side: str | None = None
    units: float = 0.0
    fresh_price: float | None = None
    drift_bps: float | None = None
    remaining_units: float = 0.0

    @property
    def submit(self) -> bool:
        return self.action == ACTION_SUBMIT


def remaining_units(target_units: float, current_units: float) -> float:
    """Signed units still needed to reach the target position."""
    return float(target_units) - float(current_units)


def evaluate_fallback(
    patient: PatientOrder,
    *,
    current_units: float,
    fresh_price: float | None,
    fallback_enabled: bool,
    control_reason: str | None = None,
    max_adverse_bps: float = 0.0,
    min_units: float = 0.0,
) -> FallbackDecision:
    """Decide whether a market fallback may be submitted, and for how much.

    Called only *after* the patient limit has reached an authoritative terminal
    state. Every guard returns a structured skip reason so the caller can log it
    rather than silently doing nothing. `max_adverse_bps <= 0` disables the cap.
    """
    remaining = remaining_units(patient.target_units, current_units)
    if patient.fallback_resolved:
        return FallbackDecision(ACTION_SKIP, SKIP_ALREADY_RESOLVED, remaining_units=remaining)
    if not fallback_enabled or not patient.fallback_pending:
        return FallbackDecision(ACTION_SKIP, SKIP_FALLBACK_DISABLED, remaining_units=remaining)
    if control_reason:
        return FallbackDecision(ACTION_SKIP, control_reason, remaining_units=remaining)
    if abs(remaining) <= max(float(min_units), 0.0) or remaining == 0.0:
        # The limit filled (fully, or enough that the residual is untradeable).
        return FallbackDecision(ACTION_SKIP, SKIP_TARGET_REACHED, remaining_units=remaining)
    if fresh_price is None or fresh_price <= 0:
        # Never reuse the stale decision price as if it were executable.
        return FallbackDecision(ACTION_SKIP, SKIP_NO_FRESH_PRICE, remaining_units=remaining)

    side = SIDE_BUY if remaining > 0 else SIDE_SELL
    drift = adverse_drift_bps(side, patient.decision_price, fresh_price)
    if max_adverse_bps > 0 and drift > max_adverse_bps:
        return FallbackDecision(
            ACTION_SKIP,
            SKIP_ADVERSE_DRIFT,
            side=side,
            units=abs(remaining),
            fresh_price=fresh_price,
            drift_bps=drift,
            remaining_units=remaining,
        )
    return FallbackDecision(
        ACTION_SUBMIT,
        REASON_TIMEOUT,
        side=side,
        units=abs(remaining),
        fresh_price=fresh_price,
        drift_bps=drift,
        remaining_units=remaining,
    )
