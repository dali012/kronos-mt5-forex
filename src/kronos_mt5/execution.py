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
#
#   WORKING ──timeout/supersede/control──▶ CANCEL_PENDING ──venue terminal──▶ SETTLING
#      │                                        │                                │
#      └── fully filled ──▶ SETTLING            └── cancel raised/rejected ──▶ CANCEL_RETRY
#                                                        (bounded backoff, always timed)
#   SETTLING ──settle delay elapsed──▶ TERMINAL ──▶ tombstone (bounded, absorbs late fills)
#
# Nothing is submitted while any owned order sits in a non-TERMINAL state. SETTLING
# exists so a late fill or reconciliation that lands *after* the cancel notification
# is folded into the position before any remaining delta is recomputed — callback
# ordering is never assumed.
STATE_WORKING = "WORKING"  # limit resting, timeout not reached
STATE_CANCEL_PENDING = "CANCEL_PENDING"  # cancel requested, awaiting a terminal event
STATE_CANCEL_RETRY = "CANCEL_RETRY"  # cancel failed/was rejected; retrying on a timer
STATE_SETTLING = "SETTLING"  # venue says closed; waiting for late fills to land
STATE_TERMINAL = "TERMINAL"  # settled; safe to size a replacement against the position

# --- fallback decisions ------------------------------------------------------
ACTION_SUBMIT = "SUBMIT"
ACTION_SKIP = "SKIP"

REASON_TIMEOUT = "timeout"
REASON_SUPERSEDED = "superseded"
SKIP_ALREADY_RESOLVED = "already_resolved"
SKIP_NOT_SETTLED = "not_settled"
SKIP_BELOW_THRESHOLD = "below_rebalance_threshold"
SKIP_UNOWNED_RESTING_ORDER = "unowned_resting_entry_order"
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
    retire_reason: str | None = None  # why it stopped being the active order
    retired_ts_ns: int = 0
    settled_ts_ns: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state == STATE_TERMINAL

    @property
    def is_settled(self) -> bool:
        """Terminal AND past the settling window — safe to size against."""
        return self.state == STATE_TERMINAL

    @property
    def is_retiring(self) -> bool:
        """Tracked but not yet safe to replace: cancel pending, retrying or settling."""
        return self.state in (STATE_CANCEL_PENDING, STATE_CANCEL_RETRY, STATE_SETTLING)

    def note(self, event: str) -> None:
        """Small bounded audit trail, useful in logs and tests."""
        self.events.append(event)
        del self.events[:-16]

    def register_fill(self, units: float) -> None:
        self.filled_units += abs(float(units))

    def begin_cancel(self, *, fallback_enabled: bool, reason: str, now_ns: int = 0) -> bool:
        """Move WORKING/CANCEL_RETRY -> CANCEL_PENDING. False if already past that."""
        if self.state not in (STATE_WORKING, STATE_CANCEL_RETRY):
            return False
        self.state = STATE_CANCEL_PENDING
        self.fallback_pending = bool(fallback_enabled) and not self.fallback_forbidden
        self.cancel_attempts += 1
        if not self.retire_reason:
            self.retire_reason = reason
            self.retired_ts_ns = now_ns
        self.note(f"cancel_requested:{reason}#{self.cancel_attempts}")
        return True

    def mark_cancel_failed(self, detail: str) -> None:
        """A cancel raised or was rejected: retry on a timer, never fall back."""
        self.state = STATE_CANCEL_RETRY
        self.fallback_pending = False
        self.fallback_forbidden = True
        self.note(f"cancel_failed:{detail}")

    def begin_settling(self, reason: str) -> bool:
        """Venue reported the order closed. Returns False for duplicate callbacks.

        Deliberately NOT terminal yet: a fill or reconciliation event can still
        arrive after the cancel notification, and it must be folded into the
        position before any remaining delta is computed.
        """
        if self.state in (STATE_SETTLING, STATE_TERMINAL):
            return False
        self.state = STATE_SETTLING
        self.note(f"settling:{reason}")
        return True

    def mark_terminal(self, reason: str, *, now_ns: int = 0) -> bool:
        """Settling window elapsed. Returns False for duplicate callbacks."""
        if self.state == STATE_TERMINAL:
            return False
        self.state = STATE_TERMINAL
        self.settled_ts_ns = now_ns
        self.note(f"terminal:{reason}")
        return True


@dataclass
class DeferredEntry:
    """The newest desired entry, parked until every owned predecessor is settled.

    Only the newest target is kept: repeated supersedes coalesce onto this one
    record, so a burst of rebalances can never queue a burst of orders.
    """

    target_units: float
    decision_price: float
    decision_ts_ns: int
    threshold_notional: float = 0.0
    requests: int = 1
    reason: str = REASON_SUPERSEDED

    def coalesce(self, other: DeferredEntry) -> None:
        """Replace the parked target with a newer one, keeping the request count."""
        self.target_units = other.target_units
        self.decision_price = other.decision_price
        self.decision_ts_ns = other.decision_ts_ns
        self.threshold_notional = other.threshold_notional
        self.reason = other.reason
        self.requests += 1


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
    threshold_notional: float = 0.0,
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

    if threshold_notional > 0 and abs(remaining) * fresh_price < threshold_notional:
        # The delta no longer clears the rebalance/minimum-notional bar: trading it
        # would cost more than the drift it corrects.
        return FallbackDecision(
            ACTION_SKIP,
            SKIP_BELOW_THRESHOLD,
            fresh_price=fresh_price,
            remaining_units=remaining,
        )

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
