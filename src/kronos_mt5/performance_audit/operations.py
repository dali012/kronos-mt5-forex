"""Operational attribution: incidents, restarts and telemetry holes.

Equity movement around an incident is reported as ASSOCIATION only. The audit
has no mechanism to establish that an incident caused a loss, and says so.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import timedelta

from kronos_mt5.performance_audit.loading import as_float, parse_ts

# Window either side of an incident used to associate an equity move with it.
ASSOCIATION_WINDOW = timedelta(hours=6)
# The companion writes equity roughly every 60 seconds, so an observation more
# than this far from a requested boundary is not describing that boundary. Ten
# minutes allows for ~10 missed snapshots before the association is refused.
FRESHNESS_TOLERANCE = timedelta(minutes=10)
ASSOCIATION_DISCLAIMER = (
    "Equity moves shown next to incidents are temporal ASSOCIATION only. This "
    "audit cannot and does not establish causation."
)
EVALUABLE = "EVALUABLE"
NOT_EVALUABLE = "NOT_EVALUABLE"


def nearest_observation(sorted_points: list[tuple], target, tolerance=FRESHNESS_TOLERANCE):
    """The observation closest to `target`, or None if none is within tolerance.

    Deterministic: `bisect` locates the insertion point and ties (equidistant
    neighbours) always resolve to the EARLIER observation, so the result never
    depends on row order. Returning None is the point — the previous
    implementation happily reached hours or days away and reported the result as
    if it described the requested boundary.
    """
    if not sorted_points:
        return None
    times = [point[0] for point in sorted_points]
    index = bisect_left(times, target)
    candidates = []
    if index > 0:
        candidates.append(sorted_points[index - 1])
    if index < len(sorted_points):
        candidates.append(sorted_points[index])
    best = None
    best_distance = None
    for point in candidates:
        distance = abs(point[0] - target)
        if best_distance is None or distance < best_distance:
            best, best_distance = point, distance
    if best is None or best_distance > tolerance:
        return None
    return best


def evaluate_incident_window(
    sorted_points: list[tuple],
    started,
    ended,
    *,
    window=ASSOCIATION_WINDOW,
    tolerance=FRESHNESS_TOLERANCE,
) -> dict:
    """Equity across an incident, or an explicit reason why it cannot be measured.

    Refuses to produce a number when either endpoint is missing or stale, when
    both endpoints resolve to the same observation, or when the selected
    observations do not actually straddle the incident.
    """
    open_incident = ended is None
    end_anchor = ended if ended is not None else started
    before_target = started - window
    after_target = end_anchor + window
    result: dict = {
        "status": NOT_EVALUABLE,
        "open_incident": open_incident,
        "end_anchor": end_anchor.isoformat(),
        "end_anchor_source": "incident_start" if open_incident else "incident_end",
        "target_before_ts": before_target.isoformat(),
        "target_after_ts": after_target.isoformat(),
        "freshness_tolerance_seconds": tolerance.total_seconds(),
        "actual_before_ts": None,
        "actual_after_ts": None,
        "before_distance_seconds": None,
        "after_distance_seconds": None,
        "equity_before": None,
        "equity_after": None,
        "equity_change": None,
        "reason": None,
        "association_only": True,
    }

    before = nearest_observation(sorted_points, before_target, tolerance)
    after = nearest_observation(sorted_points, after_target, tolerance)
    if before is not None:
        result["actual_before_ts"] = before[0].isoformat()
        result["before_distance_seconds"] = abs((before[0] - before_target).total_seconds())
        result["equity_before"] = before[1]
    if after is not None:
        result["actual_after_ts"] = after[0].isoformat()
        result["after_distance_seconds"] = abs((after[0] - after_target).total_seconds())
        result["equity_after"] = after[1]

    if before is None and after is None:
        result["reason"] = "no equity observation within tolerance of either boundary"
        return result
    if before is None:
        result["reason"] = (
            "no equity observation within tolerance of the pre-incident boundary "
            "(telemetry gap, or the incident precedes the recorded history)"
        )
        return result
    if after is None:
        result["reason"] = (
            "no equity observation within tolerance of the post-incident boundary "
            "(telemetry gap, or the incident is inside the recorded tail)"
        )
        return result
    if before[0] == after[0]:
        result["reason"] = "both boundaries resolved to the same observation"
        return result
    if not (before[0] <= started and after[0] >= end_anchor):
        result["reason"] = "selected observations do not span the incident window"
        return result

    result["status"] = EVALUABLE
    result["equity_change"] = after[1] - before[1]
    return result


def analyze_operations(
    incidents: list[dict],
    ops_events: list[dict],
    equity_rows: list[dict],
    observation_gaps: list[dict],
) -> dict:
    points: list[tuple] = []
    for row in equity_rows:
        moment = parse_ts(row.get("ts"))
        value = as_float(row.get("equity"))
        if moment is not None and value is not None and value > 0:
            points.append((moment, value))
    points.sort(key=lambda p: p[0])

    by_kind: dict[str, int] = {}
    detailed: list[dict] = []
    open_incidents = 0
    total_downtime_hours = 0.0
    for row in incidents:
        kind = str(row.get("kind") or "UNKNOWN")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        started = parse_ts(row.get("started_ts"))
        ended = parse_ts(row.get("ended_ts"))
        if started is None:
            continue
        if ended is None:
            open_incidents += 1
        duration_hours = ((ended - started).total_seconds() / 3600.0) if ended else None
        if duration_hours:
            total_downtime_hours += duration_hours
        evaluation = evaluate_incident_window(points, started, ended)
        detailed.append(
            {
                "kind": kind,
                "started_ts": started.isoformat(),
                "ended_ts": ended.isoformat() if ended else None,
                "duration_hours": duration_hours,
                **evaluation,
            }
        )
    detailed.sort(key=lambda d: (d["started_ts"], d["kind"]))

    ops_by_kind: dict[str, int] = {}
    ops_timeline: list[dict] = []
    for row in ops_events:
        kind = str(row.get("kind") or "UNKNOWN")
        ops_by_kind[kind] = ops_by_kind.get(kind, 0) + 1
        moment = parse_ts(row.get("ts"))
        if moment is not None:
            ops_timeline.append({"ts": moment.isoformat(), "kind": kind})
    ops_timeline.sort(key=lambda e: (e["ts"], e["kind"]))

    evaluable = [d for d in detailed if d["status"] == EVALUABLE]
    worst_incidents = sorted(evaluable, key=lambda d: (d["equity_change"], d["started_ts"]))[:10]
    not_evaluable_reasons: dict[str, int] = {}
    for item in detailed:
        if item["status"] != EVALUABLE:
            reason = str(item.get("reason") or "unknown")
            not_evaluable_reasons[reason] = not_evaluable_reasons.get(reason, 0) + 1

    return {
        "incidents_total": len(incidents),
        "incidents_by_kind": dict(sorted(by_kind.items())),
        "open_incidents": open_incidents,
        "total_incident_hours": total_downtime_hours,
        "incidents": detailed,
        "ops_events_by_kind": dict(sorted(ops_by_kind.items())),
        "ops_timeline": ops_timeline,
        "service_starts": ops_by_kind.get("BOT_START", 0),
        "service_stops": ops_by_kind.get("BOT_STOP", 0),
        "telemetry_gaps": observation_gaps,
        "telemetry_gap_count": len(observation_gaps),
        "largest_telemetry_gap_hours": (max((g["hours"] for g in observation_gaps), default=None)),
        "most_negative_incident_windows": worst_incidents,
        "incidents_evaluable": len(evaluable),
        "incidents_not_evaluable": len(detailed) - len(evaluable),
        "not_evaluable_reasons": dict(sorted(not_evaluable_reasons.items())),
        "association_window_hours": ASSOCIATION_WINDOW.total_seconds() / 3600.0,
        "freshness_tolerance_seconds": FRESHNESS_TOLERANCE.total_seconds(),
        "disclaimer": ASSOCIATION_DISCLAIMER,
    }
