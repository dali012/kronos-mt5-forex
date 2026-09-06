"""Operational attribution: incidents, restarts and telemetry holes.

Equity movement around an incident is reported as ASSOCIATION only. The audit
has no mechanism to establish that an incident caused a loss, and says so.
"""

from __future__ import annotations

from datetime import timedelta

from kronos_mt5.performance_audit.loading import as_float, parse_ts

# Window either side of an incident used to associate an equity move with it.
ASSOCIATION_WINDOW = timedelta(hours=6)
ASSOCIATION_DISCLAIMER = (
    "Equity moves shown next to incidents are temporal ASSOCIATION only. This "
    "audit cannot and does not establish causation."
)


def _equity_at(sorted_points: list[tuple], moment) -> float | None:
    """Last equity observation at or before `moment`."""
    chosen = None
    for ts, value in sorted_points:
        if ts <= moment:
            chosen = value
        else:
            break
    return chosen


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
        before = _equity_at(points, started - ASSOCIATION_WINDOW)
        after = _equity_at(points, (ended or started) + ASSOCIATION_WINDOW)
        detailed.append(
            {
                "kind": kind,
                "started_ts": started.isoformat(),
                "ended_ts": ended.isoformat() if ended else None,
                "duration_hours": duration_hours,
                "equity_before_window": before,
                "equity_after_window": after,
                "equity_change_in_window": (
                    after - before if (before is not None and after is not None) else None
                ),
                "association_only": True,
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

    worst_incidents = sorted(
        (d for d in detailed if d["equity_change_in_window"] is not None),
        key=lambda d: (d["equity_change_in_window"], d["started_ts"]),
    )[:10]

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
        "association_window_hours": ASSOCIATION_WINDOW.total_seconds() / 3600.0,
        "disclaimer": ASSOCIATION_DISCLAIMER,
    }
