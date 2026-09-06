"""Equity-curve analysis on a normalized daily series.

The companion writes an equity snapshot roughly every 60 seconds, so the raw
table contains ~115k highly autocorrelated observations. Computing Sharpe or
volatility from those would be meaningless: this module first collapses the
observations to one value per UTC day (the last valid observation of that day)
and derives every ratio from that daily series.

All functions are pure: they take rows and return dictionaries.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from itertools import pairwise

from kronos_mt5.performance_audit.loading import as_float, parse_ts

PERIODS_PER_YEAR = 365  # crypto trades every calendar day
MIN_DAYS_FOR_ANNUALIZATION = 30
CONFIDENT_HISTORY_DAYS = 365
# Below this many consecutive-day returns the volatility-based ratios are
# reported as unavailable rather than computed from an unusable sample.
MIN_RATIO_RETURNS = 2
# A snapshot cadence of ~60s means anything beyond this is a real telemetry hole.
GAP_ALERT_SECONDS = 3600.0


def normalize_daily(rows: list[dict]) -> dict:
    """Collapse raw equity observations into one row per UTC day.

    Returns the daily series plus the data-quality counters gathered on the way:
    invalid values, duplicate timestamps, unparsable timestamps and observation
    gaps. Nothing is interpolated — missing days are reported, never invented.
    """
    invalid_equity = 0
    unparsable_ts = 0
    duplicate_ts = 0
    seen_ts: set[str] = set()
    observations: list[tuple] = []

    for row in rows:
        moment = parse_ts(row.get("ts"))
        if moment is None:
            unparsable_ts += 1
            continue
        key = moment.isoformat()
        if key in seen_ts:
            duplicate_ts += 1
        seen_ts.add(key)
        value = as_float(row.get("equity"))
        if value is None or value <= 0:
            invalid_equity += 1
            continue
        observations.append(
            (
                moment,
                value,
                as_float(row.get("cash")),
                as_float(row.get("unrealized")),
                as_float(row.get("n_open")),
            )
        )

    observations.sort(key=lambda item: item[0])
    raw = _raw_extremes(observations)

    by_day: dict[date, dict] = {}
    day_open_counts: dict[date, list[float]] = {}
    for moment, value, cash, unrealized, n_open in observations:
        day = moment.date()
        # Last valid observation of the day wins.
        by_day[day] = {
            "date": day.isoformat(),
            "ts": moment.isoformat(),
            "equity": value,
            "cash": cash,
            "unrealized": unrealized,
            "n_open": n_open,
        }
        if n_open is not None:
            day_open_counts.setdefault(day, []).append(n_open)

    daily = []
    for day in sorted(by_day):
        entry = dict(by_day[day])
        opens = day_open_counts.get(day, [])
        entry["observations"] = sum(1 for o in observations if o[0].date() == day)
        entry["mean_n_open"] = (sum(opens) / len(opens)) if opens else None
        entry["max_n_open"] = max(opens) if opens else None
        daily.append(entry)

    # Attach the day-over-day return once the series is ordered. `gap_days`
    # records how many calendar days the return actually spans: a return that
    # bridges missing days is NOT a daily return and is excluded from the
    # volatility-based ratios. Nothing is ever interpolated.
    previous_value = None
    previous_date = None
    for entry in daily:
        current_date = date.fromisoformat(entry["date"])
        if previous_value is None or previous_value <= 0:
            entry["return"] = None
            entry["gap_days"] = None
            entry["consecutive"] = False
        else:
            entry["return"] = entry["equity"] / previous_value - 1.0
            entry["gap_days"] = (current_date - previous_date).days
            entry["consecutive"] = entry["gap_days"] == 1
        previous_value = entry["equity"]
        previous_date = current_date

    gaps = _observation_gaps(observations)
    missing = _missing_days(daily)
    return {
        "daily": daily,
        "raw": raw,
        "raw_observations": len(observations),
        "invalid_equity_rows": invalid_equity,
        "unparsable_timestamps": unparsable_ts,
        "duplicate_timestamps": duplicate_ts,
        "observation_gaps": gaps,
        "missing_days": missing,
    }


def _raw_extremes(observations: list[tuple]) -> dict:
    """Endpoints and extremes of the RAW observation series.

    The daily series intentionally keeps only the last observation of each day, so
    its minimum and maximum are not the intraday extremes. Both are reported: a
    dashboard or a hand-written SELECT will show the raw numbers, and a reader must
    be able to reconcile the two without thinking the audit is wrong.
    """
    if not observations:
        return {}
    lowest = min(observations, key=lambda o: o[1])
    highest = max(observations, key=lambda o: o[1])
    first, last = observations[0], observations[-1]
    peak = observations[0][1]
    worst = 0.0
    for _, value, *_ in observations:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    period_return = (last[1] / first[1] - 1.0) if first[1] > 0 else None
    return {
        "first_ts": first[0].isoformat(),
        "first_equity": first[1],
        "last_ts": last[0].isoformat(),
        "last_equity": last[1],
        "min_equity": lowest[1],
        "min_ts": lowest[0].isoformat(),
        "max_equity": highest[1],
        "max_ts": highest[0].isoformat(),
        "period_return_pct": period_return * 100.0 if period_return is not None else None,
        "max_drawdown_pct": worst * 100.0,
    }


def _observation_gaps(observations: list[tuple]) -> list[dict]:
    gaps = []
    for earlier, later in pairwise(observations):
        delta = (later[0] - earlier[0]).total_seconds()
        if delta > GAP_ALERT_SECONDS:
            gaps.append(
                {
                    "from": earlier[0].isoformat(),
                    "to": later[0].isoformat(),
                    "hours": round(delta / 3600.0, 3),
                    "equity_before": earlier[1],
                    "equity_after": later[1],
                    "equity_change": later[1] - earlier[1],
                }
            )
    gaps.sort(key=lambda g: (-g["hours"], g["from"]))
    return gaps


def _missing_days(daily: list[dict]) -> list[str]:
    if len(daily) < 2:
        return []
    first = date.fromisoformat(daily[0]["date"])
    last = date.fromisoformat(daily[-1]["date"])
    present = {entry["date"] for entry in daily}
    missing = []
    cursor = first
    while cursor <= last:
        if cursor.isoformat() not in present:
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return missing


def max_drawdown(daily: list[dict]) -> dict:
    """Peak-to-trough drawdown on the daily series, with recovery."""
    if not daily:
        return {
            "max_drawdown_pct": None,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "recovered": None,
            "drawdown_days": None,
        }
    peak = daily[0]["equity"]
    peak_date = daily[0]["date"]
    worst = 0.0
    worst_peak_date = peak_date
    worst_trough_date = daily[0]["date"]
    worst_peak_value = peak
    worst_trough_value = peak
    for entry in daily:
        value = entry["equity"]
        if value > peak:
            peak = value
            peak_date = entry["date"]
        if peak > 0:
            drop = value / peak - 1.0
            if drop < worst:
                worst = drop
                worst_peak_date = peak_date
                worst_trough_date = entry["date"]
                worst_peak_value = peak
                worst_trough_value = value

    recovery_date = None
    seen_trough = False
    for entry in daily:
        if entry["date"] == worst_trough_date:
            seen_trough = True
            continue
        if seen_trough and entry["equity"] >= worst_peak_value:
            recovery_date = entry["date"]
            break

    start = date.fromisoformat(worst_peak_date)
    end = date.fromisoformat(worst_trough_date)
    return {
        "max_drawdown_pct": worst * 100.0,
        "peak_date": worst_peak_date,
        "peak_equity": worst_peak_value,
        "trough_date": worst_trough_date,
        "trough_equity": worst_trough_value,
        "recovery_date": recovery_date,
        "recovered": recovery_date is not None,
        "drawdown_days": (end - start).days,
    }


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def analyze_equity(rows: list[dict]) -> dict:
    """Full equity analysis. Never raises on degenerate input."""
    normalized = normalize_daily(rows)
    daily = normalized["daily"]
    out: dict = {
        "available": bool(daily),
        "raw_observations": normalized["raw_observations"],
        "invalid_equity_rows": normalized["invalid_equity_rows"],
        "unparsable_timestamps": normalized["unparsable_timestamps"],
        "duplicate_timestamps": normalized["duplicate_timestamps"],
        "observation_gaps": normalized["observation_gaps"][:20],
        "observation_gap_count": len(normalized["observation_gaps"]),
        "missing_days": normalized["missing_days"],
        "missing_day_count": len(normalized["missing_days"]),
        "observation_series": normalized.get("raw", {}),
        "daily_series_convention": (
            "one point per UTC day, taken as the LAST valid observation of that day; "
            "no interpolation. Ratio statistics use this series. `observation_series` "
            "reports the raw endpoints and intraday extremes, which will differ."
        ),
        "daily": daily,
    }
    if not daily:
        out["note"] = "no usable equity observations"
        return out

    first, last = daily[0], daily[-1]
    first_date = date.fromisoformat(first["date"])
    last_date = date.fromisoformat(last["date"])
    # Three counts, deliberately not interchangeable:
    #   calendar_days_covered - inclusive span of dates (16 Jun..05 Sep == 82)
    #   elapsed_days          - time actually elapsed   (81) -> the CAGR exponent
    #   days_observed         - dates that actually carry an observation
    calendar_days_covered = (last_date - first_date).days + 1
    elapsed_days = (last_date - first_date).days
    start_equity, end_equity = first["equity"], last["equity"]
    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else None

    all_returns = [e["return"] for e in daily if e["return"] is not None]
    # Only a return spanning exactly one calendar day is a daily return. A
    # multi-day return carries multi-day variance and would deflate volatility
    # and inflate Sharpe if pooled with the rest.
    ratio_returns = [e["return"] for e in daily if e.get("consecutive") and e["return"] is not None]
    excluded_gap_returns = len(all_returns) - len(ratio_returns)

    positive = sum(1 for r in all_returns if r > 0)
    negative = sum(1 for r in all_returns if r < 0)
    flat = sum(1 for r in all_returns if r == 0)

    ratios_available = len(ratio_returns) >= MIN_RATIO_RETURNS
    returns = ratio_returns if ratios_available else []
    stdev = _stdev(returns)
    mean = (sum(returns) / len(returns)) if returns else None
    downside = [min(r, 0.0) for r in returns]
    downside_dev = math.sqrt(sum(d * d for d in downside) / len(downside)) if downside else None

    annualized_vol = stdev * math.sqrt(PERIODS_PER_YEAR) if stdev is not None else None
    sharpe = None
    if stdev and stdev > 0 and mean is not None:
        sharpe = mean / stdev * math.sqrt(PERIODS_PER_YEAR)
    sortino = None
    if downside_dev and downside_dev > 0 and mean is not None:
        sortino = mean / downside_dev * math.sqrt(PERIODS_PER_YEAR)

    # CAGR compounds over ELAPSED time, not over the inclusive date count. With a
    # single observation day elapsed_days is 0 and the figure is undefined.
    annualized_return = None
    if total_return is not None and elapsed_days > 0 and (1.0 + total_return) > 0:
        annualized_return = (1.0 + total_return) ** (PERIODS_PER_YEAR / elapsed_days) - 1.0

    drawdown = max_drawdown(daily)
    calmar = None
    max_dd = drawdown["max_drawdown_pct"]
    if annualized_return is not None and max_dd is not None and max_dd < 0:
        calmar = annualized_return / abs(max_dd / 100.0)

    best = max(daily, key=lambda e: (e["return"] is not None, e["return"] or 0.0))
    worst = min(daily, key=lambda e: (e["return"] is None, e["return"] or 0.0))
    opens = [e["mean_n_open"] for e in daily if e["mean_n_open"] is not None]
    days_flat_book = sum(1 for e in daily if (e["max_n_open"] or 0) == 0)

    ratios_note = None
    if not ratios_available:
        ratios_note = (
            f"only {len(ratio_returns)} consecutive-day returns are available "
            f"(minimum {MIN_RATIO_RETURNS}); volatility, Sharpe and Sortino are "
            f"reported as unavailable rather than computed from an unusable sample"
        )
    elif excluded_gap_returns:
        ratios_note = (
            f"{excluded_gap_returns} return(s) spanned missing calendar days and "
            f"were excluded from the volatility-based ratios; they remain in the "
            f"return series and in the positive/negative day counts"
        )

    low_confidence = elapsed_days < CONFIDENT_HISTORY_DAYS
    out.update(
        {
            "first_observation": first["ts"],
            "last_observation": last["ts"],
            "first_day": first["date"],
            "last_day": last["date"],
            "calendar_days_covered": calendar_days_covered,
            "elapsed_days": elapsed_days,
            "days_observed": len(daily),
            "return_periods": len(all_returns),
            "ratio_return_periods": len(ratio_returns),
            "excluded_gap_returns": excluded_gap_returns,
            "ratios_available": ratios_available,
            "ratios_note": ratios_note,
            # kept so a consumer written against schema 1.0.0 keeps working
            "days_covered": calendar_days_covered,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "min_equity": min(e["equity"] for e in daily),
            "max_equity": max(e["equity"] for e in daily),
            "absolute_return": end_equity - start_equity,
            "total_return_pct": total_return * 100.0 if total_return is not None else None,
            "daily_return_count": len(all_returns),
            "mean_daily_return_pct": mean * 100.0 if mean is not None else None,
            "median_daily_return_pct": (_median(all_returns) * 100.0 if all_returns else None),
            "mean_daily_return_basis": "consecutive-day returns only",
            "positive_days": positive,
            "negative_days": negative,
            "flat_days": flat,
            "hit_rate_pct": (positive / len(returns) * 100.0) if returns else None,
            "best_day": {"date": best["date"], "return_pct": (best["return"] or 0.0) * 100.0},
            "worst_day": {"date": worst["date"], "return_pct": (worst["return"] or 0.0) * 100.0},
            "annualized_return_pct": (
                annualized_return * 100.0 if annualized_return is not None else None
            ),
            "annualized_volatility_pct": (
                annualized_vol * 100.0 if annualized_vol is not None else None
            ),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "risk_free_rate": 0.0,
            "periods_per_year": PERIODS_PER_YEAR,
            "annualized_low_confidence": low_confidence,
            "annualized_confidence_note": (
                f"history spans {elapsed_days} elapsed days; annualized figures extrapolate "
                f"from less than {CONFIDENT_HISTORY_DAYS} days and are indicative only"
                if low_confidence
                else None
            ),
            "annualization_suppressed": elapsed_days < MIN_DAYS_FOR_ANNUALIZATION,
            "annualization_basis": "elapsed_days",
            "zero_variance": bool(stdev is not None and stdev == 0),
            "exposure": {
                "mean_positions": (sum(opens) / len(opens)) if opens else None,
                "max_positions": max((e["max_n_open"] or 0) for e in daily) if daily else None,
                "days_with_flat_book": days_flat_book,
                "days_with_flat_book_pct": (days_flat_book / len(daily) * 100.0 if daily else None),
            },
            **{f"drawdown_{k}": v for k, v in drawdown.items()},
        }
    )
    return out
