"""Chronological window definitions shared by forecast and fixed-strategy research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WFWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def generate_windows(
    start: str, end: str, train_days: int, test_days: int, step_days: int
) -> list[WFWindow]:
    """Produce rolling (sliding) train/test windows over [start, end]."""
    if min(train_days, test_days, step_days) <= 0:
        raise ValueError("window durations must be positive")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    windows: list[WFWindow] = []
    train_lo = start_ts
    while True:
        train_hi = train_lo + pd.Timedelta(days=train_days)
        test_lo = train_hi
        test_hi = test_lo + pd.Timedelta(days=test_days)
        if test_hi > end_ts:
            break
        windows.append(
            WFWindow(
                train_start=str(train_lo),
                train_end=str(train_hi),
                test_start=str(test_lo),
                test_end=str(test_hi),
            )
        )
        train_lo = train_lo + pd.Timedelta(days=step_days)
    return windows
