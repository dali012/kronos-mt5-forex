"""Unit tests for the pure walk-forward helpers (no model / no engine run)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.walk_forward import generate_windows, monte_carlo


def test_generate_windows_are_sequential_and_oos():
    windows = generate_windows(
        "2024-01-01", "2024-02-01", train_days=10, test_days=5, step_days=5
    )
    assert len(windows) >= 2
    for w in windows:
        # test window starts exactly where training ends (out-of-sample, no gap)
        assert w.test_start == w.train_end
        assert pd.Timestamp(w.train_start) < pd.Timestamp(w.train_end)
        assert pd.Timestamp(w.test_start) < pd.Timestamp(w.test_end)
    # sliding by step_days
    assert pd.Timestamp(windows[1].train_start) > pd.Timestamp(windows[0].train_start)


def test_generate_windows_empty_when_range_too_small():
    assert generate_windows("2024-01-01", "2024-01-05", 10, 5, 5) == []


def test_monte_carlo_keys_and_bounds():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, size=60))
    out = monte_carlo(rets, n_sims=500)
    for key in ("median_total_return", "prob_loss", "median_max_drawdown"):
        assert key in out
    assert 0.0 <= out["prob_loss"] <= 1.0
    assert out["median_max_drawdown"] <= 0.0


def test_monte_carlo_too_few_points():
    out = monte_carlo(pd.Series([0.01, -0.01]), n_sims=100)
    assert "note" in out
