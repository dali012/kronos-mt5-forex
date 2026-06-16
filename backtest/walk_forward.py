"""Walk-forward validation — the deliverable that decides if this is worth pursuing.

Rolls a train/calibrate window then tests on the NEXT unseen window, repeatedly,
and aggregates cost-adjusted out-of-sample metrics. A single backtest is almost
always overfit; surviving many OOS windows is the real bar.

Reports per window AND aggregated: expectancy, Sharpe, max drawdown, profit
factor, trade count. Then bootstraps the stitched OOS returns (Monte Carlo) to
estimate bust vs goal probability, and writes a QuantStats tearsheet on the
stitched out-of-sample equity curve.

Calibration: Kronos is used zero-shot (no per-window retraining), so "calibrate"
here means re-fitting only the cost-aware signal threshold on the train window
(cheap, honest first version). Pass --no-calibrate to use a fixed threshold.

Usage:
    python backtest/walk_forward.py [CSV] [--train-days N] [--test-days N] \
        [--step-days N] [--no-calibrate] [--sample-count N]
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backtest.common import (
    DATA_DIR,
    REPORTS_DIR,
    BacktestSetup,
    CostModel,
    build_engine,
    load_brain,
    load_ohlcv,
    returns_from_equity,
    summarize,
)
from backtest.common import equity_curve as _equity_curve


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


def _slice(ohlcv: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    return ohlcv[(ohlcv.index >= pd.Timestamp(lo)) & (ohlcv.index < pd.Timestamp(hi))]


def _run(
    ohlcv: pd.DataFrame, setup: BacktestSetup, costs: CostModel, brain
) -> tuple[dict, pd.Series]:
    engine, venue = build_engine(ohlcv, setup, costs, brain, log_level="ERROR")
    engine.run()
    metrics = summarize(engine, venue)
    rets = returns_from_equity(_equity_curve(engine, venue))
    engine.dispose()
    return metrics, rets


def calibrate_threshold(
    train_ohlcv: pd.DataFrame,
    setup: BacktestSetup,
    costs: CostModel,
    brain,
    grid: tuple[float, ...] = (4.0, 8.0, 12.0),
) -> float:
    """Pick the threshold (bps) maximizing train-window Sharpe (min 3 trades)."""
    best_bps, best_score = setup.signal_threshold_bps, -np.inf
    for bps in grid:
        metrics, _ = _run(train_ohlcv, replace(setup, signal_threshold_bps=bps), costs, brain)
        score = metrics["sharpe"] if metrics["n_trades"] >= 3 else -np.inf
        print(
            f"    calibrate thr={bps:>5.1f}bps -> sharpe={metrics['sharpe']:+.3f} "
            f"trades={metrics['n_trades']}"
        )
        if score > best_score:
            best_bps, best_score = bps, score
    return best_bps


def run_window(
    window: WFWindow,
    ohlcv: pd.DataFrame,
    setup: BacktestSetup,
    costs: CostModel,
    brain,
    calibrate: bool,
) -> tuple[dict, pd.Series]:
    """Calibrate on train, backtest on test, return OOS metrics + return series."""
    train = _slice(ohlcv, window.train_start, window.train_end)
    test = _slice(ohlcv, window.test_start, window.test_end)

    thr = setup.signal_threshold_bps
    if calibrate:
        thr = calibrate_threshold(train, setup, costs, brain)
    print(f"  -> threshold={thr:.1f}bps; OOS test on {len(test)} bars")

    # The test window needs `lookback` bars of history to forecast from its
    # first bar, so prepend the tail of the train window (no look-ahead: this is
    # past data relative to the test bars).
    warmup = ohlcv[ohlcv.index < pd.Timestamp(window.test_start)].tail(setup.lookback)
    test_with_warmup = pd.concat([warmup, test])
    test_with_warmup = test_with_warmup[~test_with_warmup.index.duplicated(keep="last")]

    metrics, rets = _run(test_with_warmup, replace(setup, signal_threshold_bps=thr), costs, brain)
    metrics["threshold_bps"] = thr
    return metrics, rets


def monte_carlo(stitched: pd.Series, n_sims: int = 2000, goal: float = 0.10) -> dict:
    """Bootstrap stitched OOS returns to estimate bust/goal probabilities."""
    if len(stitched) < 5:
        return {"note": "too few OOS returns for Monte Carlo"}
    rng = np.random.default_rng(42)
    r = stitched.to_numpy()
    n = len(r)
    terminals = np.empty(n_sims)
    max_dds = np.empty(n_sims)
    for i in range(n_sims):
        sample = rng.choice(r, size=n, replace=True)
        curve = np.cumprod(1.0 + sample)
        terminals[i] = curve[-1] - 1.0
        running_max = np.maximum.accumulate(curve)
        max_dds[i] = ((curve - running_max) / running_max).min()
    return {
        "median_total_return": float(np.median(terminals)),
        "p05_total_return": float(np.percentile(terminals, 5)),
        "p95_total_return": float(np.percentile(terminals, 95)),
        "prob_loss": float(np.mean(terminals < 0.0)),
        f"prob_reach_{goal:.0%}": float(np.mean(terminals >= goal)),
        "median_max_drawdown": float(np.median(max_dds)),
        "worst_max_drawdown": float(np.min(max_dds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS validation")
    parser.add_argument("csv", nargs="?", default=str(DATA_DIR / "EURUSD_15M.csv"))
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--step-days", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--forecast-every", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu | cuda:0")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--out-json", default=None, help="write metrics to this JSON path")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    ohlcv = load_ohlcv(args.csv)
    setup = BacktestSetup(
        lookback=args.lookback,
        forecast_every_n_bars=args.forecast_every,
        sample_count=args.sample_count,
        kronos_device=args.device,
        kronos_model=args.model,
        kronos_tokenizer=args.tokenizer,
    )
    costs = CostModel()

    windows = generate_windows(
        str(ohlcv.index[0]),
        str(ohlcv.index[-1]),
        args.train_days,
        args.test_days,
        args.step_days,
    )
    if not windows:
        raise SystemExit(
            "No walk-forward windows fit the data range. Reduce --train/--test/--step-days."
        )

    print(f"Loaded {len(ohlcv)} bars; {len(windows)} walk-forward window(s).")
    print(f"Loading Kronos ({setup.kronos_model})...")
    brain = load_brain(setup)

    rows = []
    oos_returns: list[pd.Series] = []
    for i, w in enumerate(windows, 1):
        print(
            f"\n[window {i}/{len(windows)}] train {w.train_start[:10]}→{w.train_end[:10]} "
            f"test {w.test_start[:10]}→{w.test_end[:10]}"
        )
        metrics, rets = run_window(w, ohlcv, setup, costs, brain, not args.no_calibrate)
        metrics["window"] = i
        rows.append(metrics)
        if not rets.empty:
            oos_returns.append(rets)

    table = pd.DataFrame(rows).set_index("window")
    cols = [
        "threshold_bps",
        "n_trades",
        "total_return",
        "win_rate",
        "profit_factor",
        "expectancy_usd",
        "sharpe",
        "max_drawdown",
    ]
    print("\n========== WALK-FORWARD OOS RESULTS ==========")
    print(
        table[[c for c in cols if c in table.columns]].to_string(float_format=lambda x: f"{x:.4f}")
    )

    print("\n--- Aggregated (across windows) ---")
    for k in ["total_return", "sharpe", "max_drawdown", "profit_factor", "expectancy_usd"]:
        if k in table.columns:
            print(f"  mean {k:16s}: {table[k].replace([np.inf, -np.inf], np.nan).mean():.4f}")
    print(f"  total OOS trades : {int(table['n_trades'].sum())}")

    mc: dict = {}
    if oos_returns:
        stitched = pd.concat(oos_returns).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        mc = monte_carlo(stitched)
        print("\n--- Monte Carlo on stitched OOS returns ---")
        for k, v in mc.items():
            print(f"  {k:22s}: {v}")

        try:
            import quantstats as qs

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out = REPORTS_DIR / "tearsheet_walkforward.html"
            qs.reports.html(stitched, output=str(out), title="Kronos FX — Walk-forward OOS")
            print(f"\n[tearsheet] wrote {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[tearsheet] QuantStats skipped ({exc!r}).")

    if args.out_json:
        import json

        agg = {
            k: float(table[k].replace([np.inf, -np.inf], np.nan).mean())
            for k in ["total_return", "sharpe", "max_drawdown", "profit_factor", "expectancy_usd"]
            if k in table.columns
        }
        agg["total_oos_trades"] = int(table["n_trades"].sum())
        payload = {
            "symbol_csv": str(args.csv).split("/")[-1],
            "config": {
                "train_days": args.train_days,
                "test_days": args.test_days,
                "step_days": args.step_days,
                "lookback": args.lookback,
                "forecast_every": args.forecast_every,
                "sample_count": args.sample_count,
                "calibrate": not args.no_calibrate,
                "device": args.device,
            },
            "windows": table.reset_index().to_dict(orient="records"),
            "aggregate": agg,
            "monte_carlo": mc,
        }
        out_path = pathlib.Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\n[results] wrote {out_path}")


if __name__ == "__main__":
    main()
