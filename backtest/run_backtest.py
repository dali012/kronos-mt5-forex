"""Single-window backtest on CSV.

Builds a BacktestEngine, loads FX bars from data/*.csv, runs KronosStrategy with
MANDATORY realistic costs (spread + slippage + commission), prints the built-in
reports, and writes a QuantStats tearsheet.

Usage:
    python backtest/run_backtest.py [CSV] [--max-bars N] [--threshold-bps X]

Verified against NautilusTrader 1.228.0. See backtest/common.py for the harness.
"""

from __future__ import annotations

import argparse

from backtest.common import (
    DATA_DIR,
    REPORTS_DIR,
    BacktestSetup,
    CostModel,
    build_engine,
    load_brain,
    load_ohlcv,
    print_reports,
    quantstats_tearsheet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-window Kronos FX backtest")
    parser.add_argument("csv", nargs="?", default=str(DATA_DIR / "EURUSD_15M.csv"))
    parser.add_argument("--max-bars", type=int, default=None, help="use only the last N bars")
    parser.add_argument("--threshold-bps", type=float, default=8.0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--forecast-every", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--device", default="cpu", help="cpu | cuda:0")
    args = parser.parse_args()

    ohlcv = load_ohlcv(args.csv)
    if args.max_bars:
        ohlcv = ohlcv.tail(args.max_bars)

    setup = BacktestSetup(
        lookback=args.lookback,
        forecast_every_n_bars=args.forecast_every,
        sample_count=args.sample_count,
        signal_threshold_bps=args.threshold_bps,
        kronos_device=args.device,
    )
    costs = CostModel()

    print(
        f"Loaded {len(ohlcv)} bars from {args.csv}\n"
        f"Costs: half_spread={costs.half_spread} commission=${costs.commission_usd} "
        f"prob_slippage={costs.prob_slippage}\n"
        f"Loading Kronos ({setup.kronos_model})..."
    )
    brain = load_brain(setup)

    engine, venue = build_engine(ohlcv, setup, costs, brain)
    print("Running backtest...")
    engine.run()

    print_reports(engine, venue)
    quantstats_tearsheet(engine, venue, REPORTS_DIR / "tearsheet_single.html")

    engine.dispose()


if __name__ == "__main__":
    main()
