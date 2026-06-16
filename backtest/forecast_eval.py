"""Direct forecast-quality diagnostic — does Kronos have a tradable edge at all?

This bypasses the whole trading engine (no NautilusTrader, no sizing, no exits)
and asks the only question that matters first: on out-of-sample bars, does the
sign/magnitude of Kronos' predicted return predict the realized forward return,
after costs?

For a grid of OOS anchor bars it forecasts `pred_len` ahead, then compares the
predicted return to the realized return over the same horizon and reports:
  - directional accuracy (all, and on the subset that clears the cost filter)
  - information coefficient (Pearson + Spearman corr of predicted vs realized)
  - net per-trade edge after costs for the LONG and SHORT signal subsets

If the cost-cleared directional accuracy isn't comfortably above 50% and the net
edge isn't positive, the strategy cannot win and tuning exits/sizing is pointless.

Usage:
    python backtest/forecast_eval.py data/EURUSD_15M.csv --device cuda:0 \
        --stride 16 --sample-count 5 --out-json reports/forecast_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kronos_mt5.brain.kronos_predictor import KronosBrain  # noqa: E402


def evaluate(
    df: pd.DataFrame,
    brain: KronosBrain,
    *,
    lookback: int,
    pred_len: int,
    stride: int,
    start_frac: float,
    sample_count: int,
    temperature: float,
    top_p: float,
    use_volume: bool = False,
) -> pd.DataFrame:
    """Roll over OOS anchors, returning a frame of (predicted, realized, prob_up)."""
    ts = pd.to_datetime(df["timestamps"])
    close = df["close"].to_numpy(dtype=float)
    cols = ["open", "high", "low", "close"] + (["volume"] if use_volume else [])
    start = max(lookback, int(len(df) * start_frac))
    rows = []
    anchors = range(start, len(df) - pred_len, stride)
    for n, t in enumerate(anchors, 1):
        window = df.iloc[t - lookback : t]
        x_ts = ts.iloc[t - lookback : t]
        y_ts = ts.iloc[t : t + pred_len]
        fc = brain.forecast(
            df=window[cols],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            sample_count=sample_count,
            temperature=temperature,
            top_p=top_p,
            use_volume=use_volume,
        )
        last_close = close[t - 1]
        realized = (close[t + pred_len - 1] - last_close) / last_close
        rows.append(
            {
                "anchor": t,
                "ts": ts.iloc[t - 1],
                "last_close": last_close,
                "pred_return": fc.expected_return,
                "prob_up": fc.prob_up,
                "realized_return": realized,
            }
        )
        if n % 50 == 0:
            print(f"  {n} forecasts...", flush=True)
    return pd.DataFrame(rows)


def summarize(
    res: pd.DataFrame, half_spread: float, threshold_bps: float, cost_bps: float | None = None
) -> dict:
    er = res["pred_return"].to_numpy()
    rr = res["realized_return"].to_numpy()
    price = res["last_close"].to_numpy()

    # round-trip cost as a return, plus the extra edge threshold (mirrors make_signal).
    # cost_bps (relative) is preferred for cross-asset comparison; otherwise fall
    # back to an absolute half-spread in price units (FX-style).
    if cost_bps is not None:
        cost_return = np.full_like(price, cost_bps / 1e4)
    else:
        cost_return = (2.0 * half_spread) / price
    required = cost_return + threshold_bps / 1e4

    dir_acc_all = float(np.mean(np.sign(er) == np.sign(rr)))
    traded = np.abs(er) > required
    n_traded = int(traded.sum())

    out = {
        "n_forecasts": int(len(res)),
        "directional_accuracy_all": dir_acc_all,
        "ic_pearson": float(np.corrcoef(er, rr)[0, 1]) if len(res) > 2 else 0.0,
        "ic_spearman": float(pd.Series(er).corr(pd.Series(rr), method="spearman")),
        "mean_pred_return": float(np.mean(er)),
        "mean_abs_pred_return": float(np.mean(np.abs(er))),
        "mean_realized_return": float(np.mean(rr)),
        "n_signals_cleared_cost": n_traded,
        "frac_signals_cleared_cost": float(np.mean(traded)),
    }

    if n_traded:
        dir_acc_traded = float(np.mean(np.sign(er[traded]) == np.sign(rr[traded])))
        # net per-signal edge after paying the round-trip spread
        signed = np.sign(er[traded]) * rr[traded] - cost_return[traded]
        out.update(
            {
                "directional_accuracy_traded": dir_acc_traded,
                "net_edge_per_trade_after_cost": float(np.mean(signed)),
                "net_edge_bps_after_cost": float(np.mean(signed) * 1e4),
                "win_rate_after_cost": float(np.mean(signed > 0)),
            }
        )
        longs = traded & (er > 0)
        shorts = traded & (er < 0)
        out["n_long"] = int(longs.sum())
        out["n_short"] = int(shorts.sum())
        out["mean_fwd_return_when_long"] = float(np.mean(rr[longs])) if longs.any() else None
        out["mean_fwd_return_when_short"] = float(np.mean(rr[shorts])) if shorts.any() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Kronos forecast-quality diagnostic")
    ap.add_argument("csv")
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--pred-len", type=int, default=12)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--target-points", type=int, default=None,
                    help="auto-pick stride to yield ~this many OOS forecasts (overrides --stride)")
    ap.add_argument("--start-frac", type=float, default=0.3, help="skip the first frac as burn-in")
    ap.add_argument("--sample-count", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--half-spread", type=float, default=0.00005)
    ap.add_argument("--threshold-bps", type=float, default=8.0)
    ap.add_argument("--use-volume", action="store_true",
                    help="feed real volume to Kronos (crypto/equities; FX has none)")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="round-trip cost in bps (relative; preferred for crypto). "
                         "Overrides --half-spread.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resample", default=None,
                    help="pandas rule to resample bars before evaluating, e.g. 1h | 4h | 1D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="NeoQuasar/Kronos-small",
                    help="e.g. NeoQuasar/Kronos-small | NeoQuasar/Kronos-base | NeoQuasar/Kronos-mini")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base",
                    help="Kronos-mini needs NeoQuasar/Kronos-Tokenizer-2k")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.resample:
        s = df.copy()
        s["timestamps"] = pd.to_datetime(s["timestamps"])
        s = s.set_index("timestamps")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df = s.resample(args.resample).agg(agg).dropna().reset_index()
        print(f"Resampled to {args.resample}: {len(df)} bars")
    stride = args.stride
    if args.target_points:
        start = max(args.lookback, int(len(df) * args.start_frac))
        n_oos = max(1, len(df) - start - args.pred_len)
        stride = max(1, n_oos // args.target_points)
        print(f"auto-stride: {n_oos} OOS bars -> stride {stride} (~{n_oos // stride} forecasts)")
    args.stride = stride  # record the effective stride in the output config
    args.windows_overlap = stride < args.pred_len

    print(f"Loaded {len(df)} bars from {args.csv}; loading {args.model} on {args.device}...")
    brain = KronosBrain(
        args.tokenizer,
        args.model,
        device=args.device,
        seed=args.seed,
    )

    res = evaluate(
        df,
        brain,
        lookback=args.lookback,
        pred_len=args.pred_len,
        stride=stride,
        start_frac=args.start_frac,
        sample_count=args.sample_count,
        temperature=args.temperature,
        top_p=args.top_p,
        use_volume=args.use_volume,
    )
    stats = summarize(res, args.half_spread, args.threshold_bps, cost_bps=args.cost_bps)

    print("\n========== FORECAST QUALITY (out-of-sample) ==========")
    for k, v in stats.items():
        print(f"  {k:32s}: {v}")

    # plain-language read
    acc = stats.get("directional_accuracy_traded", stats["directional_accuracy_all"])
    edge = stats.get("net_edge_bps_after_cost")
    print("\n--- read ---")
    print(f"  cost-cleared directional accuracy: {acc:.3f} (need > ~0.50)")
    if edge is not None:
        verdict = "POSITIVE edge after costs" if edge > 0 else "NO edge after costs"
        print(f"  net edge after costs: {edge:.2f} bps/trade -> {verdict}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": vars(args), "stats": stats}, indent=2, default=float))
        print(f"\n[results] wrote {out}")


if __name__ == "__main__":
    main()
