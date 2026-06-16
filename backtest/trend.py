"""Vectorized multi-asset trend-following backtest (vol-targeted, cost-aware).

This drops the Kronos direction thesis (no edge, per forecast_eval) in favour of
the best-evidenced systematic edge: time-series momentum / trend-following. It is
*not* a profit promise — the bar is "survives out-of-sample after realistic
costs", which is what this measures.

Design (no look-ahead, no parameter fitting -> whole sample is OOS-like):
  - signal: ensemble of sign(momentum) over several fixed lookbacks (1/3/6/12mo),
    averaged -> a [-1, 1] conviction. Standard, robust, not optimized to the data.
  - vol targeting: each instrument sized to a constant annual risk, so all
    contribute equal risk and the portfolio is diversified.
  - costs: turnover * round-trip cost (bps) charged on every weight change.
  - portfolio: equal-weight blend of the vol-targeted instrument returns.

Run:  python backtest/trend.py --symbols EURUSD GBPUSD USDJPY BTCUSD
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
REPORTS = REPO / "reports"
PPY = 252  # periods per year (business-day daily)

# realistic round-trip costs per unit of notional turnover, in bps
DEFAULT_COST_BPS = {"EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 2.0, "BTCUSD": 10.0}


def load_close(symbol: str, tf: str = "1D") -> pd.Series:
    df = pd.read_csv(DATA / f"{symbol}_{tf}.csv")
    s = pd.Series(
        df["close"].astype(float).values,
        index=pd.to_datetime(df["timestamps"]),
        name=symbol,
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def trend_returns(
    close: pd.Series,
    lookbacks: list[int],
    vol_window: int,
    target_vol: float,
    max_leverage: float,
    cost_bps: float,
    ppy: int = PPY,
) -> pd.DataFrame:
    """Per-instrument vol-targeted trend-following net returns (no look-ahead).

    Returns are NaN until the instrument has enough history (max lookback / vol
    window) so the portfolio only blends instruments that are actually live."""
    r = close.pct_change()

    # ensemble trend signal in [-1, 1]: average sign of momentum over lookbacks
    sig = sum(np.sign(close / close.shift(lb) - 1.0) for lb in lookbacks) / len(lookbacks)

    # annualized vol estimate (EWMA), floored to avoid blow-ups
    ann_vol = r.ewm(span=vol_window, min_periods=vol_window).std() * np.sqrt(ppy)
    ann_vol = ann_vol.clip(lower=0.02)

    valid = sig.notna() & ann_vol.notna()  # instrument is "live"
    # vol-targeted weight, capped
    w = (sig * (target_vol / ann_vol)).clip(-max_leverage, max_leverage).where(valid, 0.0)

    w_lag = w.shift(1).fillna(0.0)  # trade next bar -> no look-ahead
    gross = w_lag * r
    turnover = (w - w.shift(1)).abs().fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost).where(valid)  # NaN before live

    return pd.DataFrame({"gross": gross, "net": net, "weight": w, "turnover": turnover})


def stats(returns: pd.Series, ppy: int = PPY) -> dict:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0, "calmar": 0.0}
    ann_return = float(r.mean() * ppy)
    ann_vol = float(r.std() * np.sqrt(ppy))
    sharpe = ann_return / ann_vol if ann_vol else 0.0
    curve = (1 + r).cumprod()
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    calmar = ann_return / abs(dd) if dd < 0 else 0.0
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": dd,
        "calmar": calmar,
        "hit_rate": float((r > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-asset trend-following backtest")
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"])
    ap.add_argument("--lookbacks", nargs="+", type=int, default=[21, 63, 126, 252])
    ap.add_argument("--vol-window", type=int, default=33)
    ap.add_argument("--target-vol", type=float, default=0.15, help="annual vol target per instrument")
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--port-target-vol", type=float, default=0.10, help="annual vol target for the blended portfolio")
    ap.add_argument("--oos-split", default="2019-01-01", help="date splitting in-sample / out-of-sample")
    ap.add_argument("--ppy", type=int, default=252, help="periods/year (252 FX business days, 365 crypto)")
    ap.add_argument("--calendar", choices=["B", "D"], default="B", help="B=business days, D=all days (crypto)")
    ap.add_argument("--crypto-cost-bps", type=float, default=8.0, help="round-trip cost for symbols not in the FX table")
    args = ap.parse_args()
    ppy = args.ppy

    # --- per-instrument trend returns ---
    net_cols = {}
    turnovers = {}
    for sym in args.symbols:
        close = load_close(sym)
        res = trend_returns(
            close, args.lookbacks, args.vol_window, args.target_vol, args.max_leverage,
            DEFAULT_COST_BPS.get(sym, args.crypto_cost_bps), ppy=ppy,
        )
        net_cols[sym] = res["net"]
        turnovers[sym] = res["turnover"].mean() * ppy

    net = pd.DataFrame(net_cols).dropna(how="all")
    freq = "D" if args.calendar == "D" else "B"
    idx = pd.date_range(net.index.min(), net.index.max(), freq=freq)
    net = net.reindex(idx)  # keep NaN for not-yet-live instruments

    print(f"Universe: {args.symbols} | {net.index[0].date()} -> {net.index[-1].date()} "
          f"({len(net)} business days)")
    print(f"Signal: ensemble sign-momentum {args.lookbacks} | vol target {args.target_vol:.0%}/inst, "
          f"{args.port_target_vol:.0%} portfolio | costs {DEFAULT_COST_BPS}")

    print("\n=== PER-INSTRUMENT (full sample, net of costs) ===")
    hdr = ["ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate"]
    print(f"{'sym':>8} " + " ".join(f"{h:>10}" for h in hdr) + f"  {'trd/yr':>7}")
    for sym in args.symbols:
        st = stats(net[sym], ppy)
        print(f"{sym:>8} " + " ".join(f"{st[h]:>10.3f}" for h in hdr) + f"  {turnovers[sym]:>7.1f}")

    # --- portfolio: equal-weight blend over LIVE instruments, scaled to vol target ---
    n_live = net.notna().sum(axis=1)
    port = net.mean(axis=1).where(n_live > 0).dropna()
    realized = port.std() * np.sqrt(ppy)
    scale = args.port_target_vol / realized if realized else 1.0
    port = port * scale
    print(f"\n(portfolio scaled x{scale:.2f} to hit {args.port_target_vol:.0%} target vol; "
          f"avg {n_live[n_live > 0].mean():.1f} live instruments)")

    def show(label: str, r: pd.Series) -> None:
        st = stats(r, ppy)
        print(f"  {label:18}: Sharpe {st['sharpe']:+.2f} | ann_ret {st['ann_return']:+.1%} | "
              f"vol {st['ann_vol']:.1%} | maxDD {st['max_dd']:.1%} | Calmar {st['calmar']:.2f} | "
              f"hit {st['hit_rate']:.1%}")

    print("\n=== PORTFOLIO ===")
    show("full sample", port)
    split = pd.Timestamp(args.oos_split)
    show(f"in-sample <{args.oos_split[:4]}", port[port.index < split])
    show(f"OOS >={args.oos_split[:4]}", port[port.index >= split])

    print("\n=== PER-YEAR portfolio return (net) ===")
    by_year = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)
    for yr, ret in by_year.items():
        bar = "#" * int(abs(ret) * 100 / 2)
        print(f"  {yr}: {ret:+7.1%} {'+' if ret>=0 else '-'}{bar}")
    print(f"  positive years: {int((by_year > 0).sum())}/{len(by_year)}")

    # --- Monte Carlo bootstrap on portfolio daily returns ---
    rng = np.random.default_rng(42)
    r = port.dropna().to_numpy()
    finals = [np.prod(1 + rng.choice(r, len(r), replace=True)) - 1 for _ in range(2000)]
    print("\n=== MONTE CARLO (bootstrap, full horizon) ===")
    print(f"  median total return: {np.median(finals):+.1%} | "
          f"p05 {np.percentile(finals,5):+.1%} | p95 {np.percentile(finals,95):+.1%} | "
          f"P(loss) {np.mean(np.array(finals)<0):.1%}")

    # --- tearsheet ---
    try:
        import quantstats as qs

        REPORTS.mkdir(exist_ok=True)
        out = REPORTS / "tearsheet_trend.html"
        qs.reports.html(port, output=str(out), title="Trend-following portfolio")
        print(f"\n[tearsheet] wrote {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[tearsheet] skipped ({exc!r})")


if __name__ == "__main__":
    main()
