"""Market-neutral mean-reversion (pairs) backtest, cost-aware, no look-ahead.

Complements trend.py: FX *trend* is dead, but two highly-correlated majors
(EURUSD, GBPUSD) tend to revert to a common spread. We trade the spread's
z-score: short it when stretched high, long it when stretched low, market-neutral.

Design:
  - rolling hedge ratio beta (EURUSD on GBPUSD) over `beta_window`.
  - spread = log(EUR) - beta*log(GBP); z = (spread - mean)/std over `z_window`.
  - stateful threshold: enter at |z| > entry, exit at |z| < exit (fade the move).
  - pnl = pos_lag * (r_eur - beta*r_gbp); costs on both legs at each change.

Run:  python backtest/meanrev.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.trend import DEFAULT_COST_BPS, PPY, load_close, stats


def pair_returns(
    a: str,
    b: str,
    beta_window: int,
    z_window: int,
    entry: float,
    exit_: float,
) -> pd.Series:
    """Daily net return of the market-neutral mean-reversion spread a~b."""
    ca, cb = load_close(a), load_close(b)
    df = pd.concat([ca, cb], axis=1).dropna()
    df.columns = ["a", "b"]
    la, lb = np.log(df["a"]), np.log(df["b"])

    # rolling hedge ratio beta = cov(la,lb)/var(lb)
    cov = la.rolling(beta_window).cov(lb)
    var = lb.rolling(beta_window).var()
    beta = (cov / var).clip(0.2, 5.0)

    spread = la - beta * lb
    z = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()

    # stateful entry/exit on z (fade): pos in {-1,0,+1} for the spread
    pos = np.zeros(len(z))
    zv = z.to_numpy()
    cur = 0.0
    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            cur = 0.0
        elif cur == 0.0:
            if zi > entry:
                cur = -1.0   # spread rich -> short spread
            elif zi < -entry:
                cur = 1.0    # spread cheap -> long spread
        else:  # in a position, exit when reverted
            if abs(zi) < exit_:
                cur = 0.0
        pos[i] = cur
    pos = pd.Series(pos, index=df.index)

    ra, rb = df["a"].pct_change(), df["b"].pct_change()
    spread_ret = ra - beta * rb  # return of 1 unit long spread
    gross = pos.shift(1).fillna(0.0) * spread_ret

    # costs: both legs trade on a position change; notional ~ (1 + beta)
    cost_unit = (DEFAULT_COST_BPS.get(a, 3.0) + beta * DEFAULT_COST_BPS.get(b, 3.0)) / 1e4
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * cost_unit
    return (gross - cost).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean-reversion pairs backtest")
    ap.add_argument("--a", default="EURUSD")
    ap.add_argument("--b", default="GBPUSD")
    ap.add_argument("--beta-window", type=int, default=120)
    ap.add_argument("--z-window", type=int, default=63)
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", dest="exit_", type=float, default=0.5)
    ap.add_argument("--oos-split", default="2019-01-01")
    args = ap.parse_args()

    r = pair_returns(args.a, args.b, args.beta_window, args.z_window, args.entry, args.exit_)
    # scale to 10% annual vol for comparability
    realized = r.std() * np.sqrt(PPY)
    r = r * (0.10 / realized) if realized else r

    print(f"Mean-reversion pair {args.a}~{args.b} | beta_win {args.beta_window} z_win {args.z_window} "
          f"entry {args.entry} exit {args.exit_}")
    split = pd.Timestamp(args.oos_split)
    for label, seg in [("full", r), (f"in-sample <{args.oos_split[:4]}", r[r.index < split]),
                       (f"OOS >={args.oos_split[:4]}", r[r.index >= split])]:
        st = stats(seg)
        print(f"  {label:18}: Sharpe {st['sharpe']:+.2f} | ann_ret {st['ann_return']:+.1%} | "
              f"vol {st['ann_vol']:.1%} | maxDD {st['max_dd']:.1%} | hit {st.get('hit_rate',0):.1%}")

    by_year = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    print(f"  positive years: {int((by_year > 0).sum())}/{len(by_year)}")


if __name__ == "__main__":
    main()
