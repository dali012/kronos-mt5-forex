"""Pull historical FX bars from MT5 into data/<SYMBOL>_<TF>.csv for backtesting.

Windows-only (needs the MT5 terminal). Output columns must match what the
backtest bar wrangler and the Kronos brain expect:
  timestamps, open, high, low, close, volume   (volume = MT5 tick volume)

TODO(claude-code):
  import MetaTrader5 as mt5
  mt5.initialize(...); mt5.login(...)
  rates = mt5.copy_rates_range(symbol, timeframe, start, end)
  df = pd.DataFrame(rates); rename 'time'->'timestamps', 'tick_volume'->'volume'
  df.to_csv(out_path, index=False)
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
