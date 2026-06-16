# CLAUDE.md — Build instructions for Claude Code

You are building an automated forex trading system from this scaffold. Read this
file fully before writing code. Read `README.md` for architecture.

## Mission

Turn this skeleton into a working system that:
1. Uses **Kronos** to forecast forex candlesticks.
2. Uses **NautilusTrader** as the engine for signal → risk → execution → monitoring.
3. Connects to a broker's **MetaTrader 5 demo account** via a custom adapter and
   trades unattended.

## Build in this order (do NOT jump to live trading)

The single most important rule: **everything must work and be validated in
backtest before you touch the live MT5 adapter.** The adapter is the hardest,
most error-prone part and it is useless if the brain/strategy has no edge.

1. **Brain** (`src/kronos_mt5/brain/kronos_predictor.py`)
   - Implement the Kronos wrapper. Load tokenizer + model from Hugging Face
     (`NeoQuasar/Kronos-Tokenizer-base`, `NeoQuasar/Kronos-small`).
   - Input: a pandas OHLCV DataFrame + timestamps. Output: a `ForecastResult`
     with the predicted path and a scalar expected return over the horizon.
   - Handle the FX no-volume case (tick volume proxy or volume-free path).
   - Write a smoke test that forecasts from a real CSV slice.

2. **Risk + signal** (`src/kronos_mt5/risk/sizing.py`)
   - Convert a `ForecastResult` into a discrete signal with a **cost-aware
     filter**: only LONG/SHORT if the expected move exceeds spread + threshold;
     otherwise FLAT.
   - Implement position sizing (fixed-fractional risk per trade, ATR- or
     fixed-distance stop). Pure functions, unit-tested, no engine dependency.

3. **Strategy** (`src/kronos_mt5/strategies/kronos_strategy.py`)
   - A NautilusTrader `Strategy`. Maintain a rolling OHLCV buffer of the last
     `lookback` bars. On each closed bar (on a cadence), call the brain, build a
     signal, size it, and submit a **bracket order** (entry + SL + TP).
   - Subscribe to bars in `on_start`; warm up the buffer with historical bars.

4. **Backtest + walk-forward** (`backtest/`)
   - `run_backtest.py`: single-window backtest on CSV via `BacktestNode`.
   - `walk_forward.py`: roll train/calibrate → test across many OOS windows;
     aggregate cost-adjusted metrics. Use NautilusTrader's `PortfolioAnalyzer`
     and export a QuantStats tearsheet. **This is the deliverable that decides
     whether the project is worth continuing.**

5. **MT5 adapter** (`src/kronos_mt5/adapter/`)
   - Build only after stages 1–4 produce a strategy with a plausible edge.
   - `mt5_client.py`: thin async wrapper around the blocking `MetaTrader5`
     package (run every MT5 call in a thread executor). Handle connect/login,
     reconnect, symbol info, rates/ticks, order_send, positions/deals.
   - `providers.py`: `InstrumentProvider` building Nautilus `CurrencyPair`
     instruments from `mt5.symbol_info` (digits, point, volume_min/step,
     contract size, margin).
   - `data.py`: `LiveMarketDataClient` — poll/stream MT5 rates+ticks, normalize
     to Nautilus `Bar`/`QuoteTick` (timestamps in UNIX epoch **nanoseconds**).
   - `execution.py`: `LiveExecutionClient` — map Nautilus order commands to MT5
     `order_send` requests; poll positions/deals to emit fill/position events;
     implement reconciliation on startup.
   - `config.py` + `factories.py`: client config and factories for the
     `TradingNode`.

6. **Live demo** (`src/kronos_mt5/live/run_demo.py`)
   - Wire a `TradingNode` with the MT5 data + execution clients and the strategy,
     pointed at the demo account from `.env`. Add logging + a heartbeat.

## API correctness — verify, don't guess

NautilusTrader's API evolves across versions. The skeletons here are
**directionally correct but not guaranteed to match the installed version.**
Before implementing each NautilusTrader-touching file:
- Check the installed version: `python -c "import nautilus_trader; print(nautilus_trader.__version__)"`
- Verify class names, base classes, and method signatures against the docs for
  that version:
  - Strategies: https://nautilustrader.io/docs/latest/concepts/strategies/
  - Adapters: https://nautilustrader.io/docs/latest/concepts/adapters/
  - Execution: https://nautilustrader.io/docs/latest/concepts/execution/
  - Backtesting: https://nautilustrader.io/docs/latest/concepts/backtesting/
  - Portfolio/stats: https://nautilustrader.io/docs/latest/concepts/portfolio/
- For the adapter, study an existing one (e.g. the Interactive Brokers or
  Betfair adapter in the `nautilus_trader.adapters` package) as the reference
  implementation pattern. Mirror its structure.
- For Kronos, the canonical usage is in the upstream repo's
  `examples/prediction_example.py` and `examples/prediction_wo_vol_example.py`.

If a skeleton signature conflicts with the real API, **the real API wins** —
update the skeleton.

## Guardrails (hard rules)

- **Demo only.** Do not add live-account credentials or a live-trading entry
  point. `run_demo.py` is the only live runner, and it must refuse to start if
  the configured account/server doesn't look like a demo (assert on server name
  / account type where possible).
- **Never commit secrets.** `.env` is gitignored; only `.env.example` is tracked.
- **Costs are mandatory in backtests.** Configure realistic fill + slippage +
  spread models. A backtest without costs is misleading; treat it as a bug.
- **No look-ahead.** The rolling buffer must only ever contain *closed* bars.
- **Fail safe.** On any adapter disconnect or unexpected state, flatten or halt
  new entries rather than guessing.

## Conventions

- Python 3.10+, type hints everywhere, `ruff` + `black` clean.
- Pure logic (brain, risk) must be unit-testable without a broker or GPU
  (allow a tiny CPU model or a mock).
- Keep secrets and tunables in `config/settings.py` (pydantic-settings), never
  hardcoded.
- Log decisions: every signal should log the forecast, the cost filter outcome,
  the size, and the order ids.

## First task

Start with Stage 1. Implement `kronos_predictor.py`, add a CSV fixture under
`data/`, and make `tests/test_smoke.py` produce a real forecast. Then stop and
report the forecast output before moving to Stage 2.
