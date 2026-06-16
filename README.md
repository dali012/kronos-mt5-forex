# Kronos + NautilusTrader + MetaTrader 5 — Forex Bot Scaffold

A scaffold for an automated forex system that uses **Kronos** (a financial
time-series foundation model) as the forecasting "brain", **NautilusTrader** as
the event-driven trading engine (signal → risk → execution → monitoring), and a
**custom MetaTrader 5 adapter** so the engine can run on a broker's MT5 demo
account.

> **Status:** scaffold only. Most files are interface skeletons with `TODO`s.
> The intended workflow is to hand this folder to Claude Code and build it out
> in the order described in `CLAUDE.md`.

---

## Why this shape

The downstream pipeline (turning a forecast into sized, risk-managed, executed
orders with monitoring) is a solved problem and is exactly what NautilusTrader
provides. The *one* part that does **not** ship out of the box is an MT5
adapter — NautilusTrader has stable adapters for Interactive Brokers, Binance,
etc., but not MetaTrader. That adapter is the core engineering task here.

```
                    ┌─────────────────────────────────────────────┐
                    │            NautilusTrader engine             │
   MT5 terminal     │                                              │
  (demo account) ───┼─► MT5 Data Client ─► Bars/Quotes ─┐          │
        ▲           │                                    ▼          │
        │           │                          KronosStrategy       │
        │           │                    (rolling OHLCV buffer)     │
        │           │                                    │          │
        │           │                                    ▼          │
        │           │                          Kronos brain         │
        │           │                    forecast next N candles    │
        │           │                                    │          │
        │           │                                    ▼          │
        │           │                     Signal (cost-aware filter)│
        │           │                                    │          │
        │           │                                    ▼          │
        │           │                     Risk / position sizing    │
        │           │                                    │          │
        │           │                                    ▼          │
        └───────────┼──◄─ MT5 Execution Client ◄─ Orders (bracket)  │
                    │                                              │
                    │   PortfolioAnalyzer + tearsheets (monitoring)│
                    └─────────────────────────────────────────────┘
```

## Components

| Module | Responsibility |
|---|---|
| `src/kronos_mt5/brain/` | Load Kronos, expose `forecast(df) -> ForecastResult` |
| `src/kronos_mt5/adapter/` | Custom NautilusTrader adapter for MT5 (data + execution + instruments) |
| `src/kronos_mt5/strategies/` | `KronosStrategy`: buffer bars, forecast, build signal, size, submit orders |
| `src/kronos_mt5/risk/` | Position sizing + cost-aware signal filter |
| `src/kronos_mt5/live/` | Wire a `TradingNode` for the MT5 demo account |
| `backtest/` | CSV-driven backtest + walk-forward validation harness |
| `scripts/` | Pull historical FX bars from MT5; fetch the Kronos model/repo |
| `config/` | Env-driven settings (credentials, symbol, model, risk params) |

## Hard constraints to know up front

1. **`MetaTrader5` is Windows-only.** The official `MetaTrader5` pip package
   talks to a locally installed MT5 terminal via a synchronous, blocking API and
   only runs on Windows (or Windows under Wine). NautilusTrader's live clients
   are async. The adapter must run MT5 calls off the event loop (thread executor
   or a sidecar process). See `adapter/mt5_client.py`.
2. **Backtesting does not need the adapter.** In backtest, data comes from CSV;
   the data/execution clients are bypassed. Build and validate the
   brain + strategy + risk in backtest *first*, then build the adapter for live.
3. **Kronos context cap is 512** for `Kronos-small`/`Kronos-base` (2048 for
   `Kronos-mini`). Keep `lookback <= 512`.
4. **FX has no real volume.** Kronos was trained with OHLCV. Use MT5 tick volume
   as a proxy, or use the volume-free prediction path (Kronos ships a
   `prediction_wo_vol_example.py`).
5. **Kronos inference is not free.** Don't call it on every tick. Forecast on a
   cadence (e.g. once per closed bar) and run it off the engine thread in live.

## Setup (target state, after build-out)

```bash
# 1. Python 3.10+
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Fetch the Kronos model code + weights (see script)
bash scripts/setup_kronos.sh

# 4. Configure credentials (NEVER commit the real .env)
cp .env.example .env   # then fill in MT5 demo login/password/server

# 5. Validate offline first
python scripts/download_data.py            # pull historical bars -> data/*.csv
python backtest/run_backtest.py            # single-window backtest
python backtest/walk_forward.py            # rolling out-of-sample validation

# 6. Only after the stats hold up: live demo
python -m kronos_mt5.live.run_demo
```

## Definition of done (per stage)

- **Stage 1 — Brain:** `forecast()` returns a usable forecast from real FX CSV.
- **Stage 2 — Backtest:** strategy runs end-to-end on CSV; tearsheet produced.
- **Stage 3 — Validation:** walk-forward harness reports cost-adjusted metrics
  (expectancy, Sharpe, max DD, profit factor) across multiple OOS windows.
- **Stage 4 — Adapter:** instruments load from MT5; bars stream; a manual test
  order fills on the **demo** account and reconciles.
- **Stage 5 — Live demo:** `run_demo.py` trades the demo account unattended with
  monitoring/logging.

## Important disclaimer

This is research/educational scaffolding for **demo (paper) trading**. Live
forex trading carries substantial risk of loss. A profitable backtest is not
evidence of a profitable live system — costs, slippage, and overfitting routinely
erase apparent edges. Do not connect real capital until a strategy survives
walk-forward validation with realistic cost modeling. Nothing here is financial
advice.
