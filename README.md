# Kronos‑MT5‑Forex → a deployed, risk‑managed crypto trend bot

![CI](https://github.com/dali012/kronos-mt5-forex/actions/workflows/ci.yml/badge.svg)

What started as *"use the Kronos time‑series foundation model to trade forex"* turned
into an honest research project: **the AI forecasting thesis had no edge, so the
project pivoted to a systematic strategy that does** — a diversified crypto
trend‑following book, validated out‑of‑sample, ported to an event‑driven engine,
and shipped to a VPS with full risk controls and a Telegram ops layer.

> **Paper money only.** This trades the Binance **testnet**; the runner refuses to
> start against a live account. Nothing here is financial advice. A backtest — even
> a careful one — is not proof of live profit.

---

## The story (and why it's the interesting part)

1. **Hypothesis:** Kronos (a 24.7M–102M param OHLCV foundation model) forecasts the
   next candles → trade the predicted direction. Built the brain, risk sizing, a
   NautilusTrader strategy, and a cost‑aware walk‑forward harness.
2. **Tested it honestly** with an engine‑free *forecast‑quality* diagnostic (does
   `sign(predicted_return)` beat chance after costs?). Ran it on a free Kaggle GPU,
   driven end‑to‑end from the CLI.
3. **Verdict: no edge.** Across EUR/USD, GBP/USD, USD/JPY and BTC, at M15→1H→4H→daily,
   with two model sizes, with/without volume — directional IC ≈ 0. A promising 4H
   EUR/USD result **failed to replicate** on the other pairs (classic
   multiple‑comparisons false positive). The walk‑forward gate did its job and said *no*.
4. **Pivot to documented edges.** FX trend and FX mean‑reversion were also dead
   (~0 Sharpe over 20y). The one real, robust signal: **trend‑following on crypto.**
5. **Built + stress‑tested** a diversified crypto trend book (12 coins, vol‑targeted,
   ensemble momentum), then **ported it to NautilusTrader** for execution‑realistic
   validation, then **deployed it to a VPS** on the Binance futures testnet with
   exchange‑native stops, a portfolio drawdown kill‑switch, and a FastAPI + Telegram
   companion for monitoring and remote control.

## Honest results

| Approach | Out‑of‑sample verdict |
|---|---|
| Kronos direction (any timeframe / instrument / model size) | ❌ no edge (IC ≈ 0) |
| FX trend‑following (EUR/GBP/JPY, 20y) | ❌ ~0 Sharpe |
| FX mean‑reversion (EUR–GBP pair) | ❌ ~0 Sharpe |
| **Crypto trend‑following (12 coins, diversified)** | ✅ full Sharpe **0.89**, OOS≥2022 **0.45** |
| Same, in the NautilusTrader engine (real fills/fees/slippage) | ✅ Sharpe **1.04**, OOS **0.62**, maxDD −12% |

**Forward‑looking expectation is modest:** Sharpe ~0.5, ~5–8%/yr, deep (≥12%)
drawdowns, cost‑sensitive (dies past ~40 bps round‑trip), and crypto‑tailwind /
survivorship‑bias caveats apply. It's a *real, small* systematic edge — not a money
printer — which is the honest ceiling for a retail‑accessible strategy.

### Realism / robustness controls (measured, not assumed)

Each control is **off by default** (defaults reproduce the headline Sharpe 1.05).
Measured in the NautilusTrader engine, 12 coins, 2019→, 8 bps round‑trip + slippage:

| Toggle | Sharpe | maxDD | Calmar | Effect |
|---|---|---|---|---|
| Baseline | 1.05 | −12.3% | 0.62 | reference |
| + Funding costs | 0.89 | −13.6% | 0.50 | perp funding is a **real ~0.15‑Sharpe drag** — now modeled, not hidden |
| + Vol‑adaptive stop | 1.06 | −11.8% | 0.65 | best stop variant (beats fixed 20%): lower DD, higher Calmar |
| + Correlation scaling | 1.05 | −12.1% | 0.63 | trims gross when the book gets crowded |
| + Cost‑aware rebalance | 1.06 | −12.3% | 0.62 | skips sub‑cost rebalances; ~neutral, less churn |
| + Patient‑limit entries | 1.05 | −12.3% | 0.62 | no effect on daily‑bar sim (only shows on live microstructure) |
| **Kill‑switch @ 10% DD** | 0.73 | **−10.1%** | 0.38 | caps drawdown as configured (trades return for safety) |

Takeaways: **funding is the one that actually moves the number** — including it makes
the backtest *more honest*, not better. Vol‑stop and correlation scaling are small net
positives; cost‑aware rebalance is cheap insurance; patient‑limit is untestable offline
(verify on testnet). Reproduce any row with `backtest/trend_engine.py` (see `--help`);
funding rows need `scripts/download_funding.py` first.

## Architecture

```
 research            validation              live (VPS)                ops
 ─────────           ──────────              ──────────                ───
 backtest/trend.py   backtest/trend_engine   run_trend_binance ──┐    companion/api.py
 (vectorized,        (NautilusTrader,        (NautilusTrader     │    FastAPI dashboard +
  cost-aware,        realistic fills,        TradingNode on      │    JSON API + Telegram
  OOS + stress)      MTM equity)             Binance testnet)    │    alerts & commands
                                              │                  │      ▲
 forecast_eval.py    walk_forward.py          ├─ TrendStrategy ×N│      │ reads
 (the diagnostic     (Monte Carlo,            ├─ RiskManager     ├──────┘ writes
  that killed the    tearsheets)              ├─ Companion ──────┘  logs/companion.db
  Kronos thesis)                              └─ exchange-native      (SQLite, WAL)
                                                 stop-loss orders
```

- **Strategy** — vol‑targeted ensemble trend‑following; each coin sized to constant
  risk, blended into a portfolio. Exits on signal flip / vol resize.
- **Risk** — per‑position **exchange‑native reduce‑only stop‑losses** (survive a
  crash/disconnect because they live on the exchange) + a **portfolio drawdown
  kill‑switch** (flatten + halt, persisted so it survives restarts). Optional,
  off‑by‑default realism controls (funding costs, vol‑adaptive stops, correlation
  scaling, cost‑aware rebalancing, patient‑limit entries) are each backtested above
  and toggled in `.env`.
- **Companion** — decoupled via SQLite so neither process can crash the other:
  dashboard, equity curve, trade log, bearer‑auth API, Telegram alerts (fills,
  drawdown, **bot‑down via heartbeat**), and **two‑way Telegram control**
  (`/status /pnl /chart /stops /kill …` with inline confirm buttons).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate

# Research / backtest (needs torch + Kronos only for the forecast diagnostic)
pip install -r requirements.txt
python -m backtest.trend --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT --calendar D --ppy 365

# Live paper bot + companion (lightweight — no torch/GPU)
pip install -r requirements-live.txt
cp .env.example .env            # add Binance TESTNET keys (+ optional Telegram / API token)
python -m kronos_mt5.live.run_trend_binance      # the bot
python -m kronos_mt5.companion.run_api           # dashboard + Telegram, separate process
```

VPS deployment (systemd, auto‑restart, Telegram setup) → **[deploy/README.md](deploy/README.md)**.
GPU walk‑forward on Kaggle from the CLI → **[kaggle/](kaggle/)**.

## Repo layout

| Path | What |
|---|---|
| `src/kronos_mt5/brain/` | Kronos forecasting wrapper (`forecast() -> ForecastResult`) |
| `src/kronos_mt5/strategies/` | `TrendStrategy`, `RiskManager` (NautilusTrader) |
| `src/kronos_mt5/live/` | Binance‑testnet `TradingNode` runner |
| `src/kronos_mt5/companion/` | SQLite store, FastAPI dashboard/API, Telegram bot |
| `backtest/` | vectorized + engine backtests, walk‑forward, `forecast_eval` diagnostic |
| `scripts/`, `kaggle/`, `deploy/` | data fetchers, GPU runner, VPS deploy |

## Tech

Python · NautilusTrader (event‑driven trading engine) · PyTorch + Kronos (the
forecasting experiment) · pandas/numpy · FastAPI + httpx · SQLite · systemd · Telegram
Bot API · Kaggle GPU (CLI‑driven). Tests: `pytest`; lint: `ruff`; CI via GitHub Actions.

## Takeaway

The most valuable part of this project is the part that said **"no."** A disciplined
out‑of‑sample gate killed an appealing AI thesis before a single line of the hard
live‑adapter code was written, and pointed the effort at something with a real (if
modest) edge — which then got built, risk‑managed, and deployed properly.
