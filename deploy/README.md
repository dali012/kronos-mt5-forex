# Deploying the crypto-trend bot to a VPS (Binance testnet)

The live bot is lightweight — **no GPU, no torch, no Kronos**. A $5/mo VPS
(1 vCPU / 1 GB) is plenty. It trades **daily** bars, so it's nearly idle.

## 1. Server prep
```bash
sudo apt update && sudo apt install -y python3-venv git
git clone <your-repo-url> kronos-mt5-forex      # or scp the project up
cd kronos-mt5-forex
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-live.txt   # ~lightweight, no torch
```

## 2. Credentials (.env — never commit)
```bash
cp .env.example .env
nano .env   # set BINANCE_API_KEY / BINANCE_API_SECRET (FUTURES testnet keys),
            # keep BINANCE_ENVIRONMENT=TESTNET
```
Tune risk in `.env` if desired: `BINANCE_STOP_PCT`, `BINANCE_MAX_DRAWDOWN`,
`BINANCE_TARGET_VOL`, `BINANCE_HEARTBEAT_SECS`.

### Recommended forward-test config

The realism toggles are **off by default** (defaults reproduce the headline
backtest). Based on the feature matrix in the root [README](../README.md#realism--robustness-controls-measured-not-assumed),
the recommended set for a testnet forward test is:

```ini
# Safe net-positives (backtested: lower DD / less churn, no downside):
BINANCE_USE_VOL_STOP=true            # vol-adaptive stop — best stop variant
BINANCE_USE_TRAILING_STOP=true       # native trail arms near +1R; hard stop remains
BINANCE_USE_CORRELATION_SCALING=true # trim gross when the book gets crowded
BINANCE_COST_AWARE_REBALANCE=true    # skip sub-cost rebalances

# Live-only paths — enable to OBSERVE (a daily-bar backtest can't exercise them):
BINANCE_USE_PATIENT_LIMIT=true       # post-only entry; MARKET_FALLBACK=true so it can't hang
BINANCE_FUNDING_FILTER_ENABLED=true  # fights the real ~0.15-Sharpe funding drag

BINANCE_FLATTEN_ON_STOP=false        # keep positions across restarts (stops live on the exchange)
```

Rationale: the first three are measured net-positives. The last two are the only
code paths a daily-bar backtest *cannot* test (limit fills/timeouts, live funding
polling) — turn them on specifically to watch them behave on paper money before
trusting them. The full annotated block is in [`.env.example`](../.env.example).
The kill-switch (`BINANCE_MAX_DRAWDOWN=0.20`) demonstrably caps drawdown — keep it on.

## 3a. Run under systemd (recommended — auto-restart + start on boot)
Run these **from inside the repo dir** (it injects the real path, so it works
whether the repo is at /root/dev/... or /home/you/...):
```bash
REPO=$(pwd)
sudo cp deploy/kronos-trend.service /etc/systemd/system/
sudo sed -i "s#CHANGE_REPO_PATH#$REPO#g; s/CHANGE_USER/$USER/g" /etc/systemd/system/kronos-trend.service
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-trend
journalctl -u kronos-trend -f            # follow live logs (stdout -> journal)
```
Output goes to the journal; the bot also writes rotating logs to `logs/`.

## 3b. Or run under tmux + the restart wrapper (simpler, no root)
```bash
tmux new -s trend
bash scripts/run_live.sh                 # auto-restarts on crash
# detach: Ctrl-b then d   |   reattach: tmux attach -t trend
```

## 4. What to watch
- **logs/stdout.log** — startup, warm-up, rebalances, HEARTBEAT lines.
- **logs/kronos_trend*.log** — rotating engine logs (100 MB × 10).
- **logs/live_equity.csv** — hourly `timestamp,equity,cash,unrealized_pnl,n_open`.
  This is your forward-test record — plot it / compare to backtest after weeks.
- **logs/companion.db** — immutable starting-equity baseline, per-execution fills
  and slippage, Binance income rows (`REALIZED_PNL`, `COMMISSION`, `FUNDING_FEE`,
  transfers/rebates), and minute-by-minute attributed performance. `/pnl` and
  `/api/performance` show the breakdown plus an unexplained reconciliation residual.
  The income ledger refreshes every `BINANCE_INCOME_POLL_SECS` (default five minutes).
- **logs/supervisor.log** — restarts (only present with the tmux wrapper).
- Binance testnet UI: 8 positions + 8 hard reduce-only stops (one per coin).
  With `BINANCE_USE_TRAILING_STOP=true`, expect one additional dormant/active
  `TRAILING_STOP_MARKET` per position; `/stops` shows activation and callback levels.
- The bot subscribes to Binance's 1-second futures mark stream. The dashboard PnL
  and portfolio drawdown therefore update from live marks rather than daily bars.
  If any configured mark is older than `BINANCE_MARK_STALE_SECS`, the watchdog
  blocks new entries (and pending patient entries), keeps positions/stops intact,
  and reports `MARKS STALE` until every stream recovers.

## 5. Companion dashboard + Telegram alerts + remote kill (optional but recommended)
A separate FastAPI process gives a live dashboard (PnL, positions, equity curve,
trade log), Telegram alerts, and a remote panic switch. It reads the bot's SQLite
store — fully decoupled, so neither process can crash the other.

```bash
# Telegram (optional): create a bot via @BotFather, get your chat id from @userinfobot,
# put TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env. Set API_TOKEN too on a public VPS.
REPO=$(pwd)
sudo cp deploy/kronos-companion.service /etc/systemd/system/
sudo sed -i "s#CHANGE_REPO_PATH#$REPO#g; s/CHANGE_USER/$USER/g" /etc/systemd/system/kronos-companion.service
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-companion
```
Open `http://<vps-ip>:8000/` (append `?token=<API_TOKEN>` if you set one).
- **Dashboard**: starting/current equity, net/realized/unrealized PnL, commissions,
  funding, measured slippage, reconciliation residual, positions, chart, and fills.
- **⛔ KILL button** (or `curl -X POST http://<vps-ip>:8000/kill?token=...`): flattens + halts. Persisted in the DB, so it **survives a bot restart** — the bot stays halted until you hit **Resume**.
- **Telegram alerts**: each fill, drawdown crossing `API_DD_WARN_PCT`, and **bot-down** (no heartbeat for `API_BOT_DOWN_SECS` — detected by the API, since a dead bot can't alert).
- **Telegram commands (two-way control — no URL/token needed)**: once `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are set, message your bot (only **your** chat id is obeyed = the auth):
  - `/status` `/positions` `/pnl` `/equity` (today/7d/all-time) `/chart` (equity sparkline)
  - `/stops` `/orders` — protective orders + stop levels
  - `/halt` (freeze, keep positions) · `/flatten` (close all, keep trading) · `/close BTC` (one)
  - `/kill` (asks for **⚠️ Confirm/Cancel** inline buttons) · `/resume`
  - `/help` shows tappable quick-action buttons.
  - Pushes: a **daily summary** at `API_DAILY_SUMMARY_HOUR` UTC, plus fill / drawdown / bot-down alerts; stop-outs alert distinctly (🛑 STOPPED OUT).
  - Optional: send BotFather `/setcommands` to get a tappable command menu — paste:
    ```
    status - live/halted + equity
    positions - open positions
    pnl - total & unrealized P&L
    equity - today / 7d / all-time deltas
    chart - equity sparkline
    stops - stop-loss levels
    orders - open orders
    halt - freeze (keep positions)
    flatten - close all (keep trading)
    close - close one position (e.g. /close BTC)
    kill - flatten + halt (panic)
    resume - re-enable trading
    help - command menu
    ```
- Open only the needed port; on a public VPS set `API_TOKEN` and consider a firewall / reverse-proxy with auth.

## 6. Safe restart / stop
- Restart is safe: the bot reconciles existing positions and re-asserts stops.
  - systemd: `sudo systemctl restart kronos-trend`
- Stop: leaves positions open by default (exchange-native stops keep protecting
  them); set `BINANCE_FLATTEN_ON_STOP=true` to force-exit on graceful shutdown.
  - systemd: `sudo systemctl stop kronos-trend`
  - tmux wrapper: Ctrl-C inside the tmux session.
- Remote halt without stopping the process: the **KILL** button / `/kill` endpoint.

## Notes
- TESTNET only — the runner refuses `BINANCE_ENVIRONMENT=LIVE` by design.
- Keep the VPS clock synced (`timedatectl set-ntp true`) — Binance rejects
  requests with skewed timestamps.
- This is a forward test on paper money. Review monthly; do not point at real
  funds until live behavior matches the backtest over a meaningful period.
