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
- **logs/supervisor.log** — restarts (only present with the tmux wrapper).
- Binance testnet UI: 8 positions + 8 BUY-above reduce-only stops (one per coin).

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
- **Dashboard**: equity card + total/unrealized PnL, open positions, equity chart, recent fills, live/HALTED status.
- **⛔ KILL button** (or `curl -X POST http://<vps-ip>:8000/kill?token=...`): flattens + halts. Persisted in the DB, so it **survives a bot restart** — the bot stays halted until you hit **Resume**.
- **Telegram alerts**: each fill, drawdown crossing `API_DD_WARN_PCT`, and **bot-down** (no heartbeat for `API_BOT_DOWN_SECS` — detected by the API, since a dead bot can't alert).
- **Telegram commands (two-way control — no URL/token needed)**: once `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are set, message your bot:
  - `/status` — live/halted + equity + uPnL
  - `/positions` — open positions
  - `/pnl` — total & unrealized P&L
  - `/kill` — flatten + halt (panic)  ·  `/resume` — re-enable
  - `/help`
  Only **your** chat id is obeyed (that's the auth). Optional: send BotFather `/setcommands` and paste the list to get a tappable command menu.
- Open only the needed port; on a public VPS set `API_TOKEN` and consider a firewall / reverse-proxy with auth.

## 6. Safe restart / stop
- Restart is safe: the bot reconciles existing positions and re-asserts stops.
  - systemd: `sudo systemctl restart kronos-trend`
- Stop (flattens on graceful shutdown):
  - systemd: `sudo systemctl stop kronos-trend`
  - tmux wrapper: Ctrl-C inside the tmux session.
- Remote halt without stopping the process: the **KILL** button / `/kill` endpoint.

## Notes
- TESTNET only — the runner refuses `BINANCE_ENVIRONMENT=LIVE` by design.
- Keep the VPS clock synced (`timedatectl set-ntp true`) — Binance rejects
  requests with skewed timestamps.
- This is a forward test on paper money. Review monthly; do not point at real
  funds until live behavior matches the backtest over a meaningful period.
