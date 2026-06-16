#!/usr/bin/env bash
# Unattended runner with auto-restart for the crypto trend bot (Binance testnet).
# Use this under tmux/nohup, or prefer the systemd unit in deploy/ for a VPS.
#
#   nohup bash scripts/run_live.sh >/dev/null 2>&1 &
#
# Restarts the bot if it exits (crash, network drop). The bot reconciles its
# existing positions on restart and re-asserts protective stops, so restarting
# is safe.
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="src:."
mkdir -p logs

PY="${PYTHON:-.venv/bin/python}"
SUP="logs/supervisor.log"
backoff=15

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$SUP"; }

log "supervisor starting (python=$PY)"
while true; do
  log "launching kronos_mt5.live.run_trend_binance"
  "$PY" -m kronos_mt5.live.run_trend_binance >>logs/stdout.log 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    log "process exited cleanly (code 0); stopping supervisor"
    break
  fi
  log "process exited (code $code); restarting in ${backoff}s"
  sleep "$backoff"
done
