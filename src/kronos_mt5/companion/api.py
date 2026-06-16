"""Companion FastAPI app: dashboard + JSON API + remote kill + Telegram alerts.

Runs as a SEPARATE process next to the bot, reading the shared SQLite store.
- GET  /                 dashboard (single-page)
- GET  /api/state        latest equity / positions / alive
- GET  /api/equity       equity curve
- GET  /api/fills        trade log
- POST /kill   /resume   flip the shared control flag (panic switch; survives
                         bot restarts because the bot reads it from the DB)

A background loop sends Telegram alerts: new fills, drawdown-threshold crossings,
and bot-death (heartbeat staleness — which must be detected OUTSIDE the bot).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings
from kronos_mt5.companion import store
from kronos_mt5.companion.notify import Telegram

_bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    """Bearer-token guard. No-op if API_TOKEN is unset (local dev); enforced when set."""
    if not settings.api_token:
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != settings.api_token:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

DB = settings.companion_db
tg = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)


def _alive() -> tuple[bool, float | None]:
    ts = store.get_kv("bot_alive_ts", None, DB)
    if not ts:
        return False, None
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
    return age <= settings.api_bot_down_secs, age


# --- Telegram alerter loop ---------------------------------------------------
async def _alerter() -> None:
    while True:
        try:
            _check_fills()
            _check_drawdown()
            _check_bot_down()
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(settings.api_poll_secs)


def _check_fills() -> None:
    last = int(store.get_kv("alert_last_fill_rowid", "0", DB) or 0)
    new = sorted(store.read_fills(limit=500, after_rowid=last, db_path=DB), key=lambda f: f["rowid"])
    for f in new:
        icon = "🛑 STOPPED OUT" if f.get("kind") == "STOP" else ("🎯 TP" if f.get("kind") == "TP" else "✅ FILL")
        tg.send(f"{icon} {f['symbol']} {f['side']} {f['qty']:g} @ {f['price']:g}")
        store.set_kv("alert_last_fill_rowid", str(f["rowid"]), DB)


def _check_drawdown() -> None:
    eq = store.read_equity(limit=100000, db_path=DB)
    if not eq:
        return
    peak = max(e["equity"] for e in eq)
    cur = eq[-1]["equity"]
    dd = (cur / peak - 1.0) if peak else 0.0
    alerted = store.get_kv("alert_dd", "0", DB) == "1"
    if dd <= -settings.api_dd_warn_pct and not alerted:
        tg.send(f"⚠️ DRAWDOWN {dd:.1%} (equity ${cur:,.0f}, peak ${peak:,.0f})")
        store.set_kv("alert_dd", "1", DB)
    elif dd > -settings.api_dd_warn_pct / 2 and alerted:
        store.set_kv("alert_dd", "0", DB)  # recovered -> re-arm


def _check_bot_down() -> None:
    alive, age = _alive()
    down_alerted = store.get_kv("alert_bot_down", "0", DB) == "1"
    if not alive and not down_alerted:
        tg.send(f"🛑 BOT DOWN — no heartbeat for {int(age) if age else '∞'}s")
        store.set_kv("alert_bot_down", "1", DB)
    elif alive and down_alerted:
        tg.send("🟢 BOT RECOVERED — heartbeat resumed")
        store.set_kv("alert_bot_down", "0", DB)


# --- Telegram two-way control (commands) ------------------------------------
_TG_BASE = (
    f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    if settings.telegram_bot_token
    else None
)
HELP = (
    "<b>🤖 Kronos Trend — commands</b>\n\n"
    "📊 /status — live/halted + equity\n"
    "📦 /positions — open positions\n"
    "💹 /pnl · /equity — P&amp;L &amp; equity deltas\n"
    "📈 /chart — equity sparkline\n"
    "🛡 /stops · /orders — protective orders\n"
    "🧹 /flatten · /close BTC — flatten all / one\n"
    "⏸ /halt — freeze (keep positions)\n"
    "🛑 /kill — flatten + halt  ·  ▶️ /resume\n"
    "ℹ️ /help"
)
QUICK_KB = {"inline_keyboard": [
    [{"text": "📊 Status", "callback_data": "cmd:status"},
     {"text": "📦 Positions", "callback_data": "cmd:positions"}],
    [{"text": "💹 PnL", "callback_data": "cmd:pnl"},
     {"text": "🛡 Stops", "callback_data": "cmd:stops"}],
    [{"text": "📈 Chart", "callback_data": "cmd:chart"},
     {"text": "⛔ Kill", "callback_data": "cmd:kill"}],
]}
KILL_KB = {"inline_keyboard": [[
    {"text": "🛑 Confirm KILL", "callback_data": "kill:yes"},
    {"text": "Cancel", "callback_data": "kill:no"},
]]}


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _signed(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def _coin(symbol: str) -> str:
    return symbol.split("USDT")[0]  # "BTCUSDT-PERP" -> "BTC"


def _sparkline(vals: list[float], width: int = 24) -> str:
    if len(vals) > width:  # downsample
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


def _equity_summary() -> str:
    eq = store.read_equity(limit=200000, db_path=DB)
    if not eq:
        return "<i>no equity data yet</i>"
    now = eq[-1]["equity"]
    now_ts = datetime.now(timezone.utc)

    def prior(hours: int) -> float:
        cut = now_ts - timedelta(hours=hours)
        p = [e for e in eq if datetime.fromisoformat(e["ts"]) <= cut]
        return p[-1]["equity"] if p else eq[0]["equity"]

    def row(lbl: str, base: float) -> str:
        d = now - base
        pct = (d / base * 100) if base else 0
        return f"{lbl:<8}{_signed(d):>10} ({pct:+.1f}%)"

    body = (f"{'Equity':<8}{_usd(now):>10}\n" + row("Today", prior(24)) + "\n"
            + row("7d", prior(24 * 7)) + "\n" + row("AllTime", eq[0]["equity"]))
    return f"<b>💹 Equity</b>\n<pre>{body}</pre>"


def _chart_text() -> str:
    eq = store.read_equity(limit=200000, db_path=DB)
    if len(eq) < 2:
        return "<i>not enough data for a chart yet</i>"
    vals = [e["equity"] for e in eq]
    pct = ((vals[-1] / vals[0] - 1) * 100) if vals[0] else 0
    return (f"<b>📈 Equity</b>\n<pre>{_sparkline(vals)}\n"
            f"${min(vals):,.0f} … ${max(vals):,.0f}   now ${vals[-1]:,.0f} ({pct:+.1f}%)</pre>")


def _stops_text(snap: dict) -> str:
    stops = [o for o in snap.get("orders", []) if o.get("type") == "STOP_MARKET"]
    if not stops:
        return "<b>🛡 Stops</b>\n<i>none</i>"
    entries = {p["symbol"]: p.get("entry") for p in snap.get("positions", [])}
    head = f"{'COIN':<5}{'STOP':>11}{'DIST':>7}"
    rows = []
    for o in stops:
        e, trig = entries.get(o["symbol"]), o.get("trigger")
        dist = f"{(trig / e - 1) * 100:+.0f}%" if (e and trig) else "–"
        rows.append(f"{_coin(o['symbol']):<5}{(trig or 0):>11.4f}{dist:>7}")
    return f"<b>🛡 Stop-losses ({len(stops)})</b>\n<pre>{head}\n" + "\n".join(rows) + "</pre>"


def _orders_text(snap: dict) -> str:
    orders = snap.get("orders", [])
    if not orders:
        return "<b>📋 Orders</b>\n<i>none</i>"
    rows = "\n".join(
        f"{_coin(x['symbol']):<5}{x['type'][:6]:<7}{x['side'][:4]:<5}{x['qty']:>10.4f}" for x in orders
    )
    return f"<b>📋 Open orders ({len(orders)})</b>\n<pre>{rows}</pre>"


def _daily_summary_text() -> str:
    eq = store.read_equity(limit=200000, db_path=DB)
    snap = store.read_snapshot(DB) or {}
    if not eq:
        return "<b>📅 Daily summary</b>\n<i>no data</i>"
    now = eq[-1]["equity"]
    cut = datetime.now(timezone.utc) - timedelta(hours=24)
    prior = [e for e in eq if datetime.fromisoformat(e["ts"]) <= cut]
    base = prior[-1]["equity"] if prior else eq[0]["equity"]
    day = now - base
    pct = (day / base * 100) if base else 0
    peak = max(e["equity"] for e in eq)
    ddv = (now / peak - 1) * 100 if peak else 0
    n24 = sum(1 for f in store.read_fills(limit=500, db_path=DB)
              if datetime.fromisoformat(f["ts"]) >= cut)
    arrow = "📈" if day >= 0 else "📉"
    body = (f"{'Equity':<10}{_usd(now):>10}\n{'Day PnL':<10}{_signed(day):>10} ({pct:+.1f}%)\n"
            f"{'Drawdown':<10}{ddv:>9.1f}%\n{'Trades24h':<10}{n24:>10}\n"
            f"{'Open':<10}{snap.get('n_open', 0):>10}")
    return f"<b>{arrow} Daily summary</b>\n<pre>{body}</pre>"


def _handle(text: str) -> tuple[str, dict | None]:
    """Map a /command (with optional args) to (reply_html, inline_keyboard|None)."""
    parts = text.lstrip("/").split()
    cmd = parts[0].split("@")[0].lower() if parts else ""
    args = parts[1:]
    snap = store.read_snapshot(DB) or {}

    if cmd in ("start", "help", "menu"):
        return HELP, QUICK_KB
    if cmd == "status":
        alive, _ = _alive()
        mode = store.get_mode(DB)
        state = {"kill": "🛑 KILLED", "halt": "⏸ HALTED"}.get(mode, "🟢 LIVE" if alive else "🔴 DOWN")
        body = (f"{'Equity':<9}{_usd(snap.get('equity', 0)):>11}\n"
                f"{'uPnL':<9}{_signed(snap.get('unrealized', 0)):>11}\n"
                f"{'Open':<9}{snap.get('n_open', 0):>11}")
        return f"<b>🤖 Kronos Trend — {state}</b>\n<pre>{body}</pre>", QUICK_KB
    if cmd == "positions":
        pos = snap.get("positions", [])
        if not pos:
            return "<b>📦 Positions</b>\n<i>flat</i>", None
        head = f"{'COIN':<5}{'QTY':>11}{'uPnL':>8}"
        rows = "\n".join(f"{_coin(p['symbol']):<5}{p['qty']:>11.4f}{_signed(p['unrealized']):>8}"
                         for p in pos)
        return f"<b>📦 Open positions ({len(pos)})</b>\n<pre>{head}\n{rows}</pre>", None
    if cmd == "pnl":
        eq = store.read_equity(limit=200000, db_path=DB)
        base = eq[0]["equity"] if eq else snap.get("equity", 0)
        equity = snap.get("equity", 0)
        total = equity - base
        pct = (total / base * 100) if base else 0
        arrow = "📈" if total > 0 else ("📉" if total < 0 else "➖")
        body = (f"{'Equity':<11}{_usd(equity):>11}\n"
                f"{'Total':<11}{_signed(total):>11} ({pct:+.2f}%)\n"
                f"{'Unrealized':<11}{_signed(snap.get('unrealized', 0)):>11}")
        return f"<b>{arrow} P&amp;L</b>\n<pre>{body}</pre>", None
    if cmd == "equity":
        return _equity_summary(), None
    if cmd == "chart":
        return _chart_text(), None
    if cmd == "stops":
        return _stops_text(snap), None
    if cmd == "orders":
        return _orders_text(snap), None
    if cmd == "halt":
        store.set_mode("halt", DB)
        return "⏸ <b>Halted</b> — positions kept, no new entries (/resume to continue)", None
    if cmd == "resume":
        store.set_mode("run", DB)
        return "▶️ <b>Resumed</b> — trading re-enabled", None
    if cmd == "flatten":
        store.request_flatten("all", DB)
        return "🧹 flatten-all requested (trading stays enabled)", None
    if cmd == "close":
        if not args:
            return "usage: <code>/close BTC</code>", None
        coin = args[0].upper().replace("USDT", "").replace("-PERP", "")
        sym = next((p["symbol"] for p in snap.get("positions", [])
                    if _coin(p["symbol"]).upper() == coin), None)
        if not sym:
            return f"no open position for <b>{coin}</b>", None
        store.request_flatten(sym, DB)
        return f"🧹 closing <b>{coin}</b> requested", None
    if cmd == "kill":
        return "⚠️ <b>Confirm KILL</b> — flatten ALL positions and halt?", KILL_KB
    return f"❓ unknown command <code>/{cmd}</code> — try /help", None


async def _tg_post(client: httpx.AsyncClient, method: str, **params) -> dict:
    if isinstance(params.get("reply_markup"), (dict, list)):
        params["reply_markup"] = json.dumps(params["reply_markup"])
    r = await client.post(f"{_TG_BASE}/{method}", data=params)
    return r.json()


async def _send(client: httpx.AsyncClient, owner: str, text: str, kb: dict | None = None) -> None:
    params = {"chat_id": owner, "text": text, "parse_mode": "HTML"}
    if kb:
        params["reply_markup"] = kb
    await _tg_post(client, "sendMessage", **params)


async def _telegram_commands() -> None:
    """Long-poll Telegram for slash-commands + button taps from the owner only."""
    if not (_TG_BASE and settings.telegram_chat_id):
        return
    owner = str(settings.telegram_chat_id)
    offset = 0
    async with httpx.AsyncClient(timeout=40) as client:
        with contextlib.suppress(Exception):
            await _send(client, owner, "🤖 <b>controller online</b> — /help", QUICK_KB)
        while True:
            try:
                r = await client.get(f"{_TG_BASE}/getUpdates", params={"offset": offset, "timeout": 30})
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    if "callback_query" in upd:
                        await _on_callback(client, owner, upd["callback_query"])
                    else:
                        msg = upd.get("message") or {}
                        text = (msg.get("text") or "").strip()
                        if text.startswith("/") and str((msg.get("chat") or {}).get("id", "")) == owner:
                            await _send(client, owner, *_handle(text))
            except Exception:  # noqa: BLE001
                await asyncio.sleep(5)


async def _on_callback(client: httpx.AsyncClient, owner: str, cq: dict) -> None:
    data = cq.get("data", "")
    chat = str((cq.get("message") or {}).get("chat", {}).get("id", ""))
    with contextlib.suppress(Exception):
        await _tg_post(client, "answerCallbackQuery", callback_query_id=cq["id"])
    if chat != owner:
        return
    if data == "kill:yes":
        store.set_mode("kill", DB)
        await _send(client, owner, "🛑 <b>KILLED</b> — flattening + halting all positions")
    elif data == "kill:no":
        await _send(client, owner, "✅ cancelled — bot still running")
    elif data.startswith("cmd:"):
        await _send(client, owner, *_handle("/" + data[4:]))


async def _daily_summary() -> None:
    """Push a once-a-day digest at API_DAILY_SUMMARY_HOUR UTC (set <0 to disable)."""
    if not tg.enabled or settings.api_daily_summary_hour < 0:
        return
    while True:
        with contextlib.suppress(Exception):
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()
            if now.hour >= settings.api_daily_summary_hour and store.get_kv("last_summary_day", "", DB) != today:
                store.set_kv("last_summary_day", today, DB)
                tg.send(_daily_summary_text())
        await asyncio.sleep(300)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    store.init_db(DB)
    tasks = [
        asyncio.create_task(_alerter()),
        asyncio.create_task(_telegram_commands()),
        asyncio.create_task(_daily_summary()),
    ]
    yield
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t


app = FastAPI(title="Kronos Trend Companion", lifespan=_lifespan)


@app.get("/api/state", dependencies=[Depends(require_auth)])
def api_state() -> JSONResponse:
    snap = store.read_snapshot(DB) or {}
    alive, age = _alive()
    snap["alive"] = alive
    snap["heartbeat_age_secs"] = age
    snap["halted"] = store.is_halted(DB)
    return JSONResponse(snap)


@app.get("/api/equity", dependencies=[Depends(require_auth)])
def api_equity() -> JSONResponse:
    return JSONResponse(store.read_equity(limit=5000, db_path=DB))


@app.get("/api/fills", dependencies=[Depends(require_auth)])
def api_fills() -> JSONResponse:
    return JSONResponse(store.read_fills(limit=200, db_path=DB))


@app.post("/kill", dependencies=[Depends(require_auth)])
def kill() -> dict:
    store.set_mode("kill", DB)
    tg.send("🛑 KILL TRIGGERED via API — bot flattening + halting")
    return {"mode": "kill"}


@app.post("/resume", dependencies=[Depends(require_auth)])
def resume() -> dict:
    store.set_mode("run", DB)
    tg.send("▶️ RESUME via API — bot re-enabled")
    return {"mode": "run"}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Kronos Trend Companion</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{background:#0e1117;color:#e6e6e6;font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:20px}
 h1{font-size:18px;margin:0 0 14px} .muted{color:#8b949e}
 .row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px;min-width:150px}
 .card .k{color:#8b949e;font-size:12px} .card .v{font-size:22px;font-weight:600;margin-top:4px}
 .pos{color:#3fb950} .neg{color:#f85149}
 table{width:100%;border-collapse:collapse;font-size:13px} th,td{text-align:right;padding:6px 10px;border-bottom:1px solid #21262d}
 th:first-child,td:first-child{text-align:left}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
 button{background:#21262d;color:#e6e6e6;border:1px solid #30363d;border-radius:8px;padding:8px 14px;cursor:pointer}
 button.kill{background:#5a1e1e;border-color:#8b2b2b} button.resume{background:#1e4620;border-color:#2b6b2f}
 canvas{max-height:260px}
</style></head><body>
<h1>🤖 Kronos Trend Companion <span id=status class=muted></span></h1>
<div class=row id=cards></div>
<div class=row>
 <button class=kill onclick="ctl('kill')">⛔ KILL (flatten + halt)</button>
 <button class=resume onclick="ctl('resume')">▶ Resume</button>
 <span id=ctlmsg class=muted></span>
</div>
<div class=card style="flex:1"><div class=k>Equity</div><canvas id=chart></canvas></div>
<div class=card style="flex:1;margin-top:16px"><div class=k>Open positions</div><table id=pos></table></div>
<div class=card style="flex:1;margin-top:16px"><div class=k>Recent fills</div><table id=fills></table></div>
<script>
const params=new URLSearchParams(location.search);
let TOKEN = params.get('token') || localStorage.getItem('tok') || '';
if(params.get('token')) localStorage.setItem('tok', TOKEN);   // remember across refreshes
const api=(p,opt={})=>fetch(p,{...opt,headers:{...(TOKEN?{Authorization:'Bearer '+TOKEN}:{}),...(opt.headers||{})}});
let chart;
const fmt=(n,d=2)=>n==null?'–':Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
function cls(n){return n>0?'pos':n<0?'neg':''}
async function ctl(action){
  if(action==='kill' && !confirm('Flatten ALL positions and halt the bot?'))return;
  const r=await api(`/${action}`,{method:'POST'});
  document.getElementById('ctlmsg').textContent = r.ok? `${action} sent` : `failed (${r.status})`;
}
async function refresh(){
 try{
  const sr=await api('/api/state');
  if(sr.status===401){document.getElementById('status').innerHTML='<b style=color:#f85149>401 — add ?token=YOUR_API_TOKEN to the URL</b>';return;}
  const s=await sr.json();
  const eq=await (await api('/api/equity')).json();
  const fills=await (await api('/api/fills')).json();
  const base=eq.length?eq[0].equity:s.equity, pnl=(s.equity||0)-(base||0), pnlpct=base?pnl/base:0;
  const aliveDot=`<span class=dot style="background:${s.alive?'#3fb950':'#f85149'}"></span>`;
  document.getElementById('status').innerHTML=aliveDot+(s.alive?'live':'DOWN')+(s.halted?' · <b style=color:#f85149>HALTED</b>':'');
  document.getElementById('cards').innerHTML=`
   <div class=card><div class=k>Equity</div><div class=v>$${fmt(s.equity)}</div></div>
   <div class=card><div class=k>Total PnL</div><div class="v ${cls(pnl)}">$${fmt(pnl)} (${fmt(pnlpct*100)}%)</div></div>
   <div class=card><div class=k>Unrealized</div><div class="v ${cls(s.unrealized)}">$${fmt(s.unrealized)}</div></div>
   <div class=card><div class=k>Open</div><div class=v>${s.n_open??0}</div></div>
   <div class=card><div class=k>Cash</div><div class=v>$${fmt(s.cash)}</div></div>`;
  document.getElementById('pos').innerHTML='<tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>uPnL</th></tr>'+
    (s.positions||[]).map(p=>`<tr><td>${p.symbol}</td><td>${fmt(p.qty,4)}</td><td>${fmt(p.entry,4)}</td><td class="${cls(p.unrealized)}">$${fmt(p.unrealized)}</td></tr>`).join('');
  document.getElementById('fills').innerHTML='<tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr>'+
    fills.map(f=>`<tr><td>${(f.ts||'').slice(0,19).replace('T',' ')}</td><td>${f.symbol}</td><td class="${f.side=='BUY'?'pos':'neg'}">${f.side}</td><td>${fmt(f.qty,4)}</td><td>${fmt(f.price,4)}</td></tr>`).join('');
  const labels=eq.map(e=>e.ts.slice(5,16).replace('T',' ')), data=eq.map(e=>e.equity);
  if(!chart){chart=new Chart(document.getElementById('chart'),{type:'line',
    data:{labels,datasets:[{data,borderColor:'#3fb950',borderWidth:1.5,pointRadius:0,fill:false}]},
    options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8}},y:{ticks:{color:'#8b949e'}}}}});}
  else{chart.data.labels=labels;chart.data.datasets[0].data=data;chart.update('none');}
 }catch(e){document.getElementById('status').textContent='(api error)';}
}
refresh(); setInterval(refresh,5000);
</script></body></html>"""
