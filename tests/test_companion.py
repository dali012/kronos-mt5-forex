"""Tests for the companion store + API (no running bot required)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import sqlite3
import urllib.parse
from urllib.error import HTTPError
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kronos_mt5.companion import store


@pytest.fixture()
def db(tmp_path):
    p = str(tmp_path / "companion.db")
    store.init_db(p)
    return p


def test_snapshot_roundtrip(db):
    store.write_snapshot(
        {"equity": 10100.0, "cash": 10000.0, "positions": [{"symbol": "BTCUSDT"}]}, db
    )
    snap = store.read_snapshot(db)
    assert snap["equity"] == 10100.0
    assert snap["positions"][0]["symbol"] == "BTCUSDT"
    assert "ts" in snap


def test_equity_series(db):
    for i in range(3):
        store.append_equity(10000 + i, 10000, float(i), i, db)
    eq = store.read_equity(db_path=db)
    assert len(eq) == 3
    assert eq[-1]["equity"] == 10002  # chronological order


def test_fills_idempotent(db):
    assert store.record_fill("oid-1", "BTCUSDT", "SELL", 0.01, 65000.0, db_path=db) is True
    assert store.record_fill("oid-1", "BTCUSDT", "SELL", 0.01, 65000.0, db_path=db) is False  # dup
    fills = store.read_fills(db_path=db)
    assert len(fills) == 1 and fills[0]["symbol"] == "BTCUSDT"


def test_accounting_migrates_old_db(tmp_path):
    p = str(tmp_path / "old.db")
    con = sqlite3.connect(p)
    con.executescript(
        """
        CREATE TABLE snapshot (id INTEGER PRIMARY KEY, ts TEXT, data TEXT);
        CREATE TABLE equity (ts TEXT, equity REAL, cash REAL, unrealized REAL, n_open INTEGER);
        CREATE TABLE fills (
            fill_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT,
            qty REAL, price REAL, kind TEXT
        );
        CREATE TABLE kv (name TEXT PRIMARY KEY, value TEXT, ts TEXT);
        """
    )
    con.commit()
    con.close()

    store.init_db(p)
    con = sqlite3.connect(p)
    fill_cols = {row[1] for row in con.execute("PRAGMA table_info(fills)")}
    equity_cols = {row[1] for row in con.execute("PRAGMA table_info(equity)")}
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"commission", "reference_price", "slippage", "trade_id", "reconciliation"} <= fill_cols
    assert {"realized", "commissions", "funding", "slippage", "total_pnl"} <= equity_cols
    assert "income" in tables
    assert {"closed_positions", "basket_cycles", "incidents", "ops_events", "shadow_targets"} <= tables
    assert {"basket_direction", "average_correlation", "effective_bets", "market_regime"} <= equity_cols


def test_performance_accounting_reconciles_components(db):
    store.append_equity(1000.0, 1000.0, 0.0, 0, db)
    baseline = store.ensure_accounting_baseline(1000.0, 1000.0, 0.0, db)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000) + 1
    rows = [
        ("realized", "REALIZED_PNL", 20.0),
        ("commission", "COMMISSION", -2.0),
        ("funding", "FUNDING_FEE", -1.0),
        ("bonus", "WELCOME_BONUS", 5.0),
    ]
    for income_id, income_type, amount in rows:
        store.record_income(income_id, now_ms, income_type, amount, db_path=db)
    store.record_fill(
        "fill-1",
        "BTCUSDT",
        "BUY",
        1.0,
        101.5,
        reference_price=100.0,
        slippage=1.5,
        commission=0.04,
        db_path=db,
    )

    summary = store.performance_summary(1027.0, 1022.0, 5.0, db)
    assert baseline["equity"] == 1000.0
    assert summary["total_pnl"] == 27.0
    assert summary["realized_pnl"] == 20.0
    assert summary["commissions"] == 2.0
    assert summary["funding_payments"] == -1.0
    assert summary["other_income"] == 5.0
    assert summary["unrealized_change"] == 5.0
    assert summary["slippage"] == 1.5 and summary["slippage_known_fills"] == 1
    assert summary["reconciliation_residual"] == 0.0

    # Baseline is immutable across restarts/snapshots.
    again = store.ensure_accounting_baseline(2000.0, 2000.0, 0.0, db)
    assert again == baseline


def test_shadow_targets_are_idempotent_and_scored(db):
    def row(model, cycle, ts, price, weight):
        return {
            "model": model,
            "cycle_id": cycle,
            "cycle_ts": ts,
            "symbol": "BTCUSDT-PERP",
            "price": price,
            "signal": 1.0,
            "raw_weight": 1.0,
            "target_weight": weight,
            "portfolio_weight": weight,
            "portfolio_scale": weight,
            "funding_rate": 0.0001,
            "funding_scalar": 1.0,
            "cost_bps": 10.0,
        }

    rows = [
        row("live", "run-1:startup", "2025-12-31T23:00:00+00:00", 100.0, 1.0),
        row("live", "100", "2026-01-01T00:00:00+00:00", 100.0, 1.0),
        row("live", "200", "2026-01-02T00:00:00+00:00", 110.0, 1.0),
        row("challenger", "run-1:startup", "2025-12-31T23:00:00+00:00", 100.0, 0.5),
        row("challenger", "100", "2026-01-01T00:00:00+00:00", 100.0, 0.5),
        row("challenger", "200", "2026-01-02T00:00:00+00:00", 110.0, 0.5),
    ]
    assert store.record_shadow_targets(rows, db) == 6
    assert store.record_shadow_targets(rows, db) == 0
    report = store.shadow_report(db)
    assert report["models"]["live"]["cycles"] == 2
    assert report["models"]["live"]["startup_observations"] == 1
    assert report["models"]["live"]["annualized_return"] is None
    assert report["models"]["live"]["total_return"] > 0.09
    assert 0.04 < report["models"]["challenger"]["total_return"] < 0.06
    assert "not simulated" in report["funding_note"]


def test_forward_test_counts_eight_correlated_legs_as_one_basket(db):
    coins = ("BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "LTC", "LINK")
    symbols = [f"{coin}USDT-PERP" for coin in coins]
    store.write_snapshot(
        {
            "equity": 1100.0,
            "performance": {
                "starting_equity": 1000.0,
                "total_pnl": 100.0,
                "commissions": 2.0,
                "funding_payments": -1.0,
                "slippage": 0.5,
            },
        },
        db,
    )
    store.append_equity(
        1000.0, 1000.0, 0.0, 8, db,
        basket_direction="SHORT", market_regime="SHORT_LOW_VOL",
    )
    store.update_basket_cycle("SHORT", symbols, 1000.0, 100.0, db)
    store.update_basket_cycle("SHORT", symbols, 950.0, 100.0, db)
    store.append_equity(
        950.0, 950.0, 0.0, 8, db,
        basket_direction="SHORT", market_regime="SHORT_LOW_VOL",
    )
    store.update_basket_cycle("FLAT", [], 1100.0, None, db)
    store.append_equity(1100.0, 1100.0, 0.0, 0, db, basket_direction="FLAT")

    store.record_closed_position(
        "BTC:1",
        "2026-01-01T00:00:00+00:00",
        "2026-01-02T00:00:00+00:00",
        "BTCUSDT-PERP",
        "SHORT",
        1.0,
        100.0,
        90.0,
        10.0,
        0.1,
        initial_risk=10.0,
        exit_reason="HARD_STOP",
        db_path=db,
    )
    store.record_event("BOT_START", db_path=db)
    store.record_event("BOT_START", db_path=db)
    store.start_incident("MARK_STALE", "ETH", db)
    store.resolve_incident("MARK_STALE", db)

    report = store.forward_test_report(db)
    assert report["completed_basket_cycles"] == 1
    assert report["completed_symbol_positions"] == 1
    assert report["average_r"] == pytest.approx(1.0)
    assert report["maximum_drawdown_pct"] == pytest.approx(-5.0)
    assert report["restart_count"] == 1
    assert report["incidents"]["MARK_STALE"]["count"] == 1
    assert report["directional_regimes_seen"] == ["SHORT"]
    assert report["multiple_directional_regimes_seen"] is False
    assert report["market_regimes_seen"] == ["SHORT_LOW_VOL"]
    assert report["multiple_market_regimes_seen"] is False


def test_binance_income_client_signs_request():
    from kronos_mt5.companion.accounting import BinanceIncomeClient

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'[{"incomeType":"FUNDING_FEE","income":"-0.25","time":123}]'

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    client = BinanceIncomeClient(
        "api-key",
        "secret",
        "https://demo-fapi.binance.com",
        opener=opener,
        clock_ms=lambda: 999,
    )
    rows = client.fetch_income(100)
    assert rows[0]["incomeType"] == "FUNDING_FEE"
    request = captured["request"]
    parsed = urllib.parse.urlsplit(request.full_url)
    params = urllib.parse.parse_qs(parsed.query)
    signature = params.pop("signature")[0]
    unsigned = parsed.query.rsplit("&signature=", 1)[0]
    expected = hmac.new(b"secret", unsigned.encode(), hashlib.sha256).hexdigest()
    assert signature == expected
    assert request.get_header("X-mbx-apikey") == "api-key"
    assert params["startTime"] == ["100"] and captured["timeout"] == 10.0


def test_binance_income_client_includes_http_error_body():
    from kronos_mt5.companion.accounting import BinanceIncomeClient

    def opener(request, timeout):
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"code":-1021,"msg":"Timestamp outside recvWindow"}'),
        )

    client = BinanceIncomeClient(
        "api-key",
        "secret",
        "https://demo-fapi.binance.com",
        opener=opener,
        clock_ms=lambda: 999,
    )
    with pytest.raises(RuntimeError, match=r"HTTP 400.*-1021.*recvWindow"):
        client.fetch_income(100)


def test_fill_reference_price_prefers_decision_tag():
    from kronos_mt5.companion.recorder import Companion

    order = SimpleNamespace(
        tags=["REF_PX=100.25"],
        trigger_price=99.0,
        price=None,
        activation_price=None,
    )
    assert Companion._reference_price(order) == 100.25


def test_companion_records_fill_events_from_order_event_list(db, monkeypatch):
    from kronos_mt5.companion import recorder

    class FakeOrderFilled:
        last_qty = 0.01
        last_px = 65000.0
        trade_id = SimpleNamespace(value="trade-1")
        is_buy = True
        commission = None
        ts_event = 1_750_000_000_000_000_000
        order_side = SimpleNamespace(name="BUY")
        reconciliation = False

    order = SimpleNamespace(
        order_type=SimpleNamespace(name="MARKET"),
        is_reduce_only=False,
        tags=[],
        trigger_price=None,
        price=None,
        activation_price=None,
        events=[FakeOrderFilled()],
        instrument_id=SimpleNamespace(symbol=SimpleNamespace(value="BTCUSDT-PERP")),
        client_order_id=SimpleNamespace(value="order-1"),
    )
    companion = SimpleNamespace(
        cache=SimpleNamespace(orders=lambda: [order]),
        config=SimpleNamespace(db_path=db),
        _reference_price=recorder.Companion._reference_price,
    )
    monkeypatch.setattr(recorder, "OrderFilled", FakeOrderFilled)

    recorder.Companion._record_new_fills(companion)

    fills = store.read_fills(db_path=db)
    assert len(fills) == 1
    assert fills[0]["symbol"] == "BTCUSDT-PERP"
    assert fills[0]["trade_id"] == "trade-1"


def test_reconciliation_fill_is_stored_but_not_alerted(tmp_path, monkeypatch):
    from kronos_mt5.companion import api

    p = str(tmp_path / "reconciliation.db")
    store.init_db(p)
    store.record_fill(
        "old-fill",
        "BTCUSDT",
        "BUY",
        0.01,
        65000.0,
        reconciliation=True,
        db_path=p,
    )
    sent = []
    monkeypatch.setattr(api, "DB", p)
    monkeypatch.setattr(api.tg, "send", sent.append)
    api._check_fills()

    assert sent == []
    assert store.read_fills(db_path=p)[0]["reconciliation"] == 1
    assert store.get_kv("alert_last_fill_rowid", "0", p) == "1"


def test_control_modes(db):
    assert store.is_halted(db) is False and store.get_mode(db) == "run"
    store.set_mode("kill", db)
    assert store.is_halted(db) is True and store.get_mode(db) == "kill"
    store.set_mode("halt", db)
    assert store.is_halted(db) is True
    store.set_mode("run", db)
    assert store.is_halted(db) is False


def test_flatten_request(db):
    seq, target = store.get_flatten(db)
    assert seq == 0
    n = store.request_flatten("all", db)
    assert n == 1 and store.get_flatten(db) == (1, "all")
    store.request_flatten("BTCUSDT-PERP", db)
    assert store.get_flatten(db) == (2, "BTCUSDT-PERP")


def test_api_endpoints_and_kill(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from kronos_mt5.companion import api

    p = str(tmp_path / "api.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    monkeypatch.setattr(api.settings, "api_token", "")
    store.write_snapshot({"equity": 10500.0, "cash": 10000.0, "n_open": 1, "positions": []}, p)
    store.append_equity(10500.0, 10000.0, 500.0, 1, p)
    store.heartbeat(p)

    with TestClient(api.app) as client:
        st = client.get("/api/state").json()
        assert st["equity"] == 10500.0 and st["alive"] is True and st["halted"] is False
        assert len(client.get("/api/equity").json()) == 1
        assert client.get("/api/forward-test").json()["minimum_target"] == 30
        assert client.get("/api/shadows").json()["models"] == {}

        assert client.post("/kill").json()["mode"] == "kill"
        assert store.is_halted(p) is True
        assert client.get("/api/state").json()["halted"] is True
        assert client.post("/resume").json()["mode"] == "run"


def test_api_token_guard(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from kronos_mt5.companion import api

    p = str(tmp_path / "api.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    monkeypatch.setattr(api.settings, "api_token", "secret")

    with TestClient(api.app) as client:
        assert client.post("/kill").status_code == 401  # no bearer
        assert client.get("/api/state").status_code == 401  # API also guarded
        assert client.post("/kill", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/kill", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200 and ok.json()["mode"] == "kill"
        assert (
            client.get("/api/state", headers={"Authorization": "Bearer secret"}).status_code == 200
        )


def test_mark_stale_alert_and_api_halt(tmp_path, monkeypatch):
    from kronos_mt5.companion import api

    p = str(tmp_path / "marks.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    sent = []
    monkeypatch.setattr(api.tg, "send", sent.append)
    store.write_snapshot(
        {
            "equity": 5000.0,
            "cash": 5000.0,
            "unrealized": 0.0,
            "n_open": 0,
            "mark_data_stale": True,
            "stale_mark_symbols": ["ETHUSDT-PERP"],
        },
        p,
    )
    # The trend bot's watchdog is the sole incident writer; the companion alerts off it.
    store.start_incident("MARK_STALE", "ETHUSDT-PERP", p)

    api._check_mark_data()
    api._check_mark_data()  # alert is edge-triggered, not repeated every poll
    assert len(sent) == 1 and "ETHUSDT-PERP" in sent[0]
    state = json.loads(api.api_state().body)
    assert state["halted"] is True and state["mark_data_stale"] is True

    snap = store.read_snapshot(p)
    snap["mark_data_stale"] = False
    snap["stale_mark_symbols"] = []
    store.write_snapshot(snap, p)
    store.resolve_incident("MARK_STALE", p)
    api._check_mark_data()
    assert len(sent) == 2 and "RECOVERED" in sent[-1]


def test_mark_stale_alert_escalates_once(tmp_path, monkeypatch):
    from kronos_mt5.companion import api

    p = str(tmp_path / "marks-escalation.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    monkeypatch.setattr(api.settings, "api_mark_stale_escalate_secs", 60)
    sent = []
    monkeypatch.setattr(api.tg, "send", sent.append)
    store.write_snapshot(
        {
            "mark_data_stale": True,
            "stale_mark_symbols": ["BTCUSDT-PERP"],
        },
        p,
    )
    store.start_incident("MARK_STALE", "BTCUSDT-PERP", p)

    api._check_mark_data()
    store.set_kv(
        "alert_mark_stale_started_ts",
        "2000-01-01T00:00:00+00:00",
        p,
    )
    api._check_mark_data()
    api._check_mark_data()

    assert len(sent) == 2
    assert "STILL STALE" in sent[-1]
    assert store.get_kv("alert_mark_stale_escalated", "0", p) == "1"


def test_telegram_commands(tmp_path, monkeypatch):
    from kronos_mt5.companion import api

    p = str(tmp_path / "cmd.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    store.write_snapshot(
        {
            "equity": 10300.0,
            "cash": 10000.0,
            "unrealized": 300.0,
            "n_open": 1,
            "positions": [
                {"symbol": "BTCUSDT-PERP", "qty": -0.003, "entry": 65000.0, "unrealized": 300.0}
            ],
            "orders": [
                {
                    "symbol": "BTCUSDT-PERP",
                    "type": "STOP_MARKET",
                    "side": "BUY",
                    "qty": 0.003,
                    "trigger": 78000.0,
                },
                {
                    "symbol": "BTCUSDT-PERP",
                    "type": "TRAILING_STOP_MARKET",
                    "side": "BUY",
                    "qty": 0.003,
                    "trigger": 52000.0,
                    "trailing_offset_bps": 500.0,
                },
            ],
        },
        p,
    )
    store.append_equity(10000.0, 10000.0, 0.0, 0, p)
    store.append_equity(10150.0, 10000.0, 150.0, 1, p)
    store.append_equity(10300.0, 10000.0, 300.0, 1, p)
    store.heartbeat(p)

    text, kb = api._handle("/help")
    assert "/status" in text and kb is not None  # quick-action keyboard
    text, _ = api._handle("/status@mybot")  # handles /cmd@botname
    assert "Equity" in text and "LIVE" in text
    assert "BTC" in api._handle("/positions")[0]  # shortened coin name
    assert "Total" in api._handle("/pnl")[0]
    stops = api._handle("/stops")[0]
    assert "78000" in stops and "Volatility trails" in stops and "5.0%" in stops
    assert "STOP" in api._handle("/orders")[0]
    assert "Equity" in api._handle("/chart")[0] and "Today" in api._handle("/equity")[0]

    # /kill only CONFIRMS (returns a keyboard), doesn't kill yet
    ktext, kkb = api._handle("/kill")
    assert "Confirm" in ktext and kkb is not None
    assert store.get_mode(p) == "run"

    api._handle("/halt")
    assert store.get_mode(p) == "halt"
    api._handle("/resume")
    assert store.get_mode(p) == "run"

    s0 = store.get_flatten(p)[0]
    api._handle("/flatten")
    assert store.get_flatten(p) == (s0 + 1, "all")
    api._handle("/close BTC")
    seq, target = store.get_flatten(p)
    assert seq == s0 + 2 and target == "BTCUSDT-PERP"
    assert "unknown" in api._handle("/wat")[0]


def test_positions_card_png_renders():
    pytest.importorskip("PIL")

    from kronos_mt5.companion.cards import render_positions_card

    png = render_positions_card(
        {
            "positions": [
                {"symbol": "BTCUSDT-PERP", "qty": -0.0035, "unrealized": 1.0},
                {"symbol": "ETHUSDT-PERP", "qty": -0.091, "unrealized": -2.0},
            ],
        }
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 1000


def test_positions_command_sends_photo(tmp_path, monkeypatch):
    pytest.importorskip("PIL")

    from kronos_mt5.companion import api

    p = str(tmp_path / "cmd-photo.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    monkeypatch.setattr(api, "_TG_BASE", "https://api.telegram.org/bottest")
    store.write_snapshot(
        {
            "equity": 4955.0,
            "cash": 4949.0,
            "unrealized": 6.0,
            "n_open": 1,
            "positions": [
                {"symbol": "BTCUSDT-PERP", "qty": -0.0035, "entry": 65000.0, "unrealized": 1.0},
            ],
        },
        p,
    )

    class Response:
        def json(self):
            return {"ok": True}

    class Client:
        def __init__(self):
            self.calls = []

        async def post(self, url, data=None, files=None):
            self.calls.append({"url": url, "data": data, "files": files})
            return Response()

    client = Client()
    asyncio.run(api._send_command(client, "owner-chat", "/positions"))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"].endswith("/sendPhoto")
    assert call["data"]["chat_id"] == "owner-chat"
    assert call["data"]["parse_mode"] == "HTML"
    assert call["files"]["photo"][0] == "positions.png"
    assert call["files"]["photo"][1].startswith(b"\x89PNG\r\n\x1a\n")
