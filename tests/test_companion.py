"""Tests for the companion store + API (no running bot required)."""

from __future__ import annotations

import asyncio

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
    store.write_snapshot({"equity": 10500.0, "cash": 10000.0, "n_open": 1, "positions": []}, p)
    store.append_equity(10500.0, 10000.0, 500.0, 1, p)
    store.heartbeat(p)

    with TestClient(api.app) as client:
        st = client.get("/api/state").json()
        assert st["equity"] == 10500.0 and st["alive"] is True and st["halted"] is False
        assert len(client.get("/api/equity").json()) == 1

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
                }
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
    assert "78000" in api._handle("/stops")[0]  # stop level shown
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
