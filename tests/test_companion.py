"""Tests for the companion store + API (no running bot required)."""
from __future__ import annotations

import pytest

from kronos_mt5.companion import store


@pytest.fixture()
def db(tmp_path):
    p = str(tmp_path / "companion.db")
    store.init_db(p)
    return p


def test_snapshot_roundtrip(db):
    store.write_snapshot({"equity": 10100.0, "cash": 10000.0, "positions": [{"symbol": "BTCUSDT"}]}, db)
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


def test_control_flag(db):
    assert store.is_halted(db) is False
    store.set_halt(True, db)
    assert store.is_halted(db) is True
    store.set_halt(False, db)
    assert store.is_halted(db) is False


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

        assert client.post("/kill").json()["halted"] is True
        assert store.is_halted(p) is True
        assert client.get("/api/state").json()["halted"] is True
        assert client.post("/resume").json()["halted"] is False


def test_api_token_guard(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from kronos_mt5.companion import api

    p = str(tmp_path / "api.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    monkeypatch.setattr(api.settings, "api_token", "secret")

    with TestClient(api.app) as client:
        assert client.post("/kill").status_code == 401                       # no bearer
        assert client.get("/api/state").status_code == 401                   # API also guarded
        assert client.post("/kill", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.post("/kill", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200 and ok.json()["halted"] is True
        assert client.get("/api/state", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_telegram_command_replies(tmp_path, monkeypatch):
    from kronos_mt5.companion import api

    p = str(tmp_path / "cmd.db")
    store.init_db(p)
    monkeypatch.setattr(api, "DB", p)
    store.write_snapshot({"equity": 10300.0, "cash": 10000.0, "unrealized": 300.0, "n_open": 1,
                          "positions": [{"symbol": "BTCUSDT", "qty": -0.003, "unrealized": 300.0}]}, p)
    store.append_equity(10000.0, 10000.0, 0.0, 0, p)
    store.heartbeat(p)

    assert "/status" in api._command_reply("/help")
    status = api._command_reply("/status@mybot")                    # handles /cmd@botname
    assert "Equity" in status and "LIVE" in status
    assert "BTC" in api._command_reply("/positions")                # shortened coin name
    assert "Total" in api._command_reply("/pnl")
    api._command_reply("/kill")
    assert store.is_halted(p) is True
    api._command_reply("/resume")
    assert store.is_halted(p) is False
    assert "unknown" in api._command_reply("/wat")
