"""Companion recorder — the bot-side bridge to the SQLite store.

On timers it: writes an equity/positions/orders snapshot + equity point, records
new fills (tagged STOP/TP/TREND), stamps an "alive" heartbeat, and syncs the
remote control state (mode + one-shot flatten) from the store into the shared
RiskState the strategies read. Only local SQLite writes — never blocks trading.
"""
from __future__ import annotations

from datetime import timedelta

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from kronos_mt5.companion import store


class CompanionConfig(StrategyConfig, frozen=True):
    venue: str
    instrument_ids: tuple[str, ...]
    db_path: str = store.DEFAULT_DB
    snapshot_secs: int = 60
    control_secs: int = 15


class Companion(Strategy):
    def __init__(self, config: CompanionConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue)
        self._iids = [InstrumentId.from_str(i) for i in config.instrument_ids]
        self.risk_state = None  # injected; synced from the store control flags

    def on_start(self) -> None:
        store.init_db(self.config.db_path)
        store.heartbeat(self.config.db_path)
        self.clock.set_timer("companion_snapshot",
                             timedelta(seconds=self.config.snapshot_secs), callback=self._snapshot)
        self.clock.set_timer("companion_control",
                             timedelta(seconds=self.config.control_secs), callback=self._poll_control)

    # --- sync remote control (store -> shared RiskState) ----------------------
    def _poll_control(self, event) -> None:  # noqa: ANN001
        rs = self.risk_state
        if rs is None:
            return
        try:
            mode = store.get_mode(self.config.db_path)
            seq, target = store.get_flatten(self.config.db_path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"control poll failed: {exc!r}")
            return
        if rs.mode != mode:
            self.log.error(f"REMOTE CONTROL: mode={mode}")
        rs.mode = mode
        rs.flatten_seq = seq
        rs.flatten_target = target

    # --- snapshot + equity + orders + fills + alive heartbeat -----------------
    def _snapshot(self, event) -> None:  # noqa: ANN001
        try:
            acct = self.portfolio.account(self._venue)
            cash = 0.0
            if acct is not None:
                bal = acct.balance_total(USDT)
                cash = bal.as_double() if bal is not None else 0.0

            positions, upnl_total = [], 0.0
            for iid in self._iids:
                net = float(self.portfolio.net_position(iid))
                if net == 0:
                    continue
                u = self.portfolio.unrealized_pnl(iid)
                u_val = u.as_double() if u is not None else 0.0
                upnl_total += u_val
                entry = None
                open_pos = self.cache.positions_open(instrument_id=iid)
                if open_pos:
                    entry = float(open_pos[0].avg_px_open)
                positions.append(
                    {
                        "symbol": iid.symbol.value,
                        "qty": net,
                        "entry": entry,
                        "mark": (
                            self.risk_state.mark_prices.get(str(iid))
                            if self.risk_state is not None
                            else None
                        ),
                        "unrealized": u_val,
                    }
                )

            mark_stale = bool(self.risk_state and self.risk_state.market_data_stale)
            control_mode = self.risk_state.mode if self.risk_state else "run"
            store.write_snapshot(
                {
                    "equity": cash + upnl_total, "cash": cash, "unrealized": upnl_total,
                    "n_open": len(positions),
                    "mode": "halt" if mark_stale and control_mode == "run" else control_mode,
                    "control_mode": control_mode,
                    "mark_data_stale": mark_stale,
                    "stale_mark_symbols": (
                        list(self.risk_state.stale_mark_symbols) if self.risk_state else []
                    ),
                    "mark_age_secs": (
                        self.risk_state.mark_age_secs if self.risk_state else {}
                    ),
                    "positions": positions, "orders": self._open_orders(),
                },
                self.config.db_path,
            )
            store.append_equity(cash + upnl_total, cash, upnl_total, len(positions), self.config.db_path)
            self._record_new_fills()
            store.heartbeat(self.config.db_path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"companion snapshot failed: {exc!r}")

    def _open_orders(self) -> list[dict]:
        out = []
        for o in self.cache.orders_open():
            trig = getattr(o, "trigger_price", None) or getattr(o, "price", None)
            out.append({
                "symbol": o.instrument_id.symbol.value,
                "type": o.order_type.name,
                "side": o.side.name,
                "qty": float(o.quantity),
                "trigger": float(trig) if trig is not None else None,
            })
        return out

    def _record_new_fills(self) -> None:
        for order in self.cache.orders():
            try:
                filled = float(order.filled_qty)
            except Exception:  # noqa: BLE001
                filled = 0.0
            if filled <= 0 or order.avg_px is None:
                continue
            ot = order.order_type.name
            if ot == "STOP_MARKET":
                kind = "STOP"
            elif getattr(order, "is_reduce_only", False):
                kind = "TP" if ot == "LIMIT" else "STOP"
            else:
                kind = "TREND"
            store.record_fill(
                fill_id=order.client_order_id.value,
                symbol=order.instrument_id.symbol.value,
                side=order.side.name,
                qty=filled,
                price=float(order.avg_px),
                kind=kind,
                db_path=self.config.db_path,
            )
