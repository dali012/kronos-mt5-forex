"""Companion recorder — the bot-side bridge to the SQLite store.

Runs as a strategy in the trading node. On timers it: writes an equity/positions
snapshot + equity point, records any new fills, stamps an "alive" heartbeat, and
polls the shared control flag (remote /kill -> RiskState.halted -> strategies
flatten + stop entering). Only local SQLite writes — never blocks on the network.
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
        self.risk_state = None  # injected; remote kill flips this

    def on_start(self) -> None:
        store.init_db(self.config.db_path)
        store.heartbeat(self.config.db_path)
        self.clock.set_timer("companion_snapshot",
                             timedelta(seconds=self.config.snapshot_secs), callback=self._snapshot)
        self.clock.set_timer("companion_control",
                             timedelta(seconds=self.config.control_secs), callback=self._poll_control)

    # --- remote control: /kill writes halt=1 -> flip the shared RiskState -----
    def _poll_control(self, event) -> None:  # noqa: ANN001
        try:
            halted = store.is_halted(self.config.db_path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"control poll failed: {exc!r}")
            return
        if self.risk_state is not None and self.risk_state.halted != halted:
            self.risk_state.halted = halted
            self.log.error(f"REMOTE CONTROL: halt={halted} (strategies {'flattening' if halted else 'resuming'})")

    # --- snapshot + equity + fills + alive heartbeat --------------------------
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
                    {"symbol": iid.symbol.value, "qty": net, "entry": entry, "unrealized": u_val}
                )

            equity = cash + upnl_total
            n_open = len(positions)
            store.write_snapshot(
                {
                    "equity": equity, "cash": cash, "unrealized": upnl_total,
                    "n_open": n_open, "halted": bool(self.risk_state and self.risk_state.halted),
                    "positions": positions,
                },
                self.config.db_path,
            )
            store.append_equity(equity, cash, upnl_total, n_open, self.config.db_path)
            self._record_new_fills()
            store.heartbeat(self.config.db_path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"companion snapshot failed: {exc!r}")

    def _record_new_fills(self) -> None:
        # idempotent: INSERT OR IGNORE on the order id; cheap for a daily bot
        for order in self.cache.orders():
            try:
                filled = float(order.filled_qty)
            except Exception:  # noqa: BLE001
                filled = 0.0
            if filled <= 0:
                continue
            avg = order.avg_px
            if avg is None:
                continue
            store.record_fill(
                fill_id=order.client_order_id.value,
                symbol=order.instrument_id.symbol.value,
                side=order.side.name,
                qty=filled,
                price=float(avg),
                db_path=self.config.db_path,
            )
