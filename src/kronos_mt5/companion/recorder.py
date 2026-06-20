"""Companion recorder — the bot-side bridge to the SQLite store.

On timers it: writes an equity/positions/orders snapshot + equity point, records
new fills (tagged STOP/TP/TREND), stamps an "alive" heartbeat, and syncs the
remote control state (mode + one-shot flatten) from the store into the shared
RiskState the strategies read. Only local SQLite writes — never blocks trading.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nautilus_trader.model.events import OrderFilled
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
    corr_window: int = 90
    corr_threshold: float = 0.65
    corr_min_scalar: float = 0.50


class Companion(Strategy):
    def __init__(self, config: CompanionConfig) -> None:
        super().__init__(config)
        self._venue = Venue(config.venue)
        self._iids = [InstrumentId.from_str(i) for i in config.instrument_ids]
        self.risk_state = None  # injected; synced from the store control flags

    def on_start(self) -> None:
        store.init_db(self.config.db_path)
        store.record_event("BOT_START", db_path=self.config.db_path)
        store.heartbeat(self.config.db_path)
        self._snapshot(None)  # establish the immutable accounting baseline promptly
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
            if acct is None:
                self.log.warning("companion snapshot: no account yet")
                return
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
                        "opened_ts": (
                            datetime.fromtimestamp(
                                open_pos[0].ts_opened / 1e9, tz=timezone.utc
                            ).isoformat()
                            if open_pos
                            else None
                        ),
                        "mark": (
                            self.risk_state.mark_prices.get(str(iid))
                            if self.risk_state is not None
                            else None
                        ),
                        "unrealized": u_val,
                    }
                )

            orders = self._open_orders()
            self._record_new_fills()
            self._record_closed_positions()

            direction = self._basket_direction(positions)
            correlation = self._correlation_metrics()
            basket = self._basket_metrics(positions, orders, direction, correlation)
            store.update_basket_cycle(
                direction,
                [p["symbol"] for p in positions],
                cash + upnl_total,
                basket["hard_stop_risk"],
                self.config.db_path,
                started_ts=min(
                    (p["opened_ts"] for p in positions if p.get("opened_ts")),
                    default=None,
                ),
            )

            mark_stale = bool(self.risk_state and self.risk_state.market_data_stale)
            control_mode = self.risk_state.mode if self.risk_state else "run"
            equity = cash + upnl_total
            performance = store.performance_summary(
                equity,
                cash,
                upnl_total,
                self.config.db_path,
            )
            store.write_snapshot(
                {
                    "equity": equity, "cash": cash, "unrealized": upnl_total,
                    "n_open": len(positions),
                    "performance": performance,
                    "mode": "halt" if mark_stale and control_mode == "run" else control_mode,
                    "control_mode": control_mode,
                    "mark_data_stale": mark_stale,
                    "stale_mark_symbols": (
                        list(self.risk_state.stale_mark_symbols) if self.risk_state else []
                    ),
                    "mark_age_secs": (
                        self.risk_state.mark_age_secs if self.risk_state else {}
                    ),
                    "basket": basket,
                    "positions": positions, "orders": orders,
                },
                self.config.db_path,
            )
            store.append_equity(
                equity,
                cash,
                upnl_total,
                len(positions),
                self.config.db_path,
                realized=performance["realized_pnl"],
                commissions=performance["commissions"],
                funding=performance["funding_payments"],
                slippage=performance["slippage"],
                total_pnl=performance["total_pnl"],
                basket_direction=direction,
                average_correlation=correlation["average_abs_correlation"],
                effective_bets=correlation["effective_bets"],
                market_regime=(
                    f"{direction}_{correlation['volatility_regime']}"
                    if direction != "FLAT"
                    else None
                ),
            )
            store.heartbeat(self.config.db_path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"companion snapshot failed: {exc!r}")

    def on_stop(self) -> None:
        try:
            store.record_event("BOT_STOP", db_path=self.config.db_path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"companion stop event failed: {exc!r}")

    def _correlation_metrics(self) -> dict:
        if self.risk_state is None or not hasattr(self.risk_state, "correlation_metrics"):
            return {
                "series_count": 0,
                "average_abs_correlation": None,
                "effective_bets": None,
                "risk_scalar": 1.0,
                "average_annualized_volatility": None,
                "volatility_regime": "UNKNOWN",
            }
        return self.risk_state.correlation_metrics(
            self.config.corr_window,
            self.config.corr_threshold,
            self.config.corr_min_scalar,
        )

    @staticmethod
    def _basket_direction(positions: list[dict]) -> str:
        sides = {1 if p["qty"] > 0 else -1 for p in positions if p["qty"] != 0}
        if not sides:
            return "FLAT"
        if sides == {1}:
            return "LONG"
        if sides == {-1}:
            return "SHORT"
        return "MIXED"

    @staticmethod
    def _basket_metrics(
        positions: list[dict],
        orders: list[dict],
        direction: str,
        correlation: dict,
    ) -> dict:
        gross_notional = 0.0
        net_notional = 0.0
        hard_stop_risk = 0.0
        stops = {
            order["symbol"]: order["trigger"]
            for order in orders
            if order["type"] == "STOP_MARKET" and order["trigger"] is not None
        }
        for position in positions:
            reference = position.get("mark") or position.get("entry")
            if reference is not None:
                signed = float(position["qty"]) * float(reference)
                gross_notional += abs(signed)
                net_notional += signed
            stop = stops.get(position["symbol"])
            entry = position.get("entry")
            if stop is not None and entry is not None:
                hard_stop_risk += abs(float(stop) - float(entry)) * abs(float(position["qty"]))
        return {
            "direction": direction,
            "symbol_positions": len(positions),
            "gross_notional": gross_notional,
            "net_notional": net_notional,
            "unrealized_pnl": sum(float(p.get("unrealized") or 0.0) for p in positions),
            "hard_stop_risk": hard_stop_risk or None,
            **correlation,
            "interpretation": "one correlated crypto basket, not independent symbol wins",
        }

    def _open_orders(self) -> list[dict]:
        out = []
        for o in self.cache.orders_open():
            trig = (
                getattr(o, "trigger_price", None)
                or getattr(o, "activation_price", None)
                or getattr(o, "price", None)
            )
            trailing_offset = getattr(o, "trailing_offset", None)
            out.append({
                "symbol": o.instrument_id.symbol.value,
                "type": o.order_type.name,
                "side": o.side.name,
                "qty": float(o.quantity),
                "trigger": float(trig) if trig is not None else None,
                "trailing_offset_bps": (
                    float(trailing_offset) if trailing_offset is not None else None
                ),
            })
        return out

    def _record_new_fills(self) -> None:
        for order in self.cache.orders():
            ot = order.order_type.name
            if ot == "STOP_MARKET":
                kind = "STOP"
            elif ot == "TRAILING_STOP_MARKET":
                kind = "TRAIL"
            elif getattr(order, "is_reduce_only", False):
                kind = "TP" if ot == "LIMIT" else "STOP"
            else:
                kind = "TREND"
            reference = self._reference_price(order)
            for index, event in enumerate(order.events):
                if not isinstance(event, OrderFilled):
                    continue
                qty = float(event.last_qty)
                price = float(event.last_px)
                trade_id = event.trade_id.value if event.trade_id is not None else ""
                fill_id = (
                    f"{order.instrument_id.symbol.value}:{trade_id}"
                    if trade_id
                    else f"{order.client_order_id.value}:{event.ts_event}:{index}"
                )
                slippage = None
                if reference is not None:
                    slippage = (
                        (price - reference) * qty
                        if event.is_buy
                        else (reference - price) * qty
                    )
                commission = event.commission.as_double() if event.commission is not None else None
                ts = datetime.fromtimestamp(event.ts_event / 1e9, tz=timezone.utc).isoformat()
                store.record_fill(
                    fill_id=fill_id,
                    symbol=order.instrument_id.symbol.value,
                    side=event.order_side.name,
                    qty=qty,
                    price=price,
                    kind=kind,
                    ts=ts,
                    trade_id=trade_id,
                    commission=commission,
                    reference_price=reference,
                    slippage=slippage,
                    reconciliation=bool(event.reconciliation),
                    db_path=self.config.db_path,
                )

    def _record_closed_positions(self) -> None:
        for position in self.cache.positions_closed(venue=self._venue):
            opening = self.cache.order(position.opening_order_id)
            closing = (
                self.cache.order(position.closing_order_id)
                if position.closing_order_id is not None
                else None
            )
            risk_pct = self._tag_float(opening, "RISK_PCT=")
            peak_qty = float(position.peak_qty)
            entry = float(position.avg_px_open)
            initial_risk = peak_qty * entry * risk_pct if risk_pct else None
            commissions = sum(money.as_double() for money in position.commissions())
            realized = position.realized_pnl
            net_pnl = realized.as_double() if realized is not None else 0.0
            reason = self._exit_reason(closing)
            opened_ts = datetime.fromtimestamp(position.ts_opened / 1e9, tz=timezone.utc).isoformat()
            closed_ts = datetime.fromtimestamp(position.ts_closed / 1e9, tz=timezone.utc).isoformat()
            store.record_closed_position(
                position_key=f"{position.id.value}:{position.ts_opened}",
                opened_ts=opened_ts,
                closed_ts=closed_ts,
                symbol=position.instrument_id.symbol.value,
                # Closed Nautilus positions have side=FLAT; entry preserves the
                # original BUY/SELL direction for lifecycle attribution.
                side="LONG" if position.entry.name == "BUY" else "SHORT",
                peak_qty=peak_qty,
                entry_price=entry,
                exit_price=float(position.avg_px_close),
                net_pnl=net_pnl,
                commissions=commissions,
                initial_risk=initial_risk,
                exit_reason=reason,
                db_path=self.config.db_path,
            )

    @staticmethod
    def _tag_float(order, prefix: str) -> float | None:  # noqa: ANN001
        if order is None:
            return None
        for tag in order.tags or []:
            if tag.startswith(prefix):
                try:
                    return float(tag.removeprefix(prefix))
                except ValueError:
                    return None
        return None

    @staticmethod
    def _exit_reason(order) -> str:  # noqa: ANN001
        if order is None:
            return "OTHER"
        kind = order.order_type.name
        if kind == "STOP_MARKET":
            return "HARD_STOP"
        if kind == "TRAILING_STOP_MARKET":
            return "TRAILING_STOP"
        if kind == "LIMIT" and getattr(order, "is_reduce_only", False):
            return "TAKE_PROFIT"
        return "TREND_OR_MANUAL"

    @staticmethod
    def _reference_price(order) -> float | None:  # noqa: ANN001
        for tag in order.tags or []:
            if tag.startswith("REF_PX="):
                try:
                    return float(tag.removeprefix("REF_PX="))
                except ValueError:
                    pass
        for name in ("trigger_price", "price", "activation_price"):
            value = getattr(order, name, None)
            if value is not None:
                return float(value)
        return None
