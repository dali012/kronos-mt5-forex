"""Telegram notifier (used by the companion API, not the trading loop).

No-ops cleanly if no token/chat is configured, so the dashboard still works
without Telegram set up.
"""
from __future__ import annotations

import urllib.parse
import urllib.request


class Telegram:
    def __init__(self, token: str | None, chat_id: str | None) -> None:
        self.token = (token or "").strip()
        self.chat_id = (chat_id or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        """Best-effort send; never raises (alerts must not crash the API)."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        ).encode()
        try:
            with urllib.request.urlopen(url, data=data, timeout=10) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False
