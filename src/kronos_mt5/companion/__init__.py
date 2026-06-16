"""Companion observability + control layer for the live trading bot.

Decoupled from the bot via a small SQLite DB: the bot writes state/fills/equity
and polls a control flag; the FastAPI app reads the DB for the dashboard and
sends Telegram alerts. Neither process can crash the other.
"""
