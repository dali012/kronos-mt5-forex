"""Run the companion dashboard/API.

  PYTHONPATH=src:. .venv/bin/python -m kronos_mt5.companion.run_api

Then open http://<host>:8000/  (append ?token=... if API_TOKEN is set).
"""
from __future__ import annotations

import uvicorn

from config.settings import settings


def main() -> None:
    print(f"Companion API on http://{settings.api_host}:{settings.api_port}  (db={settings.companion_db})")
    uvicorn.run(
        "kronos_mt5.companion.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
