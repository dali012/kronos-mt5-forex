"""Live DEMO runner.

Wires a NautilusTrader TradingNode with the MT5 data + execution clients and the
KronosStrategy, pointed at the demo account from .env. This is the ONLY live
entry point in the scaffold and it refuses to start unless the account looks like
a demo.

VERIFY TradingNode / config API against the installed NautilusTrader version.
Docs: https://nautilustrader.io/docs/latest/concepts/ (Live trading)
"""
from __future__ import annotations

import sys

# from nautilus_trader.live.node import TradingNode
# from nautilus_trader.config import TradingNodeConfig, ImportableStrategyConfig

sys.path.insert(0, "../../..")  # TODO: replace with a proper package install
from config.settings import settings  # noqa: E402


def main() -> None:
    # Guardrail first — abort on anything that isn't clearly a demo account.
    settings.assert_demo()

    # TODO(claude-code):
    #  1. Build MT5DataClientConfig / MT5ExecClientConfig from settings.
    #  2. Build TradingNodeConfig registering:
    #       - data_clients   = {"MT5": <data config>}
    #       - exec_clients    = {"MT5": <exec config>}
    #       - strategy         = KronosStrategy(config, brain)
    #  3. node = TradingNode(config)
    #  4. node.add_data_client_factory("MT5", MT5LiveDataClientFactory)
    #     node.add_exec_client_factory("MT5", MT5LiveExecClientFactory)
    #  5. node.build(); node.run()
    raise NotImplementedError("Wire the TradingNode — see CLAUDE.md Stage 6.")


if __name__ == "__main__":
    main()
