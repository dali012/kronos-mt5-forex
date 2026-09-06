"""Fixed-parameter baseline and offline reproduction CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kronos_mt5.marketdata.manifest import save_manifest
from kronos_mt5.marketdata.pipeline import safe_output

from .config import BaselineConfig
from .metrics import summarize
from .report import clean, provenance, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser(
        "config", help="write fixed deployed strategy/replay configuration"
    )
    config.add_argument("--output", type=Path, default=Path("research/baseline-config.json"))
    config.add_argument("--symbols", nargs="+", default=None)
    config.add_argument("--omit-funding", action="store_true")
    baseline = commands.add_parser("run")
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--config", type=Path, required=True)
    baseline.add_argument("--exchange-info", type=Path, required=True)
    baseline.add_argument("--output", type=Path, default=Path("research/baseline"))
    baseline.add_argument("--include-holdout", action="store_true")
    reproduce = commands.add_parser(
        "reproduce", help="verify provenance and rerun a report offline"
    )
    reproduce.add_argument("--report", type=Path, required=True)
    reproduce.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="relocated copy of exact pinned dataset manifest",
    )
    reproduce.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser(
        "smoke", help="small deterministic synthetic engine/accounting smoke test"
    )
    smoke.add_argument("--output", type=Path, default=Path("research/synthetic-smoke"))
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            kw = {"funding_mode": "omit" if args.omit_funding else "historical"}
            if args.symbols:
                kw["symbols"] = tuple(args.symbols)
            c = BaselineConfig(**kw)
            safe_output(args.output.parent)
            save_manifest(args.output, c.payload())
            print(json.dumps({"configuration": str(args.output), "sha256": c.fingerprint()}))
        elif args.command == "run":
            result = run(
                args.manifest,
                BaselineConfig.read(args.config),
                json.loads(args.exchange_info.read_text()),
                args.output,
                include_holdout=args.include_holdout,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "report": str(args.output / "report.json"),
                        "development": result["development"],
                        "holdout": result["holdout"],
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "reproduce":
            from kronos_mt5.marketdata.pipeline import digest

            from .config import STRATEGY

            original = json.loads(args.report.read_text())
            source = provenance()
            for key in ("source_sha256", "python", "nautilus_trader", "pandas", "numpy", "pyarrow"):
                if original[key] != source[key]:
                    raise ValueError(
                        f"reproduction {key} differs from original; use original commit/environment"
                    )
            if digest(args.manifest) != original["validation"]["manifest_sha256"]:
                raise ValueError("reproduction dataset manifest checksum differs")
            conf = original["configuration"]
            if conf["strategy"] != json.loads(json.dumps(STRATEGY)):
                raise ValueError("reproduction strategy differs")
            kw = dict(conf["replay"])
            kw["symbols"] = tuple(kw["symbols"])
            result = run(
                args.manifest,
                BaselineConfig(**kw),
                original["exchange_filters"],
                args.output,
                include_holdout=original["holdout"]["status"] == "EVALUATED_FIXED_BASELINE_ONLY",
            )
            for key in ("development", "walk_forward", "holdout"):
                if result[key] != original[key]:
                    raise ValueError(f"reproduction failed: {key} results differ")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "identical_metrics": True,
                        "report": str(args.output / "report.json"),
                    }
                )
            )
        else:
            from .synthetic import smoke as run_smoke

            result = summarize(run_smoke())
            repeated = summarize(run_smoke())
            if result != repeated or result["fills"] == 0:
                raise ValueError("synthetic smoke failed determinism or produced no fills")
            safe_output(args.output)
            save_manifest(args.output / "smoke.json", clean(result))
            print(json.dumps({"ok": True, "deterministic": True, **clean(result)}, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
