"""Phase 4A experiment CLI. Noninteractive, offline, holdout-safe.

The final holdout cannot be reached from here: no subcommand exposes
``--include-holdout`` and every run asserts the report came back UNTOUCHED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kronos_mt5.baseline.provenance import ProvenanceError, collect_provenance
from kronos_mt5.marketdata.manifest import save_manifest
from kronos_mt5.marketdata.pipeline import safe_output

from .registry import BY_ID, CONTROL, EXPERIMENTS, HYPOTHESIS_COUNT, get
from .report import ranking, render
from .runner import normalized_experiment_result, run_all, run_experiment


def _write(bundle: dict, output: Path) -> None:
    output = safe_output(output)
    save_manifest(output / "experiments.json", bundle)
    save_manifest(output / "ranking.json", {"ranking": ranking(bundle["results"])})
    (output / "experiments.md").write_text(render(bundle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list registered experiments and fingerprints")

    one = commands.add_parser("run", help="run one registered experiment")
    one.add_argument("--experiment", required=True, choices=sorted(BY_ID))
    one.add_argument("--manifest", type=Path, required=True)
    one.add_argument("--exchange-info", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)

    every = commands.add_parser("run-all", help="run the control and every experiment")
    every.add_argument("--manifest", type=Path, required=True)
    every.add_argument("--exchange-info", type=Path, required=True)
    every.add_argument("--output", type=Path, required=True)
    every.add_argument("--only", nargs="+", default=None, choices=sorted(BY_ID))

    again = commands.add_parser("reproduce", help="verify provenance and rerun offline")
    again.add_argument("--report", type=Path, required=True)
    again.add_argument("--manifest", type=Path, required=True)
    again.add_argument("--exchange-info", type=Path, required=True)
    again.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print(
                json.dumps(
                    {
                        "hypotheses_tested": HYPOTHESIS_COUNT,
                        "experiments": [
                            {
                                "experiment_id": e.experiment_id,
                                "eligibility": e.eligibility,
                                "post_hoc": e.post_hoc,
                                "description": e.description,
                                "hypothesis": e.hypothesis,
                                "fingerprint": e.fingerprint(),
                            }
                            for e in EXPERIMENTS
                        ],
                    },
                    indent=2,
                )
            )
            return 0

        filters_payload = json.loads(args.exchange_info.read_text())

        if args.command == "run":
            experiment = get(args.experiment)
            control = None
            control_dev = control_windows = None
            if experiment.experiment_id != CONTROL.experiment_id:
                control = run_experiment(
                    CONTROL,
                    args.manifest,
                    filters_payload,
                    Path(args.output) / CONTROL.experiment_id,
                )
                control_dev = control["variants"]["OHLC-1.0x"]["development"]
                control_windows = control["walk_forward"]
            result = run_experiment(
                experiment,
                args.manifest,
                filters_payload,
                Path(args.output) / experiment.experiment_id,
                control_development=control_dev,
                control_windows=control_windows,
            )
            bundle = {
                "schema_version": 1,
                "provenance": collect_provenance(),
                "control_experiment_id": CONTROL.experiment_id,
                "hypotheses_tested": 0 if experiment is CONTROL else 1,
                "results": {experiment.experiment_id: result},
            }
            if control_dev is not None:
                bundle["results"] = {CONTROL.experiment_id: control, **bundle["results"]}
                bundle["hypotheses_tested"] = 1
            _write(bundle, args.output)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "experiment": experiment.experiment_id,
                        "status": result["acceptance_gate"]["status"],
                        "holdout": result["holdout"]["status"],
                        "report": str(Path(args.output) / "experiments.json"),
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "run-all":
            bundle = run_all(
                args.manifest,
                filters_payload,
                args.output,
                only=tuple(args.only) if args.only else None,
            )
            _write(bundle, args.output)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "hypotheses_tested": bundle["hypotheses_tested"],
                        "statuses": {
                            k: v["acceptance_gate"]["status"] for k, v in bundle["results"].items()
                        },
                        "holdout": {
                            k: v["holdout"]["status"] for k, v in bundle["results"].items()
                        },
                        "report": str(Path(args.output) / "experiments.json"),
                    },
                    sort_keys=True,
                )
            )
            return 0

        original = json.loads(args.report.read_text())
        current = collect_provenance()
        recorded = original["provenance"]
        for field in (
            "source_commit",
            "research_implementation_commit",
            "deployed_strategy_snapshot_sha256",
            "deployed_risk_snapshot_sha256",
            "relevant_source_sha256",
            "dependency_versions",
        ):
            if recorded.get(field) != current.get(field):
                raise ProvenanceError(f"recorded {field} does not match the current checkout")
        only = tuple(k for k in original["results"] if k != CONTROL.experiment_id)
        repeated = run_all(args.manifest, filters_payload, args.output, only=only or None)
        # Preserve the independent rerun even when comparison fails so both
        # sides of a nondeterministic result remain inspectable.
        _write(repeated, args.output)
        identical = True
        differing = []
        for key, result in original["results"].items():
            fresh = repeated["results"].get(key)
            if fresh is None:
                identical, _ = False, differing.append(f"{key}: missing")
                continue
            if normalized_experiment_result(fresh) != normalized_experiment_result(result):
                identical = False
                differing.append(f"{key}.normalized_result_including_acceptance_gate")
            if fresh["holdout"]["status"] != "UNTOUCHED":
                raise ValueError(f"{key}: holdout must remain UNTOUCHED")
        if not identical:
            raise ValueError(f"reproduction failed: {', '.join(differing)}")
        print(
            json.dumps(
                {
                    "ok": True,
                    "identical_metrics": True,
                    "experiments": sorted(original["results"]),
                    "report": str(Path(args.output) / "experiments.json"),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError, ProvenanceError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
