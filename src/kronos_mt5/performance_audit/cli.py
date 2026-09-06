"""Command-line entry point for the offline performance audit.

    python -m kronos_mt5.performance_audit --input EXPORT.tar.gz --output-dir DIR
    python -m kronos_mt5.performance_audit --input companion_sanitized.db --output-dir DIR

The input is never modified. Archives are unpacked into a temporary directory
that is removed when the run finishes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from kronos_mt5.performance_audit.audit import build_run_metadata, run_analysis
from kronos_mt5.performance_audit.render import (
    render_report,
    write_daily_csv,
    write_fill_csv,
    write_summary_json,
)
from kronos_mt5.performance_audit.source import AuditInputError, resolve_source
from kronos_mt5.performance_audit.version import TOOL_VERSION

OUTPUT_FILES = ("report.md", "summary.json", "daily_equity.csv", "fill_attribution.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kronos_mt5.performance_audit",
        description=(
            "Offline, read-only performance and data-quality audit of a sanitized "
            "Kronos companion export. Never writes to the input and never contacts "
            "a broker."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="sanitized export archive (.tar.gz/.tgz/.tar) or SQLite database (.db)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for report.md, summary.json and the CSV extracts",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the console summary")
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def _console_summary(analysis: dict, run: dict, output_dir: Path) -> str:
    equity = analysis.get("equity", {})
    findings = analysis.get("findings_summary", {})
    severities = findings.get("by_severity", {})
    lines = [
        f"input          {run['input_name']} ({run['input_kind']})",
        f"sha256         {run['input_sha256']}",
        (
            f"window         {equity.get('first_day')} .. {equity.get('last_day')} "
            f"({equity.get('days_covered')} days)"
        ),
        f"equity         {equity.get('start_equity')} -> {equity.get('end_equity')}",
        (
            f"total return   {equity.get('total_return_pct')}%"
            if equity.get("total_return_pct") is not None
            else "total return   n/a"
        ),
        (
            f"max drawdown   {equity.get('drawdown_max_drawdown_pct')}%"
            if equity.get("drawdown_max_drawdown_pct") is not None
            else "max drawdown   n/a"
        ),
        (
            f"findings       {findings.get('total', 0)} "
            f"({severities.get('ERROR', 0)} ERROR / "
            f"{severities.get('WARNING', 0)} WARNING / "
            f"{severities.get('INFO', 0)} INFO)"
        ),
        f"outputs        {output_dir}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create output directory {output_dir}: {exc}", file=sys.stderr)
        return 2

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with tempfile.TemporaryDirectory(prefix="kronos-audit-") as tmp:
        try:
            source = resolve_source(args.input, Path(tmp))
            analysis = run_analysis(source)
        except AuditInputError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"error: audit failed: {exc!r}", file=sys.stderr)
            return 1

        run = build_run_metadata(source, generated_at)
        write_summary_json(output_dir / "summary.json", analysis, run)
        write_daily_csv(output_dir / "daily_equity.csv", analysis["equity"].get("daily") or [])
        write_fill_csv(
            output_dir / "fill_attribution.csv", analysis["execution"].get("groups") or []
        )
        (output_dir / "report.md").write_text(render_report(analysis, run))

    if not args.quiet:
        print(_console_summary(analysis, run, output_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
