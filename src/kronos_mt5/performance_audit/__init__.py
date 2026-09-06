"""Offline, read-only performance forensics for a sanitized companion export.

This package NEVER writes to the input, never contacts a broker, and makes no
trading decisions. It reads a sanitized export (archive or SQLite database),
recomputes performance and accounting independently of whatever the bot recorded,
and reports what the data can and cannot support.

Read `docs/performance_audit.md` for what the audit can and cannot conclude.
"""

from kronos_mt5.performance_audit.version import SCHEMA_VERSION, TOOL_VERSION

__all__ = ["SCHEMA_VERSION", "TOOL_VERSION"]
