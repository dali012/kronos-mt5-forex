"""Versioning for the audit tool and its machine-readable output.

`SCHEMA_VERSION` is the contract for `summary.json`. Bump the minor version for
additive fields, the major version for anything a consumer must adapt to.
"""

from __future__ import annotations

TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
