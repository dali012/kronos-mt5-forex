"""Provenance checks for deterministic deployed-strategy research replays."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kronos_mt5.baseline.config import (
    DEPLOYED_COMMIT,
    DEPLOYED_RISK_SHA256,
    DEPLOYED_STRATEGY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEPLOYED_STRATEGY_PATH = Path(
    "src/kronos_mt5/baseline/deployed_trend_dc8a74c.py"
)
DEPLOYED_RISK_PATH = Path("src/kronos_mt5/baseline/deployed_risk_dc8a74c.py")

RESEARCH_ADAPTER_PATHS = (
    Path("src/kronos_mt5/baseline/__main__.py"),
    Path("src/kronos_mt5/baseline/config.py"),
    Path("src/kronos_mt5/baseline/engine.py"),
    Path("src/kronos_mt5/baseline/instruments.py"),
    Path("src/kronos_mt5/baseline/metrics.py"),
    Path("src/kronos_mt5/baseline/provenance.py"),
    Path("src/kronos_mt5/baseline/report.py"),
    Path("src/kronos_mt5/walk_forward.py"),
)

RELEVANT_SOURCE_DIRECTORIES = (
    Path("src/kronos_mt5/baseline"),
    Path("src/kronos_mt5/experiments"),
    Path("src/kronos_mt5/marketdata"),
)
RELEVANT_SOURCE_FILES = (
    Path("backtest/walk_forward.py"),
    Path("src/kronos_mt5/walk_forward.py"),
)


class ProvenanceError(RuntimeError):
    """Raised when recorded replay provenance cannot be verified."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without normalising its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using stable canonical encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_commit(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Return the commit checked out at ``repository_root``."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def relevant_source_paths(repository_root: Path = REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Return tracked Python sources that can affect a baseline replay."""

    paths: set[Path] = set(RELEVANT_SOURCE_FILES)
    for directory in RELEVANT_SOURCE_DIRECTORIES:
        absolute_directory = repository_root / directory
        paths.update(path.relative_to(repository_root) for path in absolute_directory.glob("*.py"))
    return tuple(sorted(paths))


def assert_relevant_sources_clean(
    repository_root: Path = REPOSITORY_ROOT,
    paths: tuple[Path, ...] | None = None,
) -> None:
    """Reject tracked replay source changes relative to ``HEAD``.

    Untracked and ignored datasets and reports are intentionally outside this
    check. Every relevant path must already be tracked, so a newly added replay
    module cannot be used before it is committed.
    """

    selected = paths or relevant_source_paths(repository_root)
    untracked: list[str] = []
    for path in selected:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path.as_posix()],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            untracked.append(path.as_posix())
    if untracked:
        raise ProvenanceError(
            "replay sources must be committed before a baseline run: "
            + ", ".join(untracked)
        )

    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *(path.as_posix() for path in selected)],
        cwd=repository_root,
        check=False,
    )
    if completed.returncode == 0:
        return
    if completed.returncode > 1:
        raise ProvenanceError("unable to compare replay sources with HEAD")

    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *(path.as_posix() for path in selected)],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    raise ProvenanceError(
        "baseline runs require clean tracked replay sources; modified: "
        + ", ".join(changed)
    )


def verify_deployed_snapshots(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Verify the frozen deployed strategy and its sizing/risk dependency."""

    strategy_actual = sha256_file(repository_root / DEPLOYED_STRATEGY_PATH)
    risk_actual = sha256_file(repository_root / DEPLOYED_RISK_PATH)
    strategy_verified = strategy_actual == DEPLOYED_STRATEGY_SHA256
    risk_verified = risk_actual == DEPLOYED_RISK_SHA256
    if not strategy_verified:
        raise ProvenanceError(
            "deployed strategy snapshot hash mismatch: "
            f"expected {DEPLOYED_STRATEGY_SHA256}, got {strategy_actual}"
        )
    if not risk_verified:
        raise ProvenanceError(
            "deployed risk snapshot hash mismatch: "
            f"expected {DEPLOYED_RISK_SHA256}, got {risk_actual}"
        )
    return {
        "deployed_bot_commit": DEPLOYED_COMMIT,
        "deployed_strategy_blob_sha256": DEPLOYED_STRATEGY_SHA256,
        "deployed_strategy_snapshot_sha256": strategy_actual,
        "deployed_risk_blob_sha256": DEPLOYED_RISK_SHA256,
        "deployed_risk_snapshot_sha256": risk_actual,
        "deployed_snapshot_verification_passed": strategy_verified and risk_verified,
    }


def dependency_versions() -> dict[str, str]:
    """Return dependency versions that can affect replay calculations."""

    dependencies = {"python": platform.python_version()}
    for distribution in ("nautilus_trader", "numpy", "pandas", "pyarrow"):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    return dependencies


def collect_provenance(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Collect verified source and dependency provenance for a replay."""

    if require_clean:
        assert_relevant_sources_clean(repository_root)
    snapshot = verify_deployed_snapshots(repository_root)
    relevant_hashes = {
        path.as_posix(): sha256_file(repository_root / path)
        for path in relevant_source_paths(repository_root)
    }
    adapter_hashes = {
        path.as_posix(): sha256_file(repository_root / path)
        for path in RESEARCH_ADAPTER_PATHS
    }
    commit = current_commit(repository_root)
    return {
        **snapshot,
        "source_commit": commit,
        "research_implementation_commit": commit,
        "research_adapter_sha256": adapter_hashes,
        "relevant_source_sha256": relevant_hashes,
        "dependency_versions": dependency_versions(),
    }


def verify_recorded_provenance(
    recorded: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Reject a report whose recorded replay environment differs from checkout."""

    required_fields = (
        "source_commit",
        "research_implementation_commit",
        "deployed_bot_commit",
        "deployed_strategy_blob_sha256",
        "deployed_strategy_snapshot_sha256",
        "deployed_risk_blob_sha256",
        "deployed_risk_snapshot_sha256",
        "deployed_snapshot_verification_passed",
        "research_adapter_sha256",
        "relevant_source_sha256",
        "dependency_versions",
    )
    for field in required_fields:
        if recorded.get(field) != current.get(field):
            raise ProvenanceError(f"recorded {field} does not match the current checkout")
