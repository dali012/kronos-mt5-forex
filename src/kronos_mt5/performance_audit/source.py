"""Input handling: hashing, safe archive extraction and read-only SQLite access.

Everything here treats the input as immutable evidence. The database is opened
through a read-only URI with `query_only` set, archives are only ever unpacked
into a caller-supplied temporary directory, and every archive member is validated
before extraction.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Guards against a hostile or corrupt archive exhausting the machine.
MAX_MEMBERS = 20_000
MAX_TOTAL_BYTES = 2 * 1024**3  # 2 GiB uncompressed
DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")


class AuditInputError(Exception):
    """The input cannot be used. The message is intended for the CLI user."""


class UnsafeArchiveError(AuditInputError):
    """The archive contains a member we refuse to extract."""


@dataclass
class ExportSource:
    """A resolved audit input."""

    input_path: Path
    sha256: str
    database_path: Path
    kind: str  # "archive" | "database"
    root: Path | None = None  # extracted archive root, when kind == "archive"
    metadata: dict = field(default_factory=dict)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_is_safe(name: str) -> tuple[bool, str]:
    """Validate an archive member path before it is ever written to disk."""
    if not name or name in (".", "/"):
        return False, "empty member name"
    pure = Path(name)
    if pure.is_absolute() or name.startswith("/"):
        return False, f"absolute path: {name}"
    if os.path.splitdrive(name)[0]:
        return False, f"drive-qualified path: {name}"
    parts = pure.parts
    if any(part == ".." for part in parts):
        return False, f"path traversal: {name}"
    if any(part.startswith("\\") for part in parts) or "\\" in name:
        return False, f"backslash in member name: {name}"
    return True, ""


def _validate_member(member: tarfile.TarInfo) -> None:
    safe, why = _member_is_safe(member.name)
    if not safe:
        raise UnsafeArchiveError(f"refusing unsafe archive member ({why})")
    if member.issym() or member.islnk():
        raise UnsafeArchiveError(
            f"refusing link member {member.name!r} -> {member.linkname!r}: "
            "links can escape the extraction directory"
        )
    if member.isdev() or member.ischr() or member.isblk() or member.isfifo():
        raise UnsafeArchiveError(f"refusing special file member {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise UnsafeArchiveError(f"refusing unsupported member type {member.name!r}")


def safe_extract(archive_path: Path, dest: Path) -> Path:
    """Extract `archive_path` into `dest`, rejecting anything unsafe.

    Raises `UnsafeArchiveError` on traversal, absolute paths, links or special
    files, and `AuditInputError` on a malformed archive. Nothing is written until
    every member has been validated.
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            members = []
            total = 0
            for member in tar:
                _validate_member(member)
                total += max(0, member.size)
                if len(members) >= MAX_MEMBERS:
                    raise UnsafeArchiveError(
                        f"archive has more than {MAX_MEMBERS} members; refusing"
                    )
                if total > MAX_TOTAL_BYTES:
                    raise UnsafeArchiveError(
                        f"archive expands beyond {MAX_TOTAL_BYTES} bytes; refusing"
                    )
                members.append(member)
                # Belt and braces: the resolved destination must stay inside `dest`.
                target = (dest / member.name).resolve()
                if target != dest and dest not in target.parents:
                    raise UnsafeArchiveError(
                        f"member {member.name!r} would escape the extraction directory"
                    )
            if not members:
                raise AuditInputError(f"archive {archive_path.name} is empty")
            # Every member was validated above; also hand tarfile its own `data`
            # filter where available (3.12+, backported to some 3.10/3.11 patch
            # releases) so the guarantees hold when 3.14 changes the default.
            extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(dest, members=members, **extract_kwargs)
    except (tarfile.TarError, EOFError) as exc:
        raise AuditInputError(f"malformed archive {archive_path.name}: {exc}") from exc
    return dest


def find_database(root: Path) -> Path | None:
    """Locate the sanitized companion database inside an extracted export."""
    preferred = [
        root / "kronos-performance-export" / "database" / "companion_sanitized.db",
        root / "database" / "companion_sanitized.db",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    matches = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix in DB_SUFFIXES),
        key=lambda p: (0 if "sanitized" in p.name else 1, len(p.parts), str(p)),
    )
    return matches[0] if matches else None


def _read_text(path: Path, limit: int = 4096) -> str | None:
    try:
        return path.read_text(errors="replace")[:limit].strip()
    except OSError:
        return None


def collect_export_metadata(root: Path) -> dict:
    """Non-secret provenance from an extracted export: commit, host, config.

    Only reads files the export generator already sanitized. Values are echoed
    verbatim, so nothing new is exposed that the export did not already contain.
    """
    base = root / "kronos-performance-export"
    if not base.is_dir():
        base = root
    meta: dict = {}

    git_head = _read_text(base / "project" / "git_head.txt")
    if git_head:
        for line in git_head.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key in ("branch", "commit", "commit_short", "commit_date"):
                meta[f"source_{key}"] = value.strip()

    host = _read_text(base / "services" / "host_context.txt")
    if host:
        for line in host.splitlines():
            if line.startswith("hostname:"):
                meta["source_hostname"] = line.partition(":")[2].strip()
            elif line.startswith("collected_utc:"):
                meta["export_collected_utc"] = line.partition(":")[2].strip()

    env_path = base / "config" / "env.live.sanitized.txt"
    env_text = _read_text(env_path, limit=64_000)
    if env_text:
        config: dict[str, str] = {}
        for line in env_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip()
            # Redacted placeholders are the only thing the export kept for secrets;
            # skip them entirely rather than echoing "[REDACTED]" into the report.
            if "REDACTED" in value:
                continue
            config[name] = value
        if config:
            meta["config"] = dict(sorted(config.items()))

    manifest = base / "MANIFEST.md"
    if manifest.is_file():
        meta["manifest_present"] = True
    return meta


@contextmanager
def read_only_connection(db_path: Path):
    """A SQLite connection that cannot write to the input, even by accident."""
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = 1")
        yield connection
    finally:
        connection.close()


def resolve_source(input_path: Path, workdir: Path) -> ExportSource:
    """Resolve a CLI `--input` into a database path plus provenance.

    `workdir` must be a caller-owned temporary directory; archives are only ever
    unpacked there.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise AuditInputError(f"input not found: {input_path}")
    if input_path.is_dir():
        raise AuditInputError(
            f"input {input_path} is a directory; pass the export archive or the sanitized .db file"
        )

    digest = sha256_file(input_path)
    name = input_path.name.lower()

    if name.endswith(DB_SUFFIXES):
        _assert_sqlite(input_path)
        return ExportSource(
            input_path=input_path,
            sha256=digest,
            database_path=input_path,
            kind="database",
        )

    if not name.endswith(ARCHIVE_SUFFIXES):
        raise AuditInputError(
            f"unsupported input {input_path.name!r}: expected one of "
            f"{', '.join(ARCHIVE_SUFFIXES + DB_SUFFIXES)}"
        )

    root = safe_extract(input_path, workdir)
    database = find_database(root)
    if database is None:
        raise AuditInputError(
            f"no sanitized SQLite database found inside {input_path.name}; "
            f"expected database/companion_sanitized.db"
        )
    _assert_sqlite(database)
    return ExportSource(
        input_path=input_path,
        sha256=digest,
        database_path=database,
        kind="archive",
        root=root,
        metadata=collect_export_metadata(root),
    )


def _assert_sqlite(path: Path) -> None:
    try:
        with open(path, "rb") as handle:
            header = handle.read(16)
    except OSError as exc:
        raise AuditInputError(f"cannot read {path}: {exc}") from exc
    if not header.startswith(b"SQLite format 3\x00"):
        raise AuditInputError(f"{path.name} is not a SQLite database")
