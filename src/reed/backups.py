"""Offline backup archives with path-safe verification and restore."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO

from reed import __version__

MANIFEST_NAME = "reed-backup-manifest.json"
BACKUP_SCHEMA = 1

# Derived caches living inside REED_DATA_DIR. A backup carries state that
# cannot be recreated — the corpus, the registry, the vectors — and a model
# cache is neither: it re-downloads. It is also a HuggingFace-style tree, whose
# snapshots are symlinks into blobs, and following those is exactly what the
# archive must not do. The container puts it here because a read-only root
# filesystem leaves nowhere else persistent.
EXCLUDED_DIRECTORIES = frozenset({".fastembed"})


@dataclass(frozen=True, slots=True)
class BackupManifest:
    schema: int
    reed_version: str
    created_at: str
    files: dict[str, str]


def create_backup(data_dir: Path, destination: Path) -> BackupManifest:
    source = data_dir.resolve()
    target = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Reed data directory does not exist: {source}")
    if target == source or source in target.parents:
        raise ValueError("The backup archive must be outside REED_DATA_DIR")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {target}")

    files: dict[str, str] = {}
    paths: list[tuple[Path, str]] = []
    for path in sorted(source.rglob("*")):
        relative_path = path.relative_to(source)
        # Skipped before the symlink check, not after: the cache is full of them.
        if relative_path.parts[0] in EXCLUDED_DIRECTORIES:
            continue
        if path.is_symlink():
            raise ValueError(f"Backup source contains an unsupported symlink: {path}")
        if not path.is_file():
            continue
        relative = relative_path.as_posix()
        files[relative] = _file_digest(path)
        paths.append((path, relative))
    manifest = BackupManifest(
        schema=BACKUP_SCHEMA,
        reed_version=__version__,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        files=files,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for path, relative in paths:
            archive.add(path, arcname=f"data/{relative}", recursive=False)
        raw = (json.dumps(asdict(manifest), sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(raw)
        info.mtime = int(datetime.now(UTC).timestamp())
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(raw))
    return manifest


def verify_backup(archive_path: Path) -> BackupManifest:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Backup archive does not exist: {archive_path}")
    try:
        return _read_and_verify(archive_path)
    except (tarfile.TarError, KeyError, EOFError, OSError, json.JSONDecodeError) as exc:
        # A truncated or altered archive surfaces as a tar/gzip/lookup failure.
        # Recovery is not the moment to decode a traceback.
        raise ValueError(
            f"{archive_path} is not a readable Reed backup ({type(exc).__name__}). "
            "It may be truncated, corrupt, or written by another tool."
        ) from exc


def _read_and_verify(archive_path: Path) -> BackupManifest:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_member(member)
        manifest_member = archive.getmember(MANIFEST_NAME)
        handle = archive.extractfile(manifest_member)
        if handle is None:
            raise ValueError("Backup manifest cannot be read")
        payload = json.load(handle)
        manifest = BackupManifest(
            schema=int(payload["schema"]),
            reed_version=str(payload["reed_version"]),
            created_at=str(payload["created_at"]),
            files={str(name): str(digest) for name, digest in payload["files"].items()},
        )
        if manifest.schema != BACKUP_SCHEMA:
            raise ValueError(f"Unsupported backup schema: {manifest.schema}")
        archived_files = {
            member.name.removeprefix("data/")
            for member in members
            if member.isfile() and member.name.startswith("data/")
        }
        if archived_files != set(manifest.files):
            raise ValueError("Backup file list does not match its manifest")
        for relative, expected in manifest.files.items():
            member_handle = archive.extractfile(f"data/{relative}")
            if member_handle is None:
                raise ValueError(f"Backup file cannot be read: {relative}")
            actual = _stream_digest(member_handle)
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {relative}")
        return manifest


def restore_backup(archive_path: Path, data_dir: Path) -> BackupManifest:
    """Extract a verified archive into an empty or absent ``data_dir``.

    Staging happens *inside* the target rather than beside it. Under the shipped
    Compose file ``REED_DATA_DIR`` is a volume mountpoint, so its parent is the
    container's root filesystem — which is read-only, is not writable by the
    unprivileged ``reed`` user, cannot have the mountpoint ``rmdir``'d out from
    under it, and lives on a different filesystem than the volume. An adjacent
    scratch directory therefore cannot work there at all. Staging inside the
    target makes every rename same-filesystem by construction.

    The cost is that publication is a handful of renames rather than one atomic
    directory swap. Nothing recoverable rides on that: the target is required to
    be empty, so a failed restore is rolled back to exactly what it found.
    """
    manifest = verify_backup(archive_path)
    target = data_dir.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty data directory: {target}")
    created_target = not target.exists()
    target.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".reed-restore-", dir=target))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for relative in manifest.files:
                destination = scratch / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                member = archive.getmember(f"data/{relative}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Backup file cannot be read: {relative}")
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                # Restore the recorded permissions instead of inheriting the
                # umask, which would widen a 0600 registry to 0644. setuid,
                # setgid and sticky bits are dropped: an archive is data.
                destination.chmod(member.mode & 0o777)
        for entry in scratch.iterdir():
            entry.rename(target / entry.name)
        scratch.rmdir()
    except Exception:
        _discard_partial_restore(target, created_target=created_target)
        raise
    return manifest


def _discard_partial_restore(target: Path, *, created_target: bool) -> None:
    """Put the target back the way the restore found it: empty, or absent.

    Restore refuses a non-empty target, so everything below it arrived with this
    attempt and nothing recoverable can be lost. ``rmtree`` cannot unlink a
    mountpoint, which is the wanted outcome when the target is a volume — and
    recreating a target this attempt did not create is best-effort, because
    failing in here would mask the failure that got us here.
    """
    shutil.rmtree(target, ignore_errors=True)
    if not created_target:
        with contextlib.suppress(OSError):
            target.mkdir(parents=True, exist_ok=True)


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe backup path: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError(f"Unsupported backup member: {member.name}")
    if not member.isfile() and not member.isdir():
        raise ValueError(f"Unsupported backup member type: {member.name}")


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return _stream_digest(handle)


def _stream_digest(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    while block := handle.read(1 << 20):
        digest.update(block)
    return digest.hexdigest()
