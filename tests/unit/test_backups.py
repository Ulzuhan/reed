from __future__ import annotations

import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest

from reed.backups import MANIFEST_NAME, create_backup, restore_backup, verify_backup


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_backup_round_trip_verifies_every_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "uploads").mkdir(parents=True)
    (data / "reed.db").write_bytes(b"registry")
    (data / "uploads" / "policy.md").write_text("policy", encoding="utf-8")
    archive = tmp_path / "backup.tar.gz"

    created = create_backup(data, archive)
    verified = verify_backup(archive)
    restored = tmp_path / "restored"
    restore_backup(archive, restored)

    assert verified == created
    assert (restored / "reed.db").read_bytes() == b"registry"
    assert (restored / "uploads" / "policy.md").read_text(encoding="utf-8") == "policy"


def test_backup_refuses_overwrites_and_nonempty_restore_targets(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "file").write_text("value", encoding="utf-8")
    archive = tmp_path / "backup.tar.gz"
    create_backup(data, archive)

    with pytest.raises(FileExistsError, match="overwrite"):
        create_backup(data, archive)
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        restore_backup(archive, target)


def test_backup_rejects_missing_sources_nested_destinations_and_symlinks(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        create_backup(missing, tmp_path / "missing.tar.gz")

    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError, match="outside"):
        create_backup(data, data / "nested.tar.gz")

    (data / "target").write_text("value", encoding="utf-8")
    (data / "link").symlink_to("target")
    with pytest.raises(ValueError, match="symlink"):
        create_backup(data, tmp_path / "linked.tar.gz")


def test_restore_keeps_recorded_permissions_but_drops_setuid(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    registry = data / "reed.db"
    registry.write_bytes(b"registry")
    registry.chmod(0o600)
    tool = data / "tool"
    tool.write_text("payload", encoding="utf-8")
    tool.chmod(0o4755)
    if not tool.stat().st_mode & 0o4000:
        pytest.skip("this filesystem does not keep the setuid bit")
    archive = tmp_path / "backup.tar.gz"
    create_backup(data, archive)

    restored = tmp_path / "restored"
    restore_backup(archive, restored)

    # The umask would otherwise widen a private registry to 0644.
    assert (restored / "reed.db").stat().st_mode & 0o7777 == 0o600
    assert (restored / "tool").stat().st_mode & 0o7777 == 0o755


def test_verify_names_a_corrupt_archive_instead_of_raising_a_lookup_error(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "reed.db").write_bytes(b"registry" * 200)
    archive = tmp_path / "backup.tar.gz"
    manifest = create_backup(data, archive)

    # An interrupted copy: the manifest never arrives, and the raw lookup error
    # for it used to be what an operator saw mid-recovery.
    intact = archive.read_bytes()
    archive.write_bytes(intact[: len(intact) // 2])
    with pytest.raises(ValueError, match="not a readable Reed backup"):
        verify_backup(archive)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_backup(tmp_path / "absent.tar.gz")

    # Content that still decodes must name the file whose digest moved. Written
    # rather than bit-flipped: where a flipped byte lands in the compressed
    # stream depends on the platform's zlib.
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as output:
        _add(output, "data/reed.db", b"not what the manifest recorded")
        _add(output, MANIFEST_NAME, (json.dumps(asdict(manifest), sort_keys=True) + "\n").encode())

    with pytest.raises(ValueError, match=r"Checksum mismatch for reed\.db"):
        verify_backup(tampered)


def test_backup_rejects_path_traversal_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        payload = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe backup path"):
        verify_backup(archive)


def test_backup_skips_the_model_cache_but_still_refuses_stray_symlinks(tmp_path: Path) -> None:
    # The container puts FastEmbed's cache inside REED_DATA_DIR because a
    # read-only root filesystem leaves nowhere else persistent, and that cache
    # is a HuggingFace tree whose snapshots are symlinks into blobs.
    data = tmp_path / "data"
    (data / "uploads").mkdir(parents=True)
    (data / "reed.db").write_bytes(b"registry")
    (data / "uploads" / "policy.md").write_text("policy", encoding="utf-8")
    cache = data / ".fastembed" / "models--Qdrant--bm25" / "snapshots" / "abc"
    cache.mkdir(parents=True)
    (data / ".fastembed" / "blob").write_text("weights", encoding="utf-8")
    (cache / "arabic.txt").symlink_to(data / ".fastembed" / "blob")

    manifest = create_backup(data, tmp_path / "backup.tar.gz")

    assert sorted(manifest.files) == ["reed.db", "uploads/policy.md"]

    # A symlink anywhere else is still refused: skipping the cache must not
    # weaken the archive against path escapes.
    (data / "uploads" / "escape.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(ValueError, match="symlink"):
        create_backup(data, tmp_path / "second.tar.gz")
