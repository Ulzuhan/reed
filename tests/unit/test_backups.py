from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from reed.backups import create_backup, restore_backup, verify_backup


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


def test_backup_rejects_path_traversal_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        payload = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe backup path"):
        verify_backup(archive)
