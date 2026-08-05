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
    create_backup(data, archive)
    intact = archive.read_bytes()

    # An interrupted copy: the manifest never arrives, and the raw lookup error
    # for it used to be what an operator saw mid-recovery.
    archive.write_bytes(intact[: len(intact) // 2])
    with pytest.raises(ValueError, match="not a readable Reed backup"):
        verify_backup(archive)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_backup(tmp_path / "absent.tar.gz")

    # Altered content still reports which file failed, which is more useful.
    altered = bytearray(intact)
    altered[len(altered) // 2] ^= 0xFF
    archive.write_bytes(altered)
    with pytest.raises(ValueError, match=r"Checksum mismatch for reed\.db"):
        verify_backup(archive)


def test_backup_rejects_path_traversal_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        payload = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe backup path"):
        verify_backup(archive)
