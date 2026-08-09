"""What reed leaves on disk stays with the account that runs it.

The container makes this moot — one process, one user, an isolated volume — but
the bare-metal flow in `docs/runbooks.md` is supported too, and there the umask
is the only thing standing between another local account and the corpus.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from reed.backups import create_backup, restore_backup
from reed.config import Settings
from reed.ingest.pipeline import register_upload
from reed.services import build_services, prepare_data_dir
from tests.conftest import capturing_reed_warnings


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def readable_by_others(path: Path) -> bool:
    return bool(mode_of(path) & 0o077)


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        profile="fake",
        data_dir=tmp_path / "data",
        collection="test_chunks",
        _env_file=None,
    )


def test_a_new_data_directory_and_its_registry_are_private(tmp_path: Path) -> None:
    services = build_services(settings_for(tmp_path))
    try:
        _ = services.registry  # creates reed.db
    finally:
        services.close()

    data = tmp_path / "data"
    assert not readable_by_others(data), f"data dir is {mode_of(data):o}"
    assert not readable_by_others(data / "reed.db"), f"registry is {mode_of(data / 'reed.db'):o}"


def test_an_existing_wide_directory_is_reported_rather_than_narrowed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An operator may have widened it on purpose; take that away silently and
    a deployment sharing the corpus with a group breaks on the next restart."""
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o755)

    with capturing_reed_warnings(caplog):
        prepare_data_dir(data)

    assert mode_of(data) == 0o755
    assert "readable beyond this account" in caplog.text
    assert "chmod 700" in caplog.text


def test_a_stored_original_is_private(tmp_path: Path) -> None:
    services = build_services(settings_for(tmp_path))
    source = tmp_path / "policy.md"
    source.write_text("# Expenses\n\nAbove 75 euros, ask first.\n", encoding="utf-8")
    try:
        record, duplicate = register_upload(services, source=source, filename="policy.md")
    finally:
        services.close()

    assert not duplicate
    stored = Path(record.stored_path or "")
    assert stored.is_file()
    assert not readable_by_others(stored), f"stored original is {mode_of(stored):o}"
    assert not readable_by_others(stored.parent), f"uploads dir is {mode_of(stored.parent):o}"


def test_a_restored_data_directory_is_private(tmp_path: Path) -> None:
    data = tmp_path / "source"
    data.mkdir()
    (data / "reed.db").write_bytes(b"registry")
    archive = tmp_path / "backup.tar.gz"
    create_backup(data, archive)

    target = tmp_path / "restored"
    restore_backup(archive, target)

    assert not readable_by_others(target), f"restored data dir is {mode_of(target):o}"
