from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import reed.cli as cli
from reed.config import Settings
from reed.ingest.pipeline import IngestResult
from reed.ingest.registry import IndexGeneration


def generation(status: str = "active", *, error: str | None = None) -> IndexGeneration:
    return IndexGeneration(
        id=f"{status}-generation-id",
        logical_collection="test_chunks",
        physical_collection=f"test_chunks__{status}",
        fingerprint={"dense_model": "fake-embeddings", "dense_digest": "abc123digest"},
        status=status,  # type: ignore[arg-type]
        document_count=2,
        chunk_count=4,
        error=error,
        created_at="2026-08-04T00:00:00+00:00",
        completed_at="2026-08-04T00:01:00+00:00",
        activated_at="2026-08-04T00:01:00+00:00",
    )


def test_serve_passes_validated_options_to_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(profile="fake", data_dir=tmp_path, _env_file=None)
    called: dict[str, Any] = {}

    def record_uvicorn(target: str, **kwargs: object) -> None:
        called["target"] = target
        called.update(kwargs)

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("uvicorn.run", record_uvicorn)

    assert cli.main(["serve", "--host", "127.0.0.1", "--port", "8765", "--reload"]) == 0
    assert called["factory"] is True
    assert called["target"] == "reed.api.app:create_app"
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8765
    assert called["reload"] is True


@pytest.mark.parametrize("value", ["0", "65536", "not-a-number"])
def test_invalid_ports_are_rejected_by_the_parser(value: str) -> None:
    with pytest.raises(SystemExit) as raised:
        cli._build_parser().parse_args(["serve", "--port", value])
    assert raised.value.code == 2


def test_ingest_reports_every_outcome_and_closes_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = [tmp_path / name for name in ("ready.md", "duplicate.md", "error.md", "crash.md")]
    for path in paths:
        path.write_text("# Document\n\nText.", encoding="utf-8")

    closed = {"value": False}
    services = SimpleNamespace(close=lambda: closed.update(value=True))

    def ingest(_services: object, path: Path) -> IngestResult:
        if path.name == "crash.md":
            raise RuntimeError("provider failed")
        if path.name == "duplicate.md":
            return IngestResult("d-duplicate", "ready", chunks=2, duplicate=True)
        if path.name == "error.md":
            return IngestResult("d-error", "error", error="could not parse")
        return IngestResult("d-ready", "ready", chunks=3)

    monkeypatch.setattr("reed.services.build_services", lambda: services)
    monkeypatch.setattr("reed.ingest.pipeline.ingest_path", ingest)

    result = cli._ingest(argparse.Namespace(paths=[*paths, tmp_path / "missing.md"]))
    captured = capsys.readouterr()

    assert result == 1
    assert "ready.md: ingested (3 chunks)" in captured.out
    assert "duplicate.md: already ingested (2 chunks)" in captured.out
    assert "error.md: FAILED — could not parse" in captured.err
    assert "crash.md: FAILED — RuntimeError: provider failed" in captured.err
    assert "path does not exist" in captured.err
    assert closed["value"] is True


def test_ingest_rejects_an_empty_directory_and_unsupported_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    unsupported = tmp_path / "sheet.csv"
    unsupported.write_text("a,b", encoding="utf-8")
    services = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("reed.services.build_services", lambda: services)

    assert cli._ingest(argparse.Namespace(paths=[empty, unsupported])) == 1
    errors = capsys.readouterr().err
    assert "no supported documents" in errors
    assert "unsupported file type" in errors


def test_eval_forwards_options_and_writes_a_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: dict[str, Any] = {}

    class FakeReport:
        def write(self, directory: Path) -> tuple[Path, Path]:
            called["directory"] = directory
            return tmp_path / "report.md", tmp_path / "report.json"

        def to_markdown(self) -> str:
            return "# Evaluation"

        def summary_row(self) -> str:
            return "| fake |"

    def run_evaluation(**kwargs: object) -> FakeReport:
        called.update(kwargs)
        return FakeReport()

    monkeypatch.setattr("reed.evals.runner.run_evaluation", run_evaluation)

    result = cli._eval(
        argparse.Namespace(
            retrieval_only=True,
            k=3,
            judge="local",
            label="regression",
            summary_row=True,
        )
    )
    captured = capsys.readouterr()

    assert result == 0
    assert called["retrieval_only"] is True
    assert called["top_k"] == 3
    assert called["judge_profile"] == "local"
    assert "# Evaluation" in captured.out
    assert "| fake |" in captured.out
    assert "report.md" in captured.err


def test_main_translates_interrupts_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(profile="fake", data_dir=tmp_path, _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    monkeypatch.setattr(cli, "_serve", lambda _args: (_ for _ in ()).throw(KeyboardInterrupt))
    assert cli.main(["serve"]) == 130
    assert "interrupted" in capsys.readouterr().err

    monkeypatch.setattr(cli, "_serve", lambda _args: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main(["serve"]) == 2
    assert "RuntimeError: boom" in capsys.readouterr().err


def test_index_cli_reports_lifecycle_operations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    active = generation()
    previous = generation("previous")
    failed = generation("failed", error="RuntimeError: broken candidate")
    closed = {"count": 0}
    registry = SimpleNamespace(
        list_generations=lambda _collection: [active, previous, failed],
        list_by_status=lambda _statuses: ["one ready document"],
    )
    # The failed generation's collection is gone; the others answer with counts
    # that deliberately differ from the stale numbers stored on the rows.
    live = {active.physical_collection: 7, previous.physical_collection: 5}
    qdrant = SimpleNamespace(
        collection_exists=lambda name: name in live,
        count=lambda collection_name, **_kwargs: SimpleNamespace(count=live[collection_name]),
    )
    services = SimpleNamespace(
        settings=SimpleNamespace(collection="test_chunks"),
        registry=registry,
        qdrant=qdrant,
        close=lambda: closed.update(count=closed["count"] + 1),
    )
    monkeypatch.setattr("reed.services.build_services", lambda: services)
    monkeypatch.setattr(
        "reed.indexes.reindex",
        lambda _services: SimpleNamespace(generation=active, previous=previous),
    )
    monkeypatch.setattr("reed.indexes.rollback", lambda _services: (previous, active))

    def cleanup(_services: object, *, keep: int) -> list[IndexGeneration]:
        assert keep == 2
        return [failed]

    monkeypatch.setattr("reed.indexes.cleanup", cleanup)

    for command in ("status", "reindex", "rollback", "cleanup"):
        arguments = argparse.Namespace(index_command=command, keep=2)
        assert cli._index(arguments) == 0

    output = capsys.readouterr().out
    assert "model=fake-embeddings digest=abc123digest" in output
    # Counted now, not read from the row, which still claims 4.
    assert "1 ready document(s) in the registry." in output
    assert "chunks=7" in output
    assert "chunks=5" in output
    assert "chunks=missing" in output
    assert "chunks=4" not in output
    assert "error=RuntimeError: broken candidate" in output
    assert "Previous generation retained" in output
    assert "retained as previous" in output
    assert "Removed generation failed-gener" in output
    assert closed["count"] == 4


def test_index_cli_handles_empty_status_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = SimpleNamespace(list_generations=lambda _collection: [])
    services = SimpleNamespace(
        settings=SimpleNamespace(collection="test_chunks"),
        registry=registry,
        close=lambda: None,
    )
    monkeypatch.setattr("reed.services.build_services", lambda: services)

    def cleanup(_services: object, *, keep: int) -> list[IndexGeneration]:
        assert keep == 2
        return []

    monkeypatch.setattr("reed.indexes.cleanup", cleanup)

    assert cli._index(argparse.Namespace(index_command="status", keep=2)) == 0
    assert cli._index(argparse.Namespace(index_command="cleanup", keep=2)) == 0

    output = capsys.readouterr().out
    assert "No index generation" in output
    assert "No old index generations" in output


def test_backup_cli_creates_verifies_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "reed.db").write_bytes(b"registry")
    archive = tmp_path / "backup.tar.gz"
    settings = Settings(profile="fake", data_dir=data_dir, _env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    assert cli._backup(argparse.Namespace(backup_command="create", destination=archive)) == 0
    assert cli._backup(argparse.Namespace(backup_command="verify", archive=archive)) == 0
    data_dir.rename(tmp_path / "old-data")
    assert cli._backup(argparse.Namespace(backup_command="restore", archive=archive)) == 0

    output = capsys.readouterr().out
    assert "Created" in output
    assert "Verified" in output
    assert "Restored 1 files" in output
    assert (data_dir / "reed.db").read_bytes() == b"registry"
