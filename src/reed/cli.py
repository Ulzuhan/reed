"""Command line entry point: ``reed serve``."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from reed import __version__
from reed.config import get_settings
from reed.log import get_logger, setup_logging

logger = get_logger(__name__)


def _bounded_integer(value: str, *, minimum: int, maximum: int, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _port(value: str) -> int:
    return _bounded_integer(value, minimum=1, maximum=65_535, name="port")


def _top_k(value: str) -> int:
    return _bounded_integer(value, minimum=1, maximum=50, name="k")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reed", description="Reed — self-hosted RAG service")
    parser.add_argument("--version", action="version", version=f"reed {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP API and chat UI")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=_port, default=None)
    serve.add_argument("--reload", action="store_true", help="Autoreload on code changes")

    ingest = sub.add_parser("ingest", help="Ingest files or folders without starting the server")
    ingest.add_argument("paths", nargs="+", type=Path)

    index = sub.add_parser("index", help="Manage safe physical index generations")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    index_sub.add_parser("status", help="Show active, previous and failed generations")
    index_sub.add_parser("reindex", help="Build and atomically activate a complete new index")
    index_sub.add_parser("rollback", help="Reactivate the most recent compatible generation")
    cleanup = index_sub.add_parser("cleanup", help="Delete old non-active generations")
    cleanup.add_argument(
        "--keep",
        type=lambda value: _bounded_integer(value, minimum=1, maximum=20, name="keep"),
        default=2,
        help="Keep this many active/previous generations (default: 2)",
    )

    backup = sub.add_parser("backup", help="Create, verify or restore an offline backup")
    backup_sub = backup.add_subparsers(dest="backup_command", required=True)
    create = backup_sub.add_parser("create", help="Archive REED_DATA_DIR with checksums")
    create.add_argument("destination", nargs="?", type=Path, default=None)
    verify = backup_sub.add_parser("verify", help="Verify every file against the manifest")
    verify.add_argument("archive", type=Path)
    restore = backup_sub.add_parser("restore", help="Restore into an empty REED_DATA_DIR")
    restore.add_argument("archive", type=Path)

    evaluate = sub.add_parser("eval", help="Run the evaluation suite over the golden question set")
    evaluate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and the judge — only the configured embeddings are used",
    )
    evaluate.add_argument("--k", type=_top_k, default=None, help="Override top_k for this run")
    evaluate.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid", "hybrid_rerank"],
        default=None,
        help="Retrieval mode to isolate in this run",
    )
    evaluate.add_argument(
        "--judge",
        choices=["openai", "local"],
        default=None,
        help="Which provider scores the answers (default: REED_EVAL_JUDGE_PROFILE)",
    )
    evaluate.add_argument("--label", default=None, help="Name this run in the report")
    evaluate.add_argument(
        "--summary-row",
        action="store_true",
        help="Also print the run as a Markdown table row, for the README",
    )

    return parser


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.host
    if host not in {"127.0.0.1", "localhost", "::1"} and not settings.api_key:
        logger.warning(
            "serving on %s without REED_API_KEY; restrict the network or configure auth",
            host,
        )
    uvicorn.run(
        "reed.api.app:create_app",
        factory=True,
        host=host,
        port=args.port if args.port is not None else settings.port,
        reload=args.reload,
        proxy_headers=True,
        log_level=settings.log_level.lower(),
    )
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from reed.ingest.parsers import SUPPORTED_SUFFIXES
    from reed.ingest.pipeline import ingest_path
    from reed.services import build_services

    services = build_services()
    failures = 0
    try:
        for path in args.paths:
            if not path.exists():
                print(f"{path}: FAILED — path does not exist", file=sys.stderr)
                failures += 1
                continue
            if path.is_dir():
                targets = sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
                if not targets:
                    print(f"{path}: FAILED — no supported documents found", file=sys.stderr)
                    failures += 1
            else:
                if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    print(f"{path}: FAILED — unsupported file type", file=sys.stderr)
                    failures += 1
                    continue
                targets = [path]

            for target in targets:
                try:
                    result = ingest_path(services, target)
                except Exception as exc:
                    logger.debug("could not ingest %s", target, exc_info=True)
                    print(
                        f"{target.name}: FAILED — {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    failures += 1
                    continue
                if result.duplicate:
                    print(f"{target.name}: already ingested ({result.chunks} chunks)")
                elif result.status == "ready":
                    print(f"{target.name}: ingested ({result.chunks} chunks)")
                else:
                    print(f"{target.name}: FAILED — {result.error}", file=sys.stderr)
                    failures += 1
    finally:
        services.close()

    return 1 if failures else 0


def _eval(args: argparse.Namespace) -> int:
    from reed.evals.dataset import RESULTS_DIR
    from reed.evals.runner import run_evaluation

    report = run_evaluation(
        retrieval_only=args.retrieval_only,
        top_k=args.k,
        judge_profile=args.judge,
        label=args.label,
        retrieval_mode=getattr(args, "mode", None),
    )
    markdown, _ = report.write(RESULTS_DIR)

    print(report.to_markdown())
    if args.summary_row:
        from reed.evals.report import SUMMARY_HEADER

        print(f"\n{SUMMARY_HEADER}\n{report.summary_row()}")
    print(f"\nWritten to {markdown}", file=sys.stderr)
    return 0


def _index(args: argparse.Namespace) -> int:
    from reed.indexes import cleanup, reindex, rollback
    from reed.services import build_services

    services = build_services()
    try:
        if args.index_command == "status":
            generations = services.registry.list_generations(services.settings.collection)
            if not generations:
                print("No index generation has been registered yet.")
                return 0
            for generation in generations:
                digest = str(generation.fingerprint.get("dense_digest", ""))
                digest_label = digest[:12] if digest else "unresolved"
                print(
                    f"{generation.status:8} {generation.id[:12]} "
                    f"collection={generation.physical_collection} "
                    f"documents={generation.document_count} chunks={generation.chunk_count} "
                    f"model={generation.fingerprint.get('dense_model', 'unknown')} "
                    f"digest={digest_label}"
                )
                if generation.error:
                    print(f"         error={generation.error}")
            return 0
        if args.index_command == "reindex":
            result = reindex(services)
            print(
                f"Activated generation {result.generation.id[:12]} "
                f"({result.generation.document_count} documents, "
                f"{result.generation.chunk_count} chunks)."
            )
            if result.previous is not None:
                print(f"Previous generation retained for rollback: {result.previous.id[:12]}")
            return 0
        if args.index_command == "rollback":
            activated, replaced = rollback(services)
            print(
                f"Activated generation {activated.id[:12]}; "
                f"generation {replaced.id[:12]} is retained as previous."
            )
            return 0
        removed = cleanup(services, keep=args.keep)
        for generation in removed:
            print(f"Removed generation {generation.id[:12]} ({generation.physical_collection}).")
        if not removed:
            print("No old index generations needed cleanup.")
        return 0
    finally:
        services.close()


def _backup(args: argparse.Namespace) -> int:
    from reed.backups import create_backup, restore_backup, verify_backup

    settings = get_settings()
    if args.backup_command == "create":
        destination = args.destination or Path(
            f"reed-backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
        )
        manifest = create_backup(settings.data_dir, destination)
        print(f"Created {destination} ({len(manifest.files)} files).")
        return 0
    if args.backup_command == "verify":
        manifest = verify_backup(args.archive)
        print(
            f"Verified {args.archive}: {len(manifest.files)} files, "
            f"Reed {manifest.reed_version}, created {manifest.created_at}."
        )
        return 0
    manifest = restore_backup(args.archive, settings.data_dir)
    print(f"Restored {len(manifest.files)} files into {settings.data_dir}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        setup_logging(get_settings().log_level)
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "eval":
            return _eval(args)
        if args.command == "index":
            return _index(args)
        if args.command == "backup":
            return _backup(args)
        return _serve(args)
    except KeyboardInterrupt:
        print("reed: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.debug("command failed", exc_info=True)
        print(f"reed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
