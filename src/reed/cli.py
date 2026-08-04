"""Command line entry point: ``reed serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reed import __version__
from reed.config import get_settings
from reed.log import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reed", description="Reed — self-hosted RAG service")
    parser.add_argument("--version", action="version", version=f"reed {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP API and chat UI")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="Autoreload on code changes")

    ingest = sub.add_parser("ingest", help="Ingest files or folders without starting the server")
    ingest.add_argument("paths", nargs="+", type=Path)

    evaluate = sub.add_parser("eval", help="Run the evaluation suite over the golden question set")
    evaluate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the LLM-judged metrics — needs no model at all",
    )
    evaluate.add_argument("--k", type=int, default=None, help="Override top_k for this run")
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
    uvicorn.run(
        "reed.api.app:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
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
            if path.is_dir():
                targets = sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            else:
                targets = [path]

            for target in targets:
                result = ingest_path(services, target)
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
    )
    markdown, _ = report.write(RESULTS_DIR)

    print(report.to_markdown())
    if args.summary_row:
        from reed.evals.report import SUMMARY_HEADER

        print(f"\n{SUMMARY_HEADER}\n{report.summary_row()}")
    print(f"\nWritten to {markdown}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(get_settings().log_level)

    if args.command == "ingest":
        return _ingest(args)
    if args.command == "eval":
        return _eval(args)
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
