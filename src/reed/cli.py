"""Command line entry point: ``reed serve``."""

from __future__ import annotations

import argparse
import sys

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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(get_settings().log_level)
    return _serve(args)


if __name__ == "__main__":
    sys.exit(main())
