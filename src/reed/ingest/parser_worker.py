"""Resource-isolated document parsing.

Parsing untrusted PDFs in the API process gives a malformed or deliberately
expensive file access to the server's memory budget. Each document therefore
gets a short-lived spawned process with a hard wall-clock deadline. Linux also
enforces address-space, CPU and file-descriptor limits.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, TypeAlias, cast

from reed.ingest.parsers import (
    DocumentLimitError,
    EmptyDocumentError,
    RawSection,
    UnsupportedFileError,
    parse_file,
)

_Result: TypeAlias = tuple[Literal["ok"], str, list[RawSection]] | tuple[Literal["error"], str, str]

_PUBLIC_ERRORS: dict[str, type[ValueError]] = {
    DocumentLimitError.__name__: DocumentLimitError,
    EmptyDocumentError.__name__: EmptyDocumentError,
    UnsupportedFileError.__name__: UnsupportedFileError,
}


class ParserIsolationError(RuntimeError):
    """The parser worker crashed or violated its isolation contract."""


@dataclass(frozen=True, slots=True)
class ParseJob:
    path: Path
    max_pages: int
    max_chars: int
    memory_mb: int
    cpu_seconds: int
    # Private deterministic hook for the timeout regression test. Production
    # callers always leave it at zero.
    delay_seconds: float = 0.0


def parse_file_isolated(
    path: Path,
    *,
    max_pages: int,
    max_chars: int,
    timeout_seconds: float,
    memory_mb: int,
    cpu_seconds: int,
) -> tuple[str, list[RawSection]]:
    job = ParseJob(
        path=path,
        max_pages=max_pages,
        max_chars=max_chars,
        memory_mb=memory_mb,
        cpu_seconds=cpu_seconds,
    )
    return _run_job(job, timeout_seconds=timeout_seconds)


def _run_job(job: ParseJob, *, timeout_seconds: float) -> tuple[str, list[RawSection]]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, job), name="reed-parser")
    process.daemon = True
    process.start()
    sender.close()

    try:
        if not receiver.poll(timeout_seconds):
            _stop(process)
            raise DocumentLimitError(
                f"Document parsing exceeded the configured timeout ({timeout_seconds:g} seconds)"
            )
        try:
            result = cast(_Result, receiver.recv())
        except EOFError as exc:
            raise ParserIsolationError(
                f"Parser worker exited unexpectedly (exit code {process.exitcode})"
            ) from exc
    finally:
        receiver.close()
        process.join(timeout=1)
        if process.is_alive():
            _stop(process)

    if result[0] == "ok":
        return result[1], result[2]

    _, error_name, message = result
    public_error = _PUBLIC_ERRORS.get(error_name)
    if public_error is not None:
        raise public_error(message)
    raise ParserIsolationError("Document parser failed inside its isolated worker")


def _worker(sender: Connection, job: ParseJob) -> None:
    try:
        _scrub_secrets()
        _apply_resource_limits(job)
        if job.delay_seconds:
            import time

            time.sleep(job.delay_seconds)
        kind, sections = parse_file(
            job.path,
            max_pages=job.max_pages,
            max_chars=job.max_chars,
        )
        result: _Result = ("ok", kind, sections)
    except Exception as exc:  # noqa: BLE001 — serialised for the trusted parent
        result = ("error", type(exc).__name__, str(exc))
    try:
        sender.send(result)
    finally:
        sender.close()


def _scrub_secrets() -> None:
    sensitive_fragments = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH")
    for name in list(os.environ):
        if any(fragment in name.upper() for fragment in sensitive_fragments):
            del os.environ[name]


def _apply_resource_limits(job: ParseJob) -> None:
    if not _is_linux():
        return

    import resource

    memory_bytes = job.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (job.cpu_seconds, job.cpu_seconds + 1))
    soft_files, hard_files = resource.getrlimit(resource.RLIMIT_NOFILE)
    descriptor_limit = min(32, soft_files, hard_files)
    resource.setrlimit(resource.RLIMIT_NOFILE, (descriptor_limit, descriptor_limit))


def _is_linux() -> bool:
    # Kept behind a function so strict static analysis on macOS does not mark
    # the Linux-only resource module as unreachable.
    return sys.platform == "linux"


def _stop(process: BaseProcess) -> None:
    process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)
