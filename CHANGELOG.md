# Changelog

All notable changes to Reed are documented here. The project follows Semantic Versioning while it
is pre-1.0: minor releases may change operational behaviour, and patch releases contain compatible
fixes.

## [Unreleased]

## [0.2.0] - 2026-08-04

### Security

- Authenticate, rate-limit and bound request bodies at the ASGI stream before multipart or JSON
  parsing, including chunked requests without `Content-Length`.
- Parse documents in a separate process with wall-clock deadlines and Linux CPU, memory and
  file-descriptor limits; harden the container with a read-only root filesystem, dropped
  capabilities and `no-new-privileges`.
- Keep retrieved excerpts out of the system message, treat them as untrusted prompt data and audit
  citation coverage, cited figures and verbatim quotes.
- Prevent user-controlled document identifiers from forging log entries.
- Pin third-party actions and container inputs by immutable digest or commit, scan dependencies,
  repository history and built images, and run CodeQL's extended security suite.

### Reliability

- Make vector-store bootstrap single-flight, bounded at startup and self-healing after transient
  Qdrant failures, with fast readiness responses and retry guidance.
- Make ingestion recoverable and deterministic across restarts, concurrent work, partial upserts
  and deletion conflicts.
- Enforce collection fingerprints so incompatible embedding, sparse-vector or chunking settings
  fail explicitly instead of mixing vectors silently.
- Bound concurrent generation, add per-client request limits and preserve authenticated CORS
  preflight behaviour.

### Quality

- Expand the suite to 187 tests across Python 3.11, 3.12 and 3.13 with an 85% branch-coverage gate,
  strict mypy and Ruff checks.
- Add a real Chromium upload, readiness, streaming-answer, citation and XSS regression plus a full
  Docker Compose upload-to-answer smoke test.
- Publish native `linux/amd64` and `linux/arm64` images with SBOM and provenance after every quality
  gate passes.
- Document operational limits, private vulnerability reporting, secure deployment and all new
  configuration controls.

[Unreleased]: https://github.com/Ulzuhan/reed/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Ulzuhan/reed/compare/v0.1.0...v0.2.0
