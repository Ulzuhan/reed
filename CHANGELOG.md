# Changelog

All notable changes to Reed are documented here. The project follows Semantic Versioning while it
is pre-1.0: minor releases may change operational behaviour, and patch releases contain compatible
fixes.

## [Unreleased]

### Fixed

- Publish the `latest` container tag from release tags only. It was gated on `is_default_branch`,
  which is also true for a tag push, so `main` and `v*` alternately claimed it; since Compose
  defaults to `latest`, an unpinned deployment could run an unreleased `main` build.
- Scope the reusable CI quality gate's concurrency group to its caller. Release and Docker both
  run on a tag push and both call it, so a group built only from the ref made one cancel the
  other's quality gate, and with it the whole run, leaving the versioned container manifest
  unpublished for v0.3.0.

## [0.3.0] - 2026-08-04

### Index safety and recovery

- Bind each physical Qdrant collection to a versioned fingerprint containing immutable model
  digest/revision, quantization, dimension, task prefixes, sparse model and chunking settings.
- Add `reed index status|reindex|rollback|cleanup`: complete candidates build and verify away from
  the active collection, activate atomically, retain a compatible rollback generation and leave a
  serving index untouched on failure.
- Reject registries created by a newer Reed schema before mutation and verify every retained
  original against its recorded SHA-256 during reindex.
- Add offline `reed backup create|verify|restore` archives with manifests, per-file checksums,
  traversal/link/device rejection and atomic restore into an empty target.

### Retrieval and evaluation

- Expand the golden suite to 41 questions over 10 documents, covering factual, multi-hop,
  negative, Spanish/cross-language, conversational, table and prompt-injection cases with exact
  evidence labels.
- Report evidence Recall@k, nDCG@k, MRR, complete multi-hop evidence, negative abstention, overall
  abstention accuracy, deterministic bootstrap intervals and calibrated score thresholds.
- Preserve exact chunks, scores, matched evidence, citation status/warnings and reproducibility
  provenance including dataset hashes, git SHA, hardware, package versions and immutable model
  identity. Package the evaluation data so installed-wheel evaluation works from an empty folder.
- Add dense, sparse, hybrid and hybrid-rerank query modes over the same complete index, optional
  multilingual reranking adapters, relevance/diversity selection and per-document chunk caps.
- Inject filename/section metadata only into embedding input, keep displayed/cited text clean, and
  move dense/sparse inference outside the embedded-Qdrant critical section.
- Keep EmbeddingGemma as the local default after an equal-configuration A/B against
  Qwen3-Embedding 0.6B; publish metrics, confidence intervals, immutable digests and the decision
  record. Qwen remains an opt-in preset.
- Enable the calibrated `0.8333333333333333` evidence threshold only in its measured score domain: local
  EmbeddingGemma, hybrid RRF, no reranker. Other modes/models require explicit calibration.

### Runtime and operations

- Replace unbounded background ingestion with a durable bounded queue, fixed worker concurrency,
  restart refill, explicit `queued → parsing → embedding → indexing → ready` stages and `503`
  backpressure with `Retry-After`.
- Add cached local-chat readiness, authenticated Prometheus metrics, no-store API responses and
  instrumentation for request outcomes, queue/ingestion state, retrieval latency and abstentions.
- Add upload progress, structured errors, stage polling and cooperative stream stop controls to the
  build-free UI.
- Document the official single-node boundary, coordinated remote-Qdrant backups, recovery,
  reindex/rollback and monitoring runbooks. OCR, DOCX/HTML and document versioning remain deferred
  to v0.4.

### Packaging, release and licensing

- Make release packaging depend on the complete reusable CI gate, verify tag/version and that the
  tag points at current `origin/main`, rebuild the wheel from the sdist, inspect packaged data and
  smoke-test retrieval-only evaluation from an empty directory.
- Keep explicit Compose parity for every `REED_*` setting and document the relationship between
  upload limits and configurable tmpfs sizing.
- License Reed v0.3 and later under Apache License 2.0. The immutable v0.1.x/v0.2.x release
  contents retain their original MIT license; separately downloaded model weights keep their own
  upstream terms.

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

[Unreleased]: https://github.com/Ulzuhan/reed/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ulzuhan/reed/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ulzuhan/reed/compare/v0.1.0...v0.2.0
