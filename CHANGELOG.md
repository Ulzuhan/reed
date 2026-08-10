# Changelog

All notable changes to Reed are documented here. The project follows Semantic Versioning while it
is pre-1.0: minor releases may change operational behavior, and patch releases contain compatible
fixes.

## [Unreleased]

### Fixed

- Embed the query outside the embedded-Qdrant lock. Retrieval delegated the whole search to
  `langchain-qdrant`, which embeds the query inside the call — so the provider round-trip ran while
  the process-global vector lock was held, and every concurrent `/v1/ask`, every `/v1/search` and
  every ingestion commit waited on it. The lock now covers only the database call, matching what
  the ingestion path already did. Reed builds the Qdrant request itself; it is deliberately the
  same request as before, down to the RRF fusion and the per-branch prefetch limit, because the
  calibrated evidence threshold is a number in that fused score domain. A new integration test
  pins the two implementations together, and the shipped evaluation returns byte-identical
  retrieval metrics.
- Only embed the vectors the queried mode uses: a `sparse` search no longer pays for a dense
  round-trip it discards, and `dense` no longer computes a sparse vector.

### Removed

- `Services.retrieval_store()` and the per-mode store cache behind it. They existed to give
  `dense` and `sparse` their own query views of one physical index; issuing the query directly
  makes the mode a parameter of the request instead of a property of a cached object.

### Fixed

- Catch the documentation up with 0.5.x. The architecture overview still described the restore that
  0.5.0 replaced — an adjacent scratch directory and an atomic rename — which was the one stale
  line that contradicted the code rather than merely aging. The README promised OCR "deferred to
  v0.5" from a v0.5 release, claimed a CI matrix that stops at Python 3.13, and told upgraders to
  install v0.4; the architecture and runbook documents were still stamped v0.4. The configuration
  table also gains the two `/v1/search` controls and notes that `REED_DATA_DIR` is created `0700`.

## [0.5.1] - 2026-08-09

### Security

- Keep `REED_DATA_DIR` to the account that runs Reed. Nothing set a mode or a umask, so on a
  bare-metal install the data directory came out `0755` and the registry and every stored original
  `0644` — readable by any other local account, which is precisely what a private RAG should not
  do. Reed now creates the directory `0700` and writes what it stores `0600`. A directory that
  already exists is left alone, since an operator may have widened it deliberately, but startup
  warns when the mode is wider and names the `chmod` that closes it. The container was never
  affected: one unprivileged user, an isolated volume.

### Fixed

- Say so when a failed restore cannot clean up after itself. The rollback swallows its own errors
  on purpose — they must not mask the failure that caused it — but it promised the target was left
  "empty, or absent" without checking. A cleanup that fails now logs what it left behind, instead
  of surfacing later as a confusing "Refusing to overwrite non-empty data directory".

## [0.5.0] - 2026-08-08

### Added

- `POST /v1/search`: ranked evidence with no generation, for callers whose own model writes the
  answer — MCP servers, agent hosts, external UIs — and for inspecting what the index returns for
  a query. It returns the same numbered `sources` as `/v1/ask`, reports the evidence threshold
  instead of applying it (`sufficient_evidence`, `min_evidence_score`), and draws on its own
  rate-limit bucket, `REED_SEARCH_RATE_LIMIT_PER_MINUTE` (default 120/minute), and its own
  retrieval-thread budget, `REED_MAX_CONCURRENT_SEARCHES` (default 16), so a burst of searches
  queues against itself instead of starving uploads and `/v1/ask`.

### Changed

- Raise the pinned Compose image to Qdrant v1.19.0. The v1.18.3 pin carries
  GHSA-5p4m-2wfm-xmqj (quadratic CPU in the dashboard's bundled js-yaml); v1.19.0 does not fix it
  either — js-yaml 4.3.1 landed a day after that release was cut — but the "track the latest
  upstream release" policy asks for the bump regardless.
- Raise the pypdf floor to 6.15. CVE-2026-71852 and CVE-2026-71870 are resource-exhaustion bugs
  reachable from a crafted PDF — large CID font width ranges and large `/ToUnicode` streams — and
  reed parses uploaded PDFs, although ingestion already isolates parsing in a spawned process
  under wall-clock and resource limits.

### Fixed

- Restore a backup into a Compose volume. `reed backup restore` staged the extracted tree in
  `REED_DATA_DIR`'s *parent*, which under the shipped Compose file is the container's read-only
  root filesystem — so the scratch directory could not be created, the mountpoint could not be
  removed, and the final rename would have crossed filesystems. Staging now happens inside the
  target, which makes every rename same-filesystem by construction. A restore that fails partway
  leaves the target exactly as it found it. The containerized path is now exercised by CI, which
  restores into a freshly created volume and serves from the result.

## [0.4.1] - 2026-08-05

### Fixed

- Apply the ingestion guards to `PUT /v1/documents/{logical_id}`. The rate limit, the body-size
  limit and the vectorstore readiness gate matched only the `POST` routes, and a multipart body is
  spooled to disk before route code can act — so a replacement upload could bypass every limit
  that plain uploads were given, including on a deployment with no API key configured.
- Keep a just-published document servable when retiring its superseded versions fails. A transient
  Qdrant error during that flush marked the fresh version as `error` and enqueued the deletion of
  its own committed points, leaving the lineage with nothing servable until a re-upload.
  Retirement is now fault-tolerant, and a ready document can no longer be overwritten into
  `error`.
- Move upload hashing and staging off the event loop. Registering an upload ran SHA-256 and a copy
  of up to the full upload size synchronously, stalling in-flight SSE streams on every upload.
- Probe Ollama readiness with its own timeout — `REED_READINESS_PROBE_TIMEOUT_SECONDS`, default
  5 seconds — instead of the 120-second generation timeout. A hung Ollama held the readiness lock
  for the full generation timeout and serialized every probe behind it.
- Claim a document's name inside the transaction that records the upload. Two concurrent uploads
  with the same name could both pass the duplicate check and create two lineages with the same
  visible name.
- Strip an HTML element carrying the `hidden` attribute even when it also declares
  `aria-hidden="false"` — `aria-hidden` changes screen-reader behaviour, not visibility, so the
  combination smuggled invisible text past the stripping that exists to prevent exactly that.
- Close resources shared with ingestion workers only after those workers have exited, instead of
  closing under them after a one-second join.
- Localize the insufficient-evidence refusal into the six languages the no-context answer already
  speaks; a question asked in French was refused in English.
- Recognize Spanish inverted-question follow-ups («¿y si…?»), which the follow-up detection
  itself enumerates, by not anchoring the match to the start of the question.
- Print calibrated evidence thresholds rounded in evaluation reports instead of at full float
  precision.
- Catch the documentation up with the 0.4.0 release: the README, SECURITY.md, the architecture
  overview and the runbooks still described v0.3, and `uv.lock` still recorded 0.3.0. CI now
  fails when the lockfile drifts (`uv lock --check`).

### Changed

- Run the container on Python 3.14.6, on a current base image build, and test the package against
  Python 3.14 in CI.
- Raise the pinned Compose images to Qdrant v1.18.3 and Ollama 0.32.5, and dependency floors to
  uvicorn 0.52.1 and selectolax 0.4.
- Watch the Compose image digests with Dependabot alongside the Dockerfile's — pinned digests
  without an updater go stale.

### Added

- A short `CONTRIBUTING.md`: single-author project, issues welcome, pull requests coordinated
  through an issue first.

## [0.4.0] - 2026-08-05

### Added

- Replace a document with `PUT /v1/documents/{logical_id}`. The version currently serving stays
  ready and indexed until the new one is, so a replacement never leaves the document unfindable.
  Superseded versions keep their registry row and their stored original but lose their chunks:
  they are not indexed at all rather than filtered at query time, so reindexing and retrieval need
  no notion of them. `GET .../versions` shows the history, `DELETE .../versions/{n}` forgets one
  superseded version, and deleting a lineage id removes every version — deleting only the current
  one would make an older version reappear in search results.
- Refuse an upload whose name already belongs to a document, naming the lineage to `PUT` to
  instead of silently creating a second copy. A genuinely different document that happens to share
  a filename can pass its own `name`. A failed upload is still re-uploadable, which is how a retry
  works.
- Give every document a logical identity that a future replacement can address: an opaque
  server-assigned `logical_id`, a display `name` and a `version`, all exposed on the upload
  response and the document list. A document's `id` is still derived from its content hash, which
  is precisely why it cannot serve as the identity of something whose content is about to change.
  Registries from earlier releases migrate in place, each existing document becoming version 1 of
  its own lineage. Replacement itself follows separately.
- Ingest `.html` and `.htm`. Scripts, styles and embedded objects are discarded, along with
  anything hidden by an inline style, the `hidden` attribute or `aria-hidden` — text a reader
  cannot see is fully visible to the embedder and to the prompt, which makes an HTML upload a place
  to smuggle instructions past whoever approved the document. Text hidden by a stylesheet rule is
  not detected, because Reed parses HTML and does not apply CSS; the limit is documented rather
  than implied. Parsing fetches nothing. Headings become Markdown and tables one line per row, as
  for DOCX.
- Ingest `.docx`. Word heading styles become Markdown headings, so section labelling and citations
  work exactly as they do for Markdown, and tables become one line per row with cell pipes escaped.
  Extraction applies tracked changes — insertions in, deletions out. Comments and footnotes are not
  ingested: a document's suggestions are not what it says. An archive declaring an implausible
  expansion is refused before anything is decompressed, and a renamed `.doc` is named as such
  rather than failing obscurely.

### Changed

- Record the extraction pipeline version in the collection fingerprint, alongside the model
  identity and chunking it already covered. A change to a parser, to section splitting, or to how
  embedding input is composed now changes the fingerprint, so it is detected rather than mixed
  into an existing index. **Every index built by an earlier release mismatches on upgrade**: run
  `reed index reindex`, which builds and activates a compatible generation and keeps the current
  one available for rollback.

### Fixed

- Let `reed backup create` run against a real deployment. FastEmbed's model cache lives inside
  `REED_DATA_DIR` in the container, because a read-only root filesystem leaves nowhere else
  persistent, and it is a HuggingFace tree whose snapshots are symlinks into blobs — which the
  archive refused, so backups failed on the default configuration. Derived caches are skipped;
  a symlink anywhere else is still refused.
- Refuse a collection whose fingerprint does not match, instead of adopting it. Adoption existed
  to carry a Reed 0.2 index into 0.3, but 0.2 embedded the raw chunk text while 0.3 embeds a
  title/section header, so it merged two incompatible kinds of vector into one collection with
  nothing left able to detect it. The path had no test coverage.
- Count `reed index status` from the registry and Qdrant instead of printing the counters stored
  when a generation was registered. An index adopted at startup recorded zeros and nothing updated
  them, so the command reported an empty active index for every deployment that had never
  reindexed, which is the state it exists to diagnose.
- Report a truncated or corrupt backup archive as such. `reed backup verify` surfaced the raw
  lookup failure for the missing manifest, mid-recovery, when it is least welcome. Altered file
  contents still name the file whose checksum failed.
- Index documents that become ready while a reindex is building. Ingestion writes to the active
  collection, so a document uploaded mid-build was committed in the registry and absent from the
  index that reindex then activated: unreachable by search and refused as a duplicate on re-upload.
  The candidate now re-checks the ready set until it stops growing, and refuses to activate rather
  than activate a known gap.
- Leave a `building` generation alone during `reed index cleanup`, and refuse to delete its registry
  row, so cleanup cannot destroy a candidate another process is still filling.
- Resolve the evidence threshold from the retrieval mode actually queried rather than the configured
  one. RRF, dense, sparse and cross-encoder scores share no numeric scale.
- Restore backup files with their recorded permissions instead of inheriting the umask, which
  widened a `0600` registry to `0644`. setuid, setgid and sticky bits are dropped.
- Publish the `latest` container tag from release tags only. It was gated on `is_default_branch`,
  which is also true for a tag push, so `main` and `v*` alternately claimed it; since Compose
  defaults to `latest`, an unpinned deployment could run an unreleased `main` build.
- Scope the reusable CI quality gate's concurrency group to its caller. Release and Docker both
  run on a tag push and both call it, so a group built only from the ref made one cancel the
  other's quality gate, and with it the whole run, leaving the versioned container manifest
  unpublished for v0.3.0.

### Removed

- `Settings.effective_fetch_k`, which nothing used and which contradicted retrieval by ignoring the
  diversity selection that also widens the candidate pool.

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

## [0.1.0] - 2026-08-04

First public release: a self-hosted RAG service. Upload PDF, Markdown or text files, ask
questions, and get streamed answers whose claims carry citations traceable to the retrieved
passage. Hybrid retrieval (dense embeddings + BM25 fused with Reciprocal Rank Fusion, optional
local cross-encoder reranking), a fully local profile via Ollama, SSE streaming with sources
ahead of the first token, a built-in golden-set evaluation suite, and an embedded-Qdrant mode
that runs without Docker.

[Unreleased]: https://github.com/Ulzuhan/reed/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/Ulzuhan/reed/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ulzuhan/reed/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Ulzuhan/reed/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Ulzuhan/reed/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Ulzuhan/reed/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ulzuhan/reed/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ulzuhan/reed/releases/tag/v0.1.0
