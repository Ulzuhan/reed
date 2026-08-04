# Architecture

How Reed is put together, and why each piece is the way it is. The [README](../README.md) covers
what it does; this covers how.

## Layout

```
src/reed/
├── config.py       Settings — one profile switches the whole provider stack
├── providers.py    OpenAI · Ollama · fake, behind two langchain-core interfaces
├── services.py     Lazily built container shared by the API, CLI and evaluator
├── api/            FastAPI app, routes, SSE framing
├── ingest/         parsers → chunking → registry → pipeline
├── rag/            vectorstore → retriever → prompts → chain
├── evals/          dataset → retrieval metrics → judge → report → runner
└── static/         The chat UI: three files, no build step
```

Two rules keep this navigable. Nothing outside `providers.py` knows which model vendor is
configured — everything else sees `BaseChatModel` and `Embeddings`. And nothing outside
`services.py` constructs anything expensive, so a process that only serves `/health` never opens a
model client.

## Ingestion

```
file → sha256 → dedup → parse → chunk → embed (dense + sparse) → upsert → registry: ready
```

**Sections before chunks.** A parser returns *sections*, the smallest unit that still carries a
citable location: one per page for PDFs, one for the whole file otherwise. Chunking then happens
inside a section, so a chunk never straddles a page boundary and `handbook.pdf, p. 3` is always
true.

**Characters, not tokens.** Chunk sizes are measured in characters. Tokenizers disagree between
providers, and Reed must chunk identically whether the embeddings come from OpenAI or a local
Gemma. The cost is slightly uneven token counts per chunk.

**Idempotency comes from content.** The document id is `d-<sha256[:12]>` and each chunk's point id
is `uuid5(NAMESPACE, "<sha256>:<chunk_index>")`. Re-ingesting the same file overwrites exactly the
same points. This matters most after a crash: a half-finished ingestion leaves no orphans, because
the retry writes to the same ids.

**The registry is sqlite.** Chunks live in Qdrant; per-document bookkeeping — status, error message,
chunk count, content hash — lives in a single sqlite table in WAL mode. No ORM. Ingestion runs in a
background thread while requests read from another, which is what the connection lock is for.

## Storage

One collection, two named vectors:

| Vector | Source | Purpose |
|---|---|---|
| `dense` | The configured embedding model | Semantic similarity — finds paraphrases |
| `sparse` | FastEmbed BM25, with Qdrant computing IDF | Lexical match — finds exact terminology |

The dense size is **probed at startup**, never hardcoded (`text-embedding-3-small` is 1536,
EmbeddingGemma 768). If the collection already exists with a different size, startup fails with an
explanation naming the fix. Silently querying a collection built by another model is the kind of
bug that produces plausible, wrong answers for weeks.

Embedded and server Qdrant use the same client and the same query API, including hybrid search.
Embedded mode holds a file lock, so a running server and an evaluation cannot share a path — which
is why evaluations always get a temporary directory.

## Retrieval

Both vectors are searched, and Qdrant fuses the two rankings with Reciprocal Rank Fusion
server-side. Neither the round trip nor the fusion logic lives in Python.

Why hybrid at all: dense retrieval fails on rare exact terms (an internal tool name, an error code,
a policy label) because the embedding has never seen them meaningfully. Sparse retrieval fails when
the question shares no vocabulary with the passage. Real questions do both, often in the same
sentence.

Reranking is optional and off by default. When enabled, hybrid search fetches `REED_FETCH_K`
candidates and a local ONNX cross-encoder rescores them down to `REED_TOP_K`. A cross-encoder reads
the question and the chunk *together*, which catches passages that merely share vocabulary — at the
cost of a model download and per-query latency.

## Generation

The system prompt injects retrieved chunks as numbered blocks and states the contract: answer only
from the excerpts, mark every claim with `[n]`, say plainly when the excerpts do not answer the
question.

Those `[n]` numbers are the same ones sent in the `sources` SSE event, which is what makes a
citation clickable rather than decorative.

Refusal is a feature, not a failure mode. The `negative` questions in the golden set exist to
measure it, because a model that would rather invent an answer than admit silence is worse than
useless for document QA.

### The event stream

`answer_stream()` is one async generator yielding typed events, consumed by three callers: the SSE
endpoint, the non-streaming JSON endpoint, and the evaluation runner. One code path means the
evaluation measures the same behaviour users get.

```
SourcesEvent → TokenEvent* → DoneEvent          (or ErrorEvent, at any point)
```

Retrieval is synchronous — the Qdrant client and the ONNX reranker both block — so it runs in a
worker thread rather than stalling the event loop.

On the wire, a producer task feeds an `asyncio.Queue` while the response generator reads it with a
timeout. A timeout emits a heartbeat comment and cancels only the queue read, never the producer:
a slow first token becomes a `: ping` instead of a connection a proxy decides to drop.

## Evaluation

The suite ingests `eval/corpus/` through the real pipeline into a throwaway embedded Qdrant, then
asks every question in `eval/golden.jsonl` through the real answering code.

Ground truth is recorded per document, not per chunk. Chunk boundaries move whenever `chunk_size`
changes; "the expenses policy answers this" stays true, which keeps the golden set comparable
across configurations.

Retrieval metrics need no model, so they run in CI and on a laptop with no keys. Judged metrics are
one structured-output call each, concurrency-limited, and cached on disk keyed by judge model plus
the exact text judged — so re-running after a retrieval change only pays for answers that actually
changed. With no judge reachable, the run reports retrieval metrics and exits successfully rather
than failing.

## Testing

Everything runs on the `fake` profile: hash-based deterministic embeddings and a scripted chat model
that emits a cited answer. Integration tests pair those with a **real** embedded Qdrant and **real**
BM25 sparse vectors, so the named-vector collection, the hybrid query and the deletion filter are
genuinely exercised — with no API key, no network and no hand-written mocks.

The container path is covered by a CI job that builds the image, brings the compose stack up and
drives an upload-then-ask round trip. That job exists because the machine this was developed on has
no Docker: CI is the test bench, not a rubber stamp.
