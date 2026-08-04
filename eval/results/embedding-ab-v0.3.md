# Reed v0.3 embedding A/B

Decision record for the default local embedding model. This is a retrieval-only comparison; no
chat model or LLM judge took part.

## Method

- Reed commit: `3e56a207a38a89a6ab62cb795aac33f67536fef5` plus the uncommitted v0.3
  evaluation implementation under test
- Date and host: 2026-08-04, macOS arm64, 10 logical CPUs, Python 3.13.14
- Corpus: 10 synthetic documents; SHA-256
  `b4bff755b463dc01aecd2a6574c443b373e38b4a08fc02bdf2c3df309c59709c`
- Golden set: 41 questions (35 answerable, 6 negative); SHA-256
  `3d02db91ae3945c123d0cb19e248433126809ea46a3e111f6d38cf2dbd4f57c3`
- Exact evidence labels: `eval/evidence.json`; every answerable label is a normalized verbatim
  excerpt from its source
- Retrieval: hybrid dense + BM25, RRF, `top_k=4`, `fetch_k=20`, reranking off,
  diversity on (`lambda=0.7`, at most two chunks per document)
- Chunking: 1,000 characters with 150-character overlap
- Confidence intervals: deterministic 1,000-sample bootstrap, seed 0, percentile 95% interval
- Abstention was disabled during measurement (`REED_MIN_EVIDENCE_SCORE=0`) so calibration could
  compare the same complete score distribution for both candidates

The only intentional changes between runs were the dense model and its model-prescribed query
prefix. Document embeddings received the same Reed title/section metadata in both runs.

## Results

| Metric | EmbeddingGemma | Qwen3-Embedding 0.6B | Difference (Qwen − Gemma) |
|---|---:|---:|---:|
| hit@1 | 54.3% | 51.4% | −2.9 pp |
| hit@k | 74.3% | 74.3% | 0.0 pp |
| evidence Recall@k | **69.0%** | 66.2% | −2.8 pp |
| nDCG@k | **0.606** | 0.570 | −0.036 |
| MRR | **0.638** | 0.617 | −0.021 |
| all required evidence | **62.9%** | 57.1% | −5.7 pp |
| median retrieval latency | **77 ms** | 85 ms | +10.4% |
| complete command wall time | **8.23 s** | 11.09 s | +34.8% |
| approximate initial ingest/index time | **5 s** | 7 s | +40.0% |
| model blob | **621,875,917 B** | 639,150,858 B | +2.8% |

| 95% interval | EmbeddingGemma | Qwen3-Embedding 0.6B |
|---|---:|---:|
| evidence Recall@k | [0.544, 0.829] | [0.525, 0.797] |
| nDCG@k | [0.465, 0.740] | [0.444, 0.704] |
| MRR | [0.500, 0.778] | [0.482, 0.763] |
| all required evidence | [0.471, 0.788] | [0.412, 0.737] |
| abstention accuracy after calibration | [0.732, 0.951] | [0.732, 0.951] |

The negative-abstention rate is 0% in both raw runs because abstention was deliberately disabled.
Calibration recommends an RRF score threshold of `0.8333333333333333` for EmbeddingGemma
(balanced accuracy 75.7%) and `0.6666666666666666` for Qwen (77.1%). At EmbeddingGemma's exact
threshold, post-hoc policy scoring abstains on all 6 negatives and gives 58.5% overall abstention
accuracy across the imbalanced 35-positive/6-negative set. These values are not portable to
dense-only, sparse-only or cross-encoder scores.

## Immutable model identities

| Candidate | Ollama tag | Digest | Quantization | Dimension | Query prefix |
|---|---|---|---|---:|---|
| EmbeddingGemma | `embeddinggemma:latest` | `85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1` | BF16 | 768 | `task: search result \| query: ` |
| Qwen3-Embedding | `qwen3-embedding:0.6b` | `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d` | Q8_0 | 1,024 | `Instruct: Given a user query, retrieve relevant passages that answer the query\nQuery:` |

## Decision

Keep EmbeddingGemma as the default. The acceptance criterion required a candidate to be no worse
on the safety/retrieval metrics and clearly improve at least three relevant measures. Qwen improves
none here: it ties hit@k, is lower on the other quality metrics, and is slower. Qwen remains a
documented multilingual preset for operators to evaluate on their own corpus; this small synthetic
set is not evidence against its quality in other domains.
