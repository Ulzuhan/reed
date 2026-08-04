# Evaluation — local + rerank

- Profile: `local` · chat `qwen3.5:4b` · embeddings `embeddinggemma`
- Retrieval: hybrid (dense + BM25), top_k=4, rerank=on
- Judge: not run
- Questions: 30 · median latency 11467 ms

## Retrieval

| hit@1 | hit@k | MRR | all expected docs |
|---|---|---|---|
| 81% | 96% | 0.872 | 88% |

## Generation

_Skipped: --retrieval-only_

## Weakest retrievals

**q-015** (factual) — I'm new. How soon could I realistically start carrying the pager?
- expected: 06-incident-response-runbook.md
- retrieved: 08-onboarding-faq.md, 07-engineering-standards.md, 05-pto-and-leave.md

**q-002** (factual) — Two teams just settled something between themselves on a call. Does that have to be recorded anywhere?
- expected: 01-welcome-and-mission.md
- retrieved: 08-onboarding-faq.md, 06-incident-response-runbook.md, 01-welcome-and-mission.md

**q-016** (factual) — The CSV export button has stopped working. People can still pull the same data through the API, and nothing has been lost. How fast do we need to move on it?
- expected: 06-incident-response-runbook.md
- retrieved: 07-engineering-standards.md, 03-security-and-data-handling.md, 06-incident-response-runbook.md

