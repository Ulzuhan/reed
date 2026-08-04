# Evaluation — local, hybrid, k=4

- Profile: `local` · chat `qwen3.5:4b` · embeddings `embeddinggemma`
- Retrieval: hybrid (dense + BM25), top_k=4, rerank=off
- Judge: `local:qwen3.5:4b`
- Questions: 30 · median latency 6603 ms

## Retrieval

| hit@1 | hit@k | MRR | all expected docs |
|---|---|---|---|
| 62% | 92% | 0.756 | 81% |

## Generation

| faithfulness | answer relevancy | correctness | context precision | context recall | correct refusals |
|---|---|---|---|---|---|
| 71% | 77% | 71% | 40% | 60% | 100% |

## Weakest retrievals

**q-002** (factual) — Two teams just settled something between themselves on a call. Does that have to be recorded anywhere?
- expected: 01-welcome-and-mission.md
- retrieved: 06-incident-response-runbook.md, 08-onboarding-faq.md

**q-015** (factual) — I'm new. How soon could I realistically start carrying the pager?
- expected: 06-incident-response-runbook.md
- retrieved: 08-onboarding-faq.md, 03-security-and-data-handling.md

**q-008** (factual) — My bag was taken on the train and my work machine was in it. What do I do?
- expected: 03-security-and-data-handling.md
- retrieved: 02-remote-work-policy.md, 04-expenses-and-travel.md, 03-security-and-data-handling.md

