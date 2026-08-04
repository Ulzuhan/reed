# Model and component licenses

Reed does not redistribute model weights. Local models are downloaded separately by Ollama,
FastEmbed or Sentence Transformers, and each remains subject to its own terms. Operators are
responsible for reviewing the terms that apply to their use and model revision.

| Item | Reed role | Upstream terms | Source |
|---|---|---|---|
| EmbeddingGemma | Default local dense embeddings | Gemma Terms of Use and Prohibited Use Policy | [Google model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) · [terms](https://ai.google.dev/gemma/terms) |
| Qwen3-Embedding 0.6B | Optional multilingual dense preset | Apache-2.0 | [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| BAAI bge-reranker-v2-m3 | Optional multilingual reranker | Apache-2.0 | [BAAI model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| FastEmbed | BM25 sparse inference and optional reranker backend | Apache-2.0 | [Qdrant repository](https://github.com/qdrant/fastembed) |

Hosted OpenAI and Ollama usage is also governed by the service/runtime terms selected by the
operator. The Apache-2.0 license in Reed's root covers Reed's source and packaged artifacts, not
third-party services, models or user documents.
