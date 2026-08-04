"""Run the evaluation suite.

Everything goes through the same ingestion and answering code the server uses,
against a throwaway embedded Qdrant. Nothing here touches the user's data
directory, and nothing depends on a server being up.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import platform
import tempfile
import time
from pathlib import Path

import anyio.to_thread

from reed.config import Profile, Settings, get_settings
from reed.evals.dataset import (
    CORPUS_DIR,
    GOLDEN_PATH,
    RESULTS_DIR,
    GoldenQuestion,
    corpus_files,
    load_golden,
)
from reed.evals.judge import Judge, JudgeScores
from reed.evals.report import QuestionResult, Report, build_report
from reed.evals.retrieval import RetrievalOutcome
from reed.log import get_logger
from reed.rag.chain import answer as answer_question
from reed.services import Services, build_services

logger = get_logger(__name__)


def default_label(settings: Settings) -> str:
    parts = [settings.profile, f"k={settings.top_k}"]
    if settings.rerank_enabled:
        parts.append("rerank")
    return ", ".join(parts)


def run_evaluation(
    *,
    retrieval_only: bool = False,
    top_k: int | None = None,
    judge_profile: Profile | None = None,
    label: str | None = None,
    corpus_dir: Path | None = None,
    golden_path: Path | None = None,
    results_dir: Path | None = None,
    settings: Settings | None = None,
) -> Report:
    return asyncio.run(
        _run(
            retrieval_only=retrieval_only,
            top_k=top_k,
            judge_profile=judge_profile,
            label=label,
            corpus_dir=corpus_dir,
            golden_path=golden_path,
            results_dir=results_dir,
            settings=settings,
        )
    )


async def _run(
    *,
    retrieval_only: bool,
    top_k: int | None,
    judge_profile: Profile | None,
    label: str | None,
    corpus_dir: Path | None,
    golden_path: Path | None,
    results_dir: Path | None,
    settings: Settings | None,
) -> Report:
    base = settings or get_settings()
    questions = load_golden(golden_path)
    corpus = corpus_files(corpus_dir or CORPUS_DIR)
    if not corpus:
        raise FileNotFoundError(f"No corpus documents found in {corpus_dir or CORPUS_DIR}")

    with tempfile.TemporaryDirectory(prefix="reed-eval-") as scratch:
        # A throwaway embedded Qdrant: an evaluation must never read from, or
        # write into, whatever the user has ingested for real.
        # model_copy(update=...) deliberately skips Pydantic validation. Rebuild
        # the settings so CLI overrides cannot smuggle a negative or huge top_k.
        run_settings = Settings.model_validate(
            {
                **base.model_dump(),
                "data_dir": Path(scratch),
                "collection": "reed_eval",
                "top_k": base.top_k if top_k is None else top_k,
            }
        )
        services = build_services(run_settings)
        try:
            _ingest_corpus(services, corpus)
            judge = (
                None if retrieval_only else _build_judge(run_settings, judge_profile, results_dir)
            )
            skipped = _skip_reason(retrieval_only, judge)
            return await _evaluate(
                services,
                questions,
                judge=judge,
                label=label or default_label(run_settings),
                skipped_reason=skipped,
                retrieval_only=retrieval_only,
                provenance=_provenance(
                    run_settings,
                    corpus,
                    golden_path or GOLDEN_PATH,
                ),
            )
        finally:
            services.close()


def _ingest_corpus(services: Services, corpus: list[Path]) -> None:
    from reed.ingest.pipeline import ingest_path

    logger.info("ingesting %d corpus documents", len(corpus))
    for path in corpus:
        result = ingest_path(services, path)
        if result.status == "error":
            raise RuntimeError(f"Could not ingest {path.name}: {result.error}")


def _build_judge(
    settings: Settings, judge_profile: Profile | None, results_dir: Path | None
) -> Judge | None:
    from reed.providers import build_chat_model

    profile = judge_profile or settings.eval_judge_profile
    if profile == "openai" and not settings.openai_api_key:
        return None

    judge_settings = settings
    if settings.eval_judge_model:
        field = "openai_chat_model" if profile == "openai" else "ollama_chat_model"
        judge_settings = settings.model_copy(update={field: settings.eval_judge_model})

    model = build_chat_model(judge_settings, profile=profile)
    if profile == "openai":
        name = judge_settings.openai_chat_model
    elif profile == "local":
        name = judge_settings.ollama_chat_model
    else:
        name = "fake-chat"
    cache = (results_dir or RESULTS_DIR) / "cache"
    fingerprint = {
        "profile": profile,
        "model": name,
        "temperature": judge_settings.temperature,
        "max_output_tokens": judge_settings.max_output_tokens,
        "base_url": (
            judge_settings.ollama_base_url
            if profile == "local"
            else "openai-default"
            if profile == "openai"
            else "in-process"
        ),
    }
    return Judge(
        model=model,
        model_name=f"{profile}:{name}:{_json_fingerprint(fingerprint)}",
        cache_dir=cache,
    )


def _skip_reason(retrieval_only: bool, judge: Judge | None) -> str | None:
    if retrieval_only:
        return "--retrieval-only"
    if judge is None:
        return (
            "no judge available — set OPENAI_API_KEY, or pass --judge local to "
            "score with your own Ollama model"
        )
    return None


async def _evaluate(
    services: Services,
    questions: list[GoldenQuestion],
    *,
    judge: Judge | None,
    label: str,
    skipped_reason: str | None,
    retrieval_only: bool,
    provenance: dict[str, object],
) -> Report:
    settings = services.settings
    outcomes: list[RetrievalOutcome] = []
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, start=1):
        logger.info("[%d/%d] %s", index, len(questions), question.id)
        result, outcome = await _evaluate_one(
            services, question, judge, skipped_reason, retrieval_only
        )
        results.append(result)
        outcomes.append(outcome)

    judge_model = judge.model_name if judge and not skipped_reason else None
    return build_report(
        label=label,
        profile=settings.profile,
        chat_model=settings.chat_model_name,
        embed_model=settings.embed_model_name,
        judge_model=judge_model,
        top_k=settings.top_k,
        rerank=settings.rerank_enabled,
        outcomes=outcomes,
        results=results,
        skipped_reason=skipped_reason,
        provenance=provenance,
    )


async def _evaluate_one(
    services: Services,
    question: GoldenQuestion,
    judge: Judge | None,
    skipped_reason: str | None,
    retrieval_only: bool,
) -> tuple[QuestionResult, RetrievalOutcome]:
    from reed.evals.retrieval import score

    if retrieval_only:
        # No generation at all: retrieval metrics only need the query embedded
        # and searched, so the chat model is never touched.
        return await _retrieve_only(services, question)

    answered = await answer_question(services, question.question)
    outcome = score(question, answered.sources)

    if answered.error:
        # A dead provider would otherwise be recorded as a low quality score.
        logger.warning("%s failed: %s", question.id, answered.error)

    scores = JudgeScores()
    if judge is not None and skipped_reason is None and not answered.error:
        scores = await judge.score(
            question,
            answered.text,
            [chunk.text for chunk in answered.sources],
        )

    result = QuestionResult(
        id=question.id,
        type=question.type,
        question=question.question,
        answer=answered.text,
        expected_docs=list(question.expected_docs),
        retrieved_docs=list(outcome.retrieved_docs),
        reciprocal_rank=outcome.reciprocal_rank,
        latency_ms=answered.latency_ms,
        judge=scores,
        error=answered.error,
    )
    return result, outcome


async def _retrieve_only(
    services: Services, question: GoldenQuestion
) -> tuple[QuestionResult, RetrievalOutcome]:
    """Score retrieval without constructing or touching the chat model."""
    from reed.evals.retrieval import score
    from reed.rag.retriever import retrieve

    started = time.perf_counter()
    chunks = await anyio.to_thread.run_sync(
        lambda: retrieve(services, question.question, services.settings.top_k)
    )
    outcome = score(question, chunks)
    return (
        QuestionResult(
            id=question.id,
            type=question.type,
            question=question.question,
            answer="",
            expected_docs=list(question.expected_docs),
            retrieved_docs=list(outcome.retrieved_docs),
            reciprocal_rank=outcome.reciprocal_rank,
            latency_ms=int((time.perf_counter() - started) * 1000),
            judge=JudgeScores(),
        ),
        outcome,
    )


def _provenance(
    settings: Settings,
    corpus: list[Path],
    golden_path: Path,
) -> dict[str, object]:
    """Everything needed to explain and reproduce an evaluation run."""
    package_names = (
        "reed",
        "langchain-core",
        "langchain-qdrant",
        "qdrant-client",
        "fastembed",
    )
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in package_names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    return {
        "configuration": {
            "top_k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "dense_model": settings.embed_model_name,
            "sparse_model": settings.sparse_model,
            "query_prefix": settings.resolved_query_prefix,
            "document_prefix": settings.resolved_doc_prefix,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_context_chars": settings.max_context_chars,
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
        },
        "dataset": {
            "corpus_sha256": _corpus_digest(corpus),
            "golden_sha256": _file_digest(golden_path),
            "documents": [path.name for path in corpus],
        },
        "versions": versions,
    }


def _corpus_digest(corpus: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(corpus, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_fingerprint(values: dict[str, object]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
