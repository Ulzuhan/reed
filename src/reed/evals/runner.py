"""Run the evaluation suite.

Everything goes through the same ingestion and answering code the server uses,
against a throwaway embedded Qdrant. Nothing here touches the user's data
directory, and nothing depends on a server being up.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio.to_thread

from reed.config import Profile, RetrievalMode, Settings, get_settings
from reed.evals.dataset import (
    CORPUS_DIR,
    EVAL_DIR,
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
    parts: list[str] = [settings.profile]
    if settings.retrieval_mode != "hybrid":
        parts.append(settings.retrieval_mode)
    parts.append(f"k={settings.top_k}")
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
    retrieval_mode: RetrievalMode | None = None,
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
            retrieval_mode=retrieval_mode,
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
    retrieval_mode: RetrievalMode | None,
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
                "retrieval_mode": retrieval_mode or base.retrieval_mode,
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
        retrieval_mode=settings.retrieval_mode,
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

    from reed.rag.retriever import retrieve

    # Generation exercises the configured policy, including abstention. A
    # threshold-free query also preserves the complete score distribution for
    # comparable retrieval metrics and post-hoc calibration.
    raw_chunks = await anyio.to_thread.run_sync(
        lambda: retrieve(
            services,
            question.question,
            services.settings.top_k,
            apply_threshold=False,
        )
    )
    answered = await answer_question(services, question.question)
    outcome = score(question, raw_chunks, abstained=not answered.sources)

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

    evidence = _evidence_result_fields(question, raw_chunks, outcome, answered.text)
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
        expected_evidence=evidence.expected_evidence,
        covered_evidence=evidence.covered_evidence,
        recall_at_k=evidence.recall_at_k,
        ndcg_at_k=evidence.ndcg_at_k,
        retrieved_chunks=evidence.retrieved_chunks,
        abstained=evidence.abstained,
        citation_status=evidence.citation_status,
        warnings=evidence.warnings,
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
        lambda: retrieve(
            services,
            question.question,
            services.settings.top_k,
            apply_threshold=False,
        )
    )
    threshold = services.settings.resolved_min_evidence_score
    policy_abstained = not chunks or bool(threshold and chunks[0].score < threshold)
    outcome = score(question, chunks, abstained=policy_abstained)
    evidence = _evidence_result_fields(question, chunks, outcome, "")
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
            expected_evidence=evidence.expected_evidence,
            covered_evidence=evidence.covered_evidence,
            recall_at_k=evidence.recall_at_k,
            ndcg_at_k=evidence.ndcg_at_k,
            retrieved_chunks=evidence.retrieved_chunks,
            abstained=evidence.abstained,
            citation_status=evidence.citation_status,
            warnings=evidence.warnings,
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

    try:
        from reed.model_identity import resolve_embedding_identity

        identity = resolve_embedding_identity(settings)
        model_identity: dict[str, object] = {
            "digest": identity.digest,
            "revision": identity.revision,
            "quantization": identity.quantization,
        }
    except Exception as exc:  # noqa: BLE001 — partial provenance is still useful
        model_identity = {"warning": f"{type(exc).__name__}: {exc}"}

    return {
        "configuration": {
            "top_k": settings.top_k,
            "fetch_k": settings.fetch_k,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "dense_model": settings.embed_model_name,
            "retrieval_mode": settings.retrieval_mode,
            "sparse_model": settings.sparse_model,
            "query_prefix": settings.resolved_query_prefix,
            "document_prefix": settings.resolved_doc_prefix,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_context_chars": settings.max_context_chars,
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_output_tokens,
            "configured_min_evidence_score": settings.min_evidence_score,
            "resolved_min_evidence_score": settings.resolved_min_evidence_score,
        },
        "dataset": {
            "corpus_sha256": _corpus_digest(corpus),
            "golden_sha256": _file_digest(golden_path),
            "documents": [path.name for path in corpus],
        },
        "versions": versions,
        "model_identity": model_identity,
        "runtime": {
            "git_sha": _git_sha(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
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


_CITATION = re.compile(r"\[(\d+)]")


@dataclass(frozen=True, slots=True)
class _EvidenceResult:
    expected_evidence: list[str]
    covered_evidence: list[str]
    recall_at_k: float
    ndcg_at_k: float
    retrieved_chunks: list[dict[str, object]]
    abstained: bool
    citation_status: str
    warnings: list[str]


def _evidence_result_fields(
    question: GoldenQuestion,
    chunks: Sequence[object],
    outcome: RetrievalOutcome,
    answer: str,
) -> _EvidenceResult:
    from reed.evals.retrieval import matched_evidence

    exact_chunks: list[dict[str, object]] = []
    for rank, chunk in enumerate(chunks, start=1):
        exact_chunks.append(
            {
                "rank": rank,
                "text": str(getattr(chunk, "text", "")),
                "score": float(getattr(chunk, "score", 0.0)),
                "doc_id": str(getattr(chunk, "doc_id", "")),
                "filename": str(getattr(chunk, "filename", "")),
                "page": getattr(chunk, "page", None),
                "section": getattr(chunk, "section", None),
                "matched_evidence": matched_evidence(question, chunk),  # type: ignore[arg-type]
            }
        )
    citations = [int(value) for value in _CITATION.findall(answer)]
    citation_status = "not_checked"
    if answer:
        citation_status = (
            "valid"
            if citations and all(1 <= citation <= len(chunks) for citation in citations)
            else "missing_or_invalid"
        )
    warnings: list[str] = []
    if question.expected_docs and not question.evidence:
        warnings.append("document-level fallback: no exact evidence labels supplied")
    if answer and citation_status != "valid":
        warnings.append("answer has no valid source citation")
    return _EvidenceResult(
        expected_evidence=list(outcome.expected_evidence_ids),
        covered_evidence=list(outcome.covered_evidence_ids),
        recall_at_k=outcome.recall_at_k,
        ndcg_at_k=outcome.ndcg_at_k,
        retrieved_chunks=exact_chunks,
        abstained=outcome.abstained,
        citation_status=citation_status,
        warnings=warnings,
    )


def _git_sha() -> str:
    # Anchored to the dataset actually measured, not the process cwd: running
    # `reed eval` from inside an unrelated repository must not stamp that
    # repository's commit onto the report as provenance.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=EVAL_DIR.parent,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"
