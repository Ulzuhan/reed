"""Run the evaluation suite.

Everything goes through the same ingestion and answering code the server uses,
against a throwaway embedded Qdrant. Nothing here touches the user's data
directory, and nothing depends on a server being up.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from reed.config import Profile, Settings, get_settings
from reed.evals.dataset import CORPUS_DIR, RESULTS_DIR, GoldenQuestion, corpus_files, load_golden
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
    judge_profile: str | None = None,
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
    judge_profile: str | None,
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
        run_settings = base.model_copy(
            update={
                "data_dir": Path(scratch),
                "collection": "reed_eval",
                "top_k": top_k or base.top_k,
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
    settings: Settings, judge_profile: str | None, results_dir: Path | None
) -> Judge | None:
    from reed.providers import build_chat_model

    profile: Profile = judge_profile or settings.eval_judge_profile  # type: ignore[assignment]
    if profile == "openai" and not settings.openai_api_key:
        return None

    judge_settings = settings
    if settings.eval_judge_model:
        field = "openai_chat_model" if profile == "openai" else "ollama_chat_model"
        judge_settings = settings.model_copy(update={field: settings.eval_judge_model})

    model = build_chat_model(judge_settings, profile=profile)
    name = (
        judge_settings.openai_chat_model
        if profile == "openai"
        else judge_settings.ollama_chat_model
    )
    cache = (results_dir or RESULTS_DIR) / "cache"
    return Judge(model=model, model_name=f"{profile}:{name}", cache_dir=cache)


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
) -> Report:
    settings = services.settings
    outcomes: list[RetrievalOutcome] = []
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, start=1):
        logger.info("[%d/%d] %s", index, len(questions), question.id)
        result, outcome = await _evaluate_one(services, question, judge, skipped_reason)
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
    )


async def _evaluate_one(
    services: Services,
    question: GoldenQuestion,
    judge: Judge | None,
    skipped_reason: str | None,
) -> tuple[QuestionResult, RetrievalOutcome]:
    from reed.evals.retrieval import score

    answered = await answer_question(services, question.question)
    outcome = score(question, answered.sources)

    scores = JudgeScores()
    if judge is not None and skipped_reason is None:
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
    )
    return result, outcome
