"""LLM-judged answer quality.

Written here rather than pulled from a library on purpose. RAGAS — the usual
choice — still depends on ``langchain-community`` and does not import against
the LangChain 1.x layout Reed is built on, so adopting it would mean pinning
the whole stack backwards. Implementing the four metrics directly costs a few
hundred lines, keeps the dependency tree honest, and makes what each number
actually measures readable instead of folkloric.

Every metric is one structured-output call, so a judge can be any model that
supports it — including a local one, which keeps the whole evaluation offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, Field

from reed.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from reed.evals.dataset import GoldenQuestion

logger = get_logger(__name__)

VerdictT = TypeVar("VerdictT", bound=BaseModel)

MAX_CONCURRENT_JUDGE_CALLS = 4

# Ratings come back on a 1-5 rubric because models are more consistent picking
# from a short scale than emitting a float.
RATING_MIN = 1
RATING_MAX = 5


def _normalise(rating: int) -> float:
    clamped = max(RATING_MIN, min(RATING_MAX, rating))
    return (clamped - RATING_MIN) / (RATING_MAX - RATING_MIN)


class Claim(BaseModel):
    text: str = Field(description="A single factual claim taken from the answer")
    supported: bool = Field(description="True if the excerpts directly support this claim")


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim] = Field(description="Every factual claim the answer makes")


class Rating(BaseModel):
    score: int = Field(ge=RATING_MIN, le=RATING_MAX, description="1 = poor, 5 = excellent")
    reason: str = Field(description="One sentence justifying the score")


class ChunkRelevance(BaseModel):
    index: int = Field(description="1-based excerpt number")
    relevant: bool = Field(description="True if this excerpt helps answer the question")


class PrecisionVerdict(BaseModel):
    excerpts: list[ChunkRelevance]


class Statement(BaseModel):
    text: str = Field(description="A single statement from the reference answer")
    attributable: bool = Field(description="True if the excerpts contain this statement")


class RecallVerdict(BaseModel):
    statements: list[Statement]


class RefusalVerdict(BaseModel):
    refused: bool = Field(
        description="True if the answer says the documents do not cover the question"
    )
    reason: str


@dataclass(frozen=True, slots=True)
class JudgeScores:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    correct_refusal: float | None = None
    notes: str = ""


class Judge:
    """Scores one answer at a time, caching verdicts on disk.

    The cache is keyed on the judge model plus the exact text being judged, so
    re-running an evaluation after changing only the retrieval settings still
    pays for the answers that actually changed.
    """

    def __init__(
        self, model: BaseChatModel, model_name: str, cache_dir: Path | None = None
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGE_CALLS)

    async def score(
        self,
        question: GoldenQuestion,
        answer: str,
        context_blocks: list[str],
    ) -> JudgeScores:
        cached = self._read_cache(question, answer, context_blocks)
        if cached is not None:
            return cached

        scores = await self._score_uncached(question, answer, context_blocks)
        self._write_cache(question, answer, context_blocks, scores)
        return scores

    async def _score_uncached(
        self,
        question: GoldenQuestion,
        answer: str,
        context_blocks: list[str],
    ) -> JudgeScores:
        if question.is_negative:
            verdict = await self._ask(
                RefusalVerdict,
                "Decide whether the answer admits that the documents do not cover the "
                "question, instead of inventing an answer.\n\n"
                f"Question: {question.question}\n\nAnswer: {answer}",
            )
            refused = 1.0 if (verdict and verdict.refused) else 0.0
            return JudgeScores(
                correct_refusal=refused,
                notes=verdict.reason if verdict else "judge call failed",
            )

        context = _numbered(context_blocks)
        faithfulness, relevancy, correctness, precision, recall = await asyncio.gather(
            self._faithfulness(answer, context),
            self._relevancy(question.question, answer),
            self._correctness(question.question, answer, question.reference_answer),
            self._precision(question.question, context_blocks),
            self._recall(question.reference_answer, context),
        )
        return JudgeScores(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            correctness=correctness,
            context_precision=precision,
            context_recall=recall,
        )

    async def _faithfulness(self, answer: str, context: str) -> float | None:
        """Share of the answer's claims the retrieved excerpts actually support."""
        verdict = await self._ask(
            FaithfulnessVerdict,
            "Break the answer into its individual factual claims. For each one, decide "
            "whether the excerpts directly support it. A claim that is merely plausible, "
            "or that relies on outside knowledge, is NOT supported.\n\n"
            f"Excerpts:\n{context}\n\nAnswer: {answer}",
        )
        if verdict is None or not verdict.claims:
            return None
        return sum(claim.supported for claim in verdict.claims) / len(verdict.claims)

    async def _relevancy(self, question: str, answer: str) -> float | None:
        """Does the answer address what was asked — regardless of correctness."""
        rating = await self._ask(
            Rating,
            "Rate how directly the answer addresses the question. Ignore whether it is "
            "factually right; judge only whether it answers what was asked. "
            "5 = answers precisely, 1 = evades or answers something else.\n\n"
            f"Question: {question}\n\nAnswer: {answer}",
        )
        return _normalise(rating.score) if rating else None

    async def _correctness(self, question: str, answer: str, reference: str) -> float | None:
        rating = await self._ask(
            Rating,
            "Compare the answer against the reference answer. 5 = every fact matches, "
            "3 = partially right or missing a key detail, 1 = contradicts the reference. "
            "Wording may differ; only the facts matter.\n\n"
            f"Question: {question}\n\nReference answer: {reference}\n\nAnswer: {answer}",
        )
        return _normalise(rating.score) if rating else None

    async def _precision(self, question: str, blocks: list[str]) -> float | None:
        """Share of retrieved excerpts that were actually worth retrieving."""
        if not blocks:
            return None
        verdict = await self._ask(
            PrecisionVerdict,
            "For each numbered excerpt, decide whether it contains information that "
            "helps answer the question. Judge every excerpt.\n\n"
            f"Question: {question}\n\nExcerpts:\n{_numbered(blocks)}",
        )
        if verdict is None or not verdict.excerpts:
            return None
        return sum(item.relevant for item in verdict.excerpts) / len(verdict.excerpts)

    async def _recall(self, reference: str, context: str) -> float | None:
        """Share of the reference answer that retrieval actually surfaced."""
        verdict = await self._ask(
            RecallVerdict,
            "Split the reference answer into individual statements. For each one, decide "
            "whether the excerpts contain it.\n\n"
            f"Excerpts:\n{context}\n\nReference answer: {reference}",
        )
        if verdict is None or not verdict.statements:
            return None
        return sum(s.attributable for s in verdict.statements) / len(verdict.statements)

    async def _ask(self, schema: type[VerdictT], prompt: str) -> VerdictT | None:
        """One structured-output call, tolerant of a model that fumbles the format."""
        async with self._semaphore:
            try:
                structured = self.model.with_structured_output(schema)
                return await structured.ainvoke(prompt)  # type: ignore[return-value]
            except Exception as exc:  # noqa: BLE001
                # A small local judge will occasionally emit invalid JSON. Drop
                # that one metric rather than losing the whole run.
                logger.warning("judge call failed (%s): %s", schema.__name__, exc)
                return None

    # -- cache ----------------------------------------------------------

    def _cache_path(
        self, question: GoldenQuestion, answer: str, context_blocks: list[str]
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            json.dumps(
                [self.model_name, question.id, question.reference_answer, answer, context_blocks],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(
        self, question: GoldenQuestion, answer: str, context_blocks: list[str]
    ) -> JudgeScores | None:
        path = self._cache_path(question, answer, context_blocks)
        if path is None or not path.exists():
            return None
        try:
            return JudgeScores(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None

    def _write_cache(
        self,
        question: GoldenQuestion,
        answer: str,
        context_blocks: list[str],
        scores: JudgeScores,
    ) -> None:
        path = self._cache_path(question, answer, context_blocks)
        if path is not None:
            path.write_text(json.dumps(asdict(scores)), encoding="utf-8")


def _numbered(blocks: list[str]) -> str:
    return "\n\n".join(f"[{n}] {block}" for n, block in enumerate(blocks, start=1))
