from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
CI_CALL = "uses: ./.github/workflows/ci.yml"
SCOPED_CI_CALL = re.compile(
    re.escape(CI_CALL) + r"\n\s+with:\n\s+concurrency-scope: (\S+)",
)
CONCURRENCY_GROUP = re.compile(r"^concurrency:\n  group: (.+)$", re.MULTILINE)


def _group(workflow: str) -> str:
    match = CONCURRENCY_GROUP.search((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    assert match is not None, f"{workflow} declares no concurrency group"
    return match.group(1)


def test_every_workflow_declares_a_distinct_concurrency_group() -> None:
    groups = [_group(workflow.name) for workflow in sorted(WORKFLOWS.glob("*.yml"))]

    assert len(set(groups)) == len(groups), f"workflows share a group: {groups}"


def test_ci_scopes_its_concurrency_group_per_caller() -> None:
    # Release and Docker both call CI on the same tag push. A group built only
    # from github.ref is identical for both, because the github context inside
    # a called workflow belongs to the caller, and one cancels the other's
    # quality gate seconds after it starts.
    assert "inputs.concurrency-scope" in _group("ci.yml")


def test_every_ci_caller_passes_a_distinct_scope() -> None:
    calls = 0
    scopes: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        calls += text.count(CI_CALL)
        scopes.extend(SCOPED_CI_CALL.findall(text))

    assert calls >= 2, "expected Release and Docker to reuse the CI quality gate"
    assert len(scopes) == calls, "a caller invokes CI without a concurrency scope"
    assert len(set(scopes)) == calls, f"CI callers share a scope: {sorted(scopes)}"
