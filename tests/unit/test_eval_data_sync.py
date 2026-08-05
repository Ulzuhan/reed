"""The evaluation dataset is committed twice; the copies must not drift.

`eval/` is what a repository checkout measures and reviews; the copy under
`src/reed/evals/data/` is what the installed wheel ships. If they diverge, CI
silently measures different data than users run against.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKOUT_DATA = REPO_ROOT / "eval"
PACKAGED_DATA = REPO_ROOT / "src" / "reed" / "evals" / "data"

SHARED_FILES = ("golden.jsonl", "evidence.json")


def _corpus_names(root: Path) -> set[str]:
    return {path.name for path in (root / "corpus").iterdir() if path.is_file()}


def test_both_copies_ship_the_same_corpus_files() -> None:
    assert _corpus_names(CHECKOUT_DATA) == _corpus_names(PACKAGED_DATA)


def test_both_copies_are_byte_identical() -> None:
    relative_paths = [Path(name) for name in SHARED_FILES]
    relative_paths += [Path("corpus") / name for name in sorted(_corpus_names(CHECKOUT_DATA))]

    diverged = [
        str(relative)
        for relative in relative_paths
        if (CHECKOUT_DATA / relative).read_bytes() != (PACKAGED_DATA / relative).read_bytes()
    ]
    assert not diverged, f"eval dataset copies have drifted: {diverged}"
