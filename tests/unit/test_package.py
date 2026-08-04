from importlib.metadata import version
from pathlib import Path

import reed
from reed.evals.dataset import corpus_files, load_golden


def test_version_is_exposed() -> None:
    assert reed.__version__ == version("reed")


def test_evaluation_data_lives_inside_the_package() -> None:
    questions = load_golden(Path(reed.__file__).parent / "evals" / "data" / "golden.jsonl")
    corpus = corpus_files(Path(reed.__file__).parent / "evals" / "data" / "corpus")

    assert len(questions) == 41
    assert len(corpus) == 10
    assert all(question.evidence for question in questions if not question.is_negative)
