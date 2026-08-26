"""A caller-supplied name is data. It must not be able to pick the directory."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

import pytest

from reed.ingest.pipeline import safe_filename

# Shapes that try to climb out of the uploads directory, in both separator
# styles. The backslash ones are the point: `Path(...).name` on POSIX leaves
# them whole, and the result is later joined onto a directory.
HOSTILE = [
    "../../etc/passwd.txt",
    "..\\..\\etc\\passwd.txt",
    "../..\\mixed/separators\\policy.md",
    "/absolute/path.md",
    "\\\\server\\share\\policy.md",
    "C:\\Windows\\System32\\config.md",
    "C:policy.md",
    "..",
    ".",
    "....//....//passwd.txt",
]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_no_separator_of_either_flavour_survives(hostile: str) -> None:
    result = safe_filename(hostile)

    assert "/" not in result
    assert "\\" not in result
    assert result not in {".", ".."}


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize("flavour", [PurePosixPath, PureWindowsPath])
def test_the_stored_path_stays_in_the_uploads_directory(
    hostile: str, flavour: type[PurePosixPath] | type[PureWindowsPath]
) -> None:
    """Checked under both path flavours, on whichever host runs the test.

    The bug was never reachable on POSIX and was never going to be caught by a
    POSIX-only assertion; modelling the Windows join explicitly is what makes
    this test mean something on the CI that runs it.
    """
    uploads = flavour("/data/uploads")

    composed = uploads / f"d-0123456789abcdef__{safe_filename(hostile)}"

    assert composed.parent == uploads


def test_a_posix_name_is_reduced_the_same_way_a_windows_one_is() -> None:
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert safe_filename("..\\..\\etc\\passwd.txt") == "passwd.txt"


def test_names_with_nothing_printable_left_fall_back() -> None:
    assert safe_filename("") == "upload"
    assert safe_filename("\x00\x01\x02") == "upload"
    assert safe_filename("..") == "upload"
    assert safe_filename("/") == "upload"


def test_ordinary_names_are_left_alone() -> None:
    assert safe_filename("Q3 expenses (final).pdf") == "Q3 expenses (final).pdf"
    assert safe_filename("informe-año.md") == "informe-año.md"


def test_overlong_names_are_capped_but_keep_their_suffix() -> None:
    capped = safe_filename("a" * 300 + ".pdf")

    assert len(capped) == 255
    assert capped.endswith(".pdf")
