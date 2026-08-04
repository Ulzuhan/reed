from importlib.metadata import version

import reed


def test_version_is_exposed() -> None:
    assert reed.__version__ == version("reed")
