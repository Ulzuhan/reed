import reed


def test_version_is_exposed() -> None:
    assert reed.__version__.count(".") == 2
