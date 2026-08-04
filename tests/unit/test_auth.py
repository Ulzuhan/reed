from __future__ import annotations

import pytest
from fastapi import HTTPException

from reed.api.deps import require_api_key
from reed.config import Settings


def make(api_key: str) -> Settings:
    return Settings(_env_file=None, profile="fake", api_key=api_key)  # type: ignore[call-arg]


def test_no_configured_key_means_no_auth() -> None:
    require_api_key(make(""), x_api_key=None)


def test_configured_key_rejects_a_missing_header() -> None:
    with pytest.raises(HTTPException) as exc:
        require_api_key(make("s3cret"), x_api_key=None)
    assert exc.value.status_code == 401


def test_configured_key_rejects_a_wrong_header() -> None:
    with pytest.raises(HTTPException):
        require_api_key(make("s3cret"), x_api_key="wrong")


def test_configured_key_accepts_the_right_header() -> None:
    require_api_key(make("s3cret"), x_api_key="s3cret")


def test_a_non_ascii_supplied_key_is_rejected_not_crashed() -> None:
    # compare_digest raises TypeError on non-ASCII str, which would turn a
    # wrong key into a 500.
    with pytest.raises(HTTPException) as exc:
        require_api_key(make("s3cret"), x_api_key="wrông")
    assert exc.value.status_code == 401


def test_a_non_ascii_configured_key_still_works_over_the_wire() -> None:
    # Starlette decodes header bytes as latin-1, so this is what the endpoint
    # actually receives when a client sends "contraseña" as utf-8.
    as_received = "contraseña".encode().decode("latin-1")

    require_api_key(make("contraseña"), x_api_key=as_received)

    with pytest.raises(HTTPException):
        require_api_key(make("contraseña"), x_api_key="contrasena")
