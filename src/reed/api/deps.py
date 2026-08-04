"""Shared FastAPI dependencies."""

from __future__ import annotations

import contextlib
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from reed.config import Settings
from reed.services import Services


def get_settings(request: Request) -> Settings:
    services: Services = request.app.state.services
    return services.settings


def get_services(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Guard ``/v1`` routes when ``REED_API_KEY`` is configured.

    With no key configured Reed is open — the default assumes it runs on your
    own machine or behind your own network boundary.
    """
    if not settings.api_key:
        return
    # Compare the raw wire bytes. Starlette decodes headers as latin-1, so
    # round-tripping through it recovers exactly what the client sent — and
    # comparing bytes avoids compare_digest's TypeError on non-ASCII str, which
    # would turn a wrong key into a 500.
    #
    # A non-ASCII key arrives differently depending on the client: browsers,
    # curl and httpx send utf-8, but `requests` encodes str headers as latin-1.
    # Both spellings of the configured key are accepted, so no mainstream
    # client is locked out by its choice of encoding.
    supplied = (x_api_key or "").encode("latin-1", errors="replace")
    expected = [settings.api_key.encode("utf-8")]
    # A key with characters latin-1 cannot spell has utf-8 as its only wire form.
    with contextlib.suppress(UnicodeEncodeError):
        expected.append(settings.api_key.encode("latin-1"))

    matched = any(secrets.compare_digest(supplied, candidate) for candidate in expected)
    if not x_api_key or not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServicesDep = Annotated[Services, Depends(get_services)]
