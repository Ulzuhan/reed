"""Shared FastAPI dependencies."""

from __future__ import annotations

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
    # Settings restrict configured keys to ASCII, giving every HTTP client one
    # canonical wire representation without Unicode/latin-1 aliases.
    try:
        supplied = (x_api_key or "").encode("ascii")
        configured = settings.api_key.encode("ascii")
    except UnicodeEncodeError:
        supplied = b""
        configured = b"invalid-configured-key"
    if not supplied or not secrets.compare_digest(supplied, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )


def enforce_rate_limit(
    request: Request,
    services: Services,
    *,
    scope: str,
    limit: int,
) -> None:
    client = request.client.host if request.client else "unknown"
    if not services.rate_limiter.allow(scope, client, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {scope}",
            headers={"Retry-After": "60"},
        )


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServicesDep = Annotated[Services, Depends(get_services)]
