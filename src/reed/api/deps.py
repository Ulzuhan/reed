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
    try:
        supplied = (x_api_key or "").encode("ascii")
    except UnicodeEncodeError:
        supplied = b""
    if not api_key_bytes_match(settings, supplied):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )


def api_key_bytes_match(settings: Settings, supplied: bytes | None) -> bool:
    """Constant-time comparison over the canonical ASCII wire form."""
    configured = settings.api_key.encode("ascii")
    candidate = supplied or b""
    return bool(candidate) and secrets.compare_digest(candidate, configured)


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServicesDep = Annotated[Services, Depends(get_services)]
