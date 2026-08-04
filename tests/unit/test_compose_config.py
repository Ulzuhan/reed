from __future__ import annotations

import re
from pathlib import Path

from pydantic import AliasChoices

from reed.config import Settings

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"
ENV_ENTRY = re.compile(r"^\s+-\s+([A-Z][A-Z0-9_]*)(?:=|\s*$)", re.MULTILINE)


def _env_name(field_name: str) -> str:
    alias = Settings.model_fields[field_name].validation_alias
    if isinstance(alias, AliasChoices):
        return str(alias.choices[0])
    if isinstance(alias, str):
        return alias
    return f"REED_{field_name.upper()}"


def test_compose_explicitly_passes_every_setting() -> None:
    configured = set(ENV_ENTRY.findall(COMPOSE.read_text(encoding="utf-8")))
    expected = {_env_name(name) for name in Settings.model_fields}

    assert expected <= configured, f"Compose is missing: {sorted(expected - configured)}"


def test_compose_keeps_automatic_embedding_prefixes_unset() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "- REED_EMBED_QUERY_PREFIX\n" in text
    assert "- REED_EMBED_DOC_PREFIX\n" in text
