"""Keep `.env.example` honest.

Documented settings that no longer exist, and real settings nobody documented,
are both silent papercuts for anyone deploying this.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import AliasChoices

from reed.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

ASSIGNMENT = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def env_var_for(field_name: str) -> str:
    """The variable that actually populates a setting.

    Most take the REED_ prefix; a field with a validation alias (OPENAI_API_KEY)
    keeps the conventional name instead.
    """
    alias = Settings.model_fields[field_name].validation_alias
    if isinstance(alias, AliasChoices):
        return str(alias.choices[0])
    if isinstance(alias, str):
        return alias
    return f"REED_{field_name.upper()}"


@pytest.fixture(scope="module")
def documented() -> set[str]:
    return set(ASSIGNMENT.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def actual() -> set[str]:
    return {env_var_for(name) for name in Settings.model_fields}


def test_every_documented_variable_is_real(documented: set[str], actual: set[str]) -> None:
    unknown = documented - actual
    assert not unknown, f".env.example documents settings that do not exist: {sorted(unknown)}"


def test_every_setting_is_documented(documented: set[str], actual: set[str]) -> None:
    missing = actual - documented
    assert not missing, f"settings missing from .env.example: {sorted(missing)}"


def test_the_example_does_not_ship_a_real_key() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "sk-" not in text, ".env.example looks like it contains a real API key"
