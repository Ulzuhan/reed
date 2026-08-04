"""Browser-level regression for the complete upload → ask → cite flow."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def test_user_can_upload_ask_and_open_a_safe_citation(page: Page, base_url: str) -> None:
    attack = '<img src=x onerror="window.reedPwned=true">'
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    page.goto(base_url)

    page.set_input_files(
        "#file-input",
        files={
            "name": "browser-policy.md",
            "mimeType": "text/markdown",
            "buffer": (
                "# Browser policy\n\nExpenses above 75 euros require approval.\n\n" + attack
            ).encode(),
        },
    )

    expect(page.locator(".doc-name")).to_have_text("browser-policy.md")
    expect(page.locator(".doc-meta")).to_contain_text("chunks", timeout=30_000)

    page.locator("#question").fill("What is the expense threshold?")
    page.locator("#send").click()

    answer = page.locator(".message.assistant .bubble")
    expect(answer).to_contain_text("canned answer", timeout=30_000)
    expect(page.locator(".message.assistant .cite")).to_have_count(1)
    expect(page.locator(".message.assistant .source")).to_have_count(1)

    page.locator(".message.assistant .source summary").click()
    expect(page.locator(".message.assistant .source-excerpt")).to_contain_text(attack)
    assert page.locator(".message.assistant .source img").count() == 0
    assert page.evaluate("window.reedPwned") is None
    assert console_errors == []
