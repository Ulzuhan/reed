"""Wiring: one container holding the objects the API, CLI and evaluator share.

Models are built lazily so that starting the server never blocks on a network
call, and so the `fake` profile stays instant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reed.config import Settings, get_settings
from reed.log import get_logger, setup_logging

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

logger = get_logger(__name__)


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chat: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None

    @property
    def chat(self) -> BaseChatModel:
        if self._chat is None:
            from reed.providers import build_chat_model

            self._chat = build_chat_model(self.settings)
            logger.info("chat model ready: %s", self.settings.chat_model_name)
        return self._chat

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from reed.providers import build_embeddings

            self._embeddings = build_embeddings(self.settings)
            logger.info("embedding model ready: %s", self.settings.embed_model_name)
        return self._embeddings

    def close(self) -> None:
        """Release resources held by long-lived clients."""


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    settings.validate_ready()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return Services(settings)
