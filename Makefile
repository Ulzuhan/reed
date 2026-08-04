.DEFAULT_GOAL := help
.PHONY: help install dev lint format type test test-unit test-coverage test-e2e eval eval-offline clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (including dev)
	uv sync

dev: ## Run the API with autoreload on http://localhost:8000
	uv run uvicorn reed.api.app:create_app --factory --reload --port 8000

lint: ## Check formatting and lint rules
	uv run ruff check .
	uv run ruff format --check .

format: ## Autoformat and autofix
	uv run ruff format .
	uv run ruff check --fix .

type: ## Type-check the package and tests
	uv run mypy src tests

test: ## Run the whole test suite
	uv run pytest

test-unit: ## Run only unit tests (no embedded Qdrant)
	uv run pytest tests/unit

test-coverage: ## Run the suite with the same 85% branch-coverage gate as CI
	uv run pytest --cov=reed --cov-branch --cov-report=term-missing:skip-covered --cov-fail-under=85

test-e2e: ## Run browser E2E against a Reed server already listening on localhost:8000
	uv run --group e2e pytest e2e --base-url http://127.0.0.1:8000

eval: ## Run the full evaluation suite (retrieval + LLM judge)
	uv run reed eval

eval-offline: ## Run retrieval metrics only (no LLM required)
	uv run reed eval --retrieval-only

clean: ## Remove caches and local runtime data
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
