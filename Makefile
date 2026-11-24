# Common developer commands for local-code-rag.
#
# Network policy: only `install` may reach the network (PyPI, first sync).
# Every other target runs uv with --offline, so it can never download anything
# behind your back; if .venv is missing or stale, it fails and tells you to run
# `make install`.

UV     ?= uv
UV_RUN := $(UV) run --locked --offline

.DEFAULT_GOAL := help
.PHONY: help install hooks lint format format-check typecheck test test-integration check

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Create/refresh .venv exactly from uv.lock (network: PyPI, only when packages are missing)
	$(UV) sync --locked

hooks: ## Install the git pre-commit hook (local ruff hooks, no network)
	$(UV_RUN) pre-commit install

lint: ## Lint with ruff
	$(UV_RUN) ruff check .

format: ## Format code and apply safe lint fixes
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

format-check: ## Fail if any file is not formatted
	$(UV_RUN) ruff format --check .

typecheck: ## Static type check (mypy --strict)
	$(UV_RUN) mypy

test: ## Unit tests (integration tests excluded)
	$(UV_RUN) pytest

test-integration: ## Integration tests (need local Ollama and the configured models)
	$(UV_RUN) pytest -m integration

check: lint format-check typecheck test ## Run all quality gates
