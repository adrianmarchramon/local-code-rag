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

# --- Local inference (Phase 0, Step 4) ---------------------------------------------------
.PHONY: verify-local
verify-local: ## Verify local generation + embeddings under the network guard (needs Ollama + models)
	$(UV_RUN) python -m src.health_check

# --- Model provisioning (Phase 0, Steps 3 and 5) -----------------------------------------
# make cannot import src/config.py, so these names repeat it on purpose and
# tests/test_model_provisioning.py fails if they drift. The PROVISION_ prefix avoids
# clashing with the CODE_MODEL / EMBEDDING_MODEL environment overrides read by config.py.
OLLAMA_CLI          ?= ollama
PROVISION_CODE_BASE := qwen2.5-coder:7b
PROVISION_CODE_RAG  := qwen2.5-coder-7b-rag
PROVISION_EMBEDDING := nomic-embed-text
PROVISION_MODELFILE := ollama/Modelfile

.PHONY: pull-models models
pull-models: ## PROVISIONING, uses the network (Ollama registry): pull the base models that are missing
	@for model in $(PROVISION_CODE_BASE) $(PROVISION_EMBEDDING); do \
		if $(OLLAMA_CLI) show "$$model" >/dev/null 2>&1; then \
			echo "present, skipped: $$model"; \
		else \
			echo "pulling: $$model"; $(OLLAMA_CLI) pull "$$model" || exit 1; \
		fi; \
	done

# Re-running is safe: an unchanged Modelfile yields the same model. After editing
# ollama/Modelfile, run `make models` again to rebuild the derived model.
models: ## Create the derived code model from ollama/Modelfile (no network; needs the base model)
	@$(OLLAMA_CLI) show $(PROVISION_CODE_BASE) >/dev/null 2>&1 || { \
		echo "error: base model $(PROVISION_CODE_BASE) is not local; run 'make pull-models' first (provisioning, uses the network)" >&2; \
		exit 1; }
	$(OLLAMA_CLI) create $(PROVISION_CODE_RAG) -f $(PROVISION_MODELFILE)
	@$(OLLAMA_CLI) show $(PROVISION_CODE_RAG) --parameters
