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
	@# umask 022: no group-writable files in .venv (nltk refuses such data directories).
	umask 022 && $(UV) sync --locked

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

# --- Ingestion (Phase 1, Step 4) ---------------------------------------------------------
# Read-only and offline: loads REPO_PATH (or --repo) and prints what was kept and why.
# Examples: make ingest INGEST_FLAGS="--list"; make ingest INGEST_FLAGS="--json --repo /path"
INGEST_FLAGS ?=
.PHONY: ingest
ingest: ## Load REPO_PATH with the Phase 1 filters and print the report (offline, read-only)
	$(UV_RUN) python -m src.ingestion.loader $(INGEST_FLAGS)

# --- Chunking (Phase 2, Step 6) --------------------------------------------------------
# Read-only and offline: loads REPO_PATH (or --repo), chunks it and prints the report.
# --list prints symbols and --show prints file content: run them in your own terminal.
# Examples: make chunk CHUNK_FLAGS="--json"; make chunk CHUNK_FLAGS="--show src/app.py"
CHUNK_FLAGS ?=
.PHONY: chunk
chunk: ## Chunk REPO_PATH with the Phase 2 strategies and print the report (offline, read-only)
	$(UV_RUN) python -m src.ingestion.chunker $(CHUNK_FLAGS)

# --- Vector index (Phase 3, Step 4) ----------------------------------------------------
# Loads and chunks REPO_PATH (or --repo), embeds with the local model (needs Ollama; loopback
# only, under the network guard) and atomically activates a new index in VECTOR_STORE_DIR.
# Unchanged nodes reuse their stored vectors. Prints counts, digests and paths, never code.
# Examples: make index; make index INDEX_FLAGS="--json"; make index INDEX_FLAGS="--repo /path"
INDEX_FLAGS ?=
.PHONY: index
index: ## Build or refresh the vector index of REPO_PATH (needs local Ollama; no search)
	$(UV_RUN) python -m src.indexing.build_index $(INDEX_FLAGS)

# Phase 2 (Step 7) adds its key test, tests/test_chunking_corpus_integration.py, to this target.
# --- Phase 1 key test (Step 7) ------------------------------------------------------------
# Only the corpus key test, not the Ollama integration tests: offline and read-only. It skips,
# with the reason, if REPO_PATH is missing or its HEAD is not the frozen commit.
.PHONY: test-corpus
test-corpus: ## Phase 1 key test on the frozen corpus at REPO_PATH (offline, read-only; no Ollama)
	$(UV_RUN) pytest -m integration -rs tests/test_ingestion_corpus_integration.py tests/test_chunking_corpus_integration.py

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
