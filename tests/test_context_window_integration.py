"""Integration: the configured code model really runs with LLM_CONTEXT_WINDOW tokens.

Loads CODE_MODEL with a tiny prompt and no per-request ``num_ctx`` (so the Modelfile's
``PARAMETER num_ctx`` must apply), reads the context length the server actually allocated
from ``/api/ps``, then unloads the model. One model load, so it stays fast; the full needle
(truncation) test was a documented one-off because it takes ~15 min on CPU.

Excluded from `make test`; run with `make test-integration`.
"""

import pytest

from src.config import CODE_MODEL, LLM_CONTEXT_WINDOW, OLLAMA_BASE_URL, REQUEST_TIMEOUT
from src.model_names import normalize_model_name
from src.ollama_client import OllamaHttpClient

pytestmark = pytest.mark.integration


def test_code_model_runs_with_the_configured_context_window() -> None:
    client = OllamaHttpClient(OLLAMA_BASE_URL, REQUEST_TIMEOUT)
    try:
        client.generate(
            CODE_MODEL, "Reply with OK.", {"temperature": 0, "num_predict": 1}, keep_alive="60s"
        )
        loaded = {normalize_model_name(m.name): m for m in client.running_models()}
    finally:
        client.generate(CODE_MODEL, "", {}, keep_alive=0)  # empty prompt + 0 = unload now

    model = loaded.get(normalize_model_name(CODE_MODEL))
    assert model is not None, f"{CODE_MODEL} was not loaded; loaded: {sorted(loaded)}"
    assert model.context_length == LLM_CONTEXT_WINDOW
