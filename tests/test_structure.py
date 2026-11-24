"""Structure contract: every scaffolded module imports as ``src.<package>.<module>``.

Later phases import from these exact paths (e.g. ``from src.config import ...``),
so a renamed or missing module should fail here, not deep inside a later phase.
"""

import importlib

import pytest

SCAFFOLDED_MODULES: tuple[str, ...] = (
    "src",
    "src.config",
    "src.ingestion.loader",
    "src.ingestion.chunker",
    "src.indexing.embeddings",
    "src.indexing.vector_store",
    "src.retrieval.retriever",
    "src.retrieval.reranker",
    "src.generation.llm",
    "src.generation.prompt",
    "src.api.main",
    "src.api.schemas",
    "src.evaluation.ragas_eval",
)


@pytest.mark.parametrize("module_name", SCAFFOLDED_MODULES)
def test_scaffolded_module_is_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
