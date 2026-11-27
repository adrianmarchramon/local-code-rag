"""Consistency guard for docs/decisions.md (no network, no Ollama).

Structure: the exact H2 headings exist. Anti-drift: the document names the default
models and context window taken from src/config.py, the single source of truth; the
DEFAULT_* constants do not depend on environment variables. Hygiene: no absolute user
paths. `TODO(user)` markers are allowed. Phase 1: every discard reason, every language label,
the frozen corpus commit and the expected totals are named. Phase 2: the installed tree-sitter
versions, every span kind and the chunk cap are named; Steps 3-6: every strategy, every
fallback reason and the node-ID namespace are named; Step 7: the chunk manifest's path, corpus
digest and totals are named, read from the manifest itself. Phase 3: the embedding recipe
(prefixes, verified digests, fingerprint) and the pinned Chroma versions are named.
"""

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest
from chunk_manifest import MANIFEST_PATH as CHUNK_MANIFEST_PATH
from corpus_manifest import FROZEN_CORPUS_SHA

from src import config
from src.ingestion.chunker import DEFAULT_MAX_CHUNK_CHARS
from src.ingestion.nodes import NODE_ID_NAMESPACE
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules
from src.ingestion.spans import SpanKind
from src.ingestion.strategies import FallbackReason, Strategy

DOC = Path(__file__).resolve().parents[1] / "docs" / "decisions.md"
REQUIRED_H2 = (
    "## 1. Use case",
    "## 2. Model choice and rationale",
    "## 3. The privacy argument",
    "## 4. Corpus selection",
    "## 5. Stack decisions",
    "## 6. Known gaps and open questions",
    "## 7. Phase 1 — Ingestion filtering (Steps 1\u20133)",  # \u2013: en dash (RUF001)
    "## 8. Phase 1 — Loading and metadata (Steps 4\u20135)",  # \u2013: en dash (RUF001)
    "## 9. Phase 1 — Languages, tests and closure (Steps 6\u20137)",  # \u2013: en dash
    "## 10. Phase 2 — Syntax-aware chunking (Steps 1\u20132)",  # \u2013: en dash
    "## 11. Phase 2 — Metadata, fallbacks and the chunker (Steps 3\u20136)",  # \u2013: en dash
    "## 12. Phase 2 — Step 7 and closure",
    "## 13. Phase 3 — Embeddings and indexing (Steps 1\u20134)",  # \u2013: en dash
)
USER_PATH = re.compile(r"/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\", re.IGNORECASE)


def _doc_lines() -> list[str]:
    return DOC.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("heading", REQUIRED_H2)
def test_required_h2_heading_is_present(heading: str) -> None:
    assert heading in _doc_lines()


@pytest.mark.parametrize(
    "default",
    [
        config.DEFAULT_CODE_MODEL,
        config.CODE_BASE_MODEL,
        config.DEFAULT_EMBEDDING_MODEL,
        str(config.DEFAULT_LLM_CONTEXT_WINDOW),
    ],
)
def test_document_names_the_configured_defaults(default: str) -> None:
    assert default in "\n".join(_doc_lines()), f"docs/decisions.md does not mention {default!r}"


def test_document_contains_no_absolute_user_paths() -> None:
    assert [m for line in _doc_lines() for m in USER_PATH.findall(line)] == []


@pytest.mark.parametrize("reason", [reason.value for reason in DiscardReason])
def test_document_names_every_discard_reason(reason: str) -> None:
    assert f"`{reason}`" in "\n".join(_doc_lines()), (
        f"docs/decisions.md does not explain {reason!r}"
    )


@pytest.mark.parametrize("label", sorted(IngestionRules().languages))
def test_document_names_every_language_label(label: str) -> None:
    assert f"`{label}`" in "\n".join(_doc_lines()), f"docs/decisions.md does not list {label!r}"


def test_document_names_the_frozen_corpus_and_the_expected_totals() -> None:
    text = "\n".join(_doc_lines())

    assert FROZEN_CORPUS_SHA in text
    assert "88 files / 491,997 B" in text


@pytest.mark.parametrize("package", ["tree-sitter", "tree-sitter-python"])
def test_document_names_the_installed_grammar_versions(package: str) -> None:
    pinned = f"`{package}` {version(package)}"
    assert pinned in "\n".join(_doc_lines()), f"docs/decisions.md does not mention {pinned}"


@pytest.mark.parametrize("kind", [kind.value for kind in SpanKind])
def test_document_names_every_span_kind(kind: str) -> None:
    assert f"`{kind}`" in "\n".join(_doc_lines()), f"docs/decisions.md does not explain {kind!r}"


def test_document_names_the_chunk_cap() -> None:
    assert f"cap {DEFAULT_MAX_CHUNK_CHARS} characters" in "\n".join(_doc_lines())


@pytest.mark.parametrize("code", [s.value for s in Strategy] + [r.value for r in FallbackReason])
def test_document_names_every_strategy_and_fallback_reason(code: str) -> None:
    assert f"`{code}`" in "\n".join(_doc_lines()), f"docs/decisions.md does not explain {code!r}"


def test_document_names_the_node_id_namespace() -> None:
    assert str(NODE_ID_NAMESPACE) in "\n".join(_doc_lines())


def test_document_names_the_chunk_manifest_and_its_figures() -> None:
    text = "\n".join(_doc_lines())
    manifest = json.loads(CHUNK_MANIFEST_PATH.read_text(encoding="utf-8"))
    report = manifest["report"]

    assert "`tests/fixtures/chunk_manifest.json`" in text
    assert manifest["corpus"]["commit"] in text
    assert manifest["digest"]["corpus"] in text
    assert f"{report['documents']} \u2192 {report['nodes']} nodes" in text  # \u2192: arrow
    assert f"{report['forced']} forced" in text


def test_document_names_the_embedding_recipe_and_the_pinned_store_versions() -> None:
    from src.indexing.embeddings import recipe_for

    text = "\n".join(_doc_lines())
    recipe = recipe_for(config.DEFAULT_EMBEDDING_MODEL)

    assert recipe.fingerprint() in text
    assert all(digest in text for digest in recipe.verified_digests)
    assert f"`{recipe.document_prefix}`" in text
    assert f"`{recipe.query_prefix}`" in text
    for package in ("chromadb", "llama-index-vector-stores-chroma"):
        assert f"`{package}=={version(package)}`" in text
