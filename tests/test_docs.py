"""Consistency guard for docs/decisions.md (no network, no Ollama).

Structure: the exact H2 headings exist. Anti-drift: the document names the default
models and context window taken from src/config.py, the single source of truth; the
DEFAULT_* constants do not depend on environment variables. Hygiene: no absolute user
paths. `TODO(user)` markers are allowed.
"""

import re
from pathlib import Path

import pytest

from src import config

DOC = Path(__file__).resolve().parents[1] / "docs" / "decisions.md"
REQUIRED_H2 = (
    "## 1. Use case",
    "## 2. Model choice and rationale",
    "## 3. The privacy argument",
    "## 4. Corpus selection",
    "## 5. Stack decisions",
    "## 6. Known gaps and open questions",
    "## 7. Phase 1 — Ingestion filtering (Steps 1\u20133)",  # \u2013: en dash (RUF001)
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
