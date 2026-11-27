"""Phase 3, Step 7: the indexing tests the phase document asks for, made honest.

Hermetic part (fake embedder, real Chroma): build, persist, load in a new process, search
among distractors with neutral file names; zero document embeddings on load; drift; the
language filter; the literal Phase 4-5 layout plus ``load_index(vector_store=store)``. The
fake embedder tests the plumbing, not semantics, and a deliberately broken embedder proves
the relevance assertion can fail.

Integration part (``@pytest.mark.integration``, real ``nomic-embed-text`` through Ollama on
loopback, no ``skip``: if Ollama is missing, it fails): "how are input data validated?"
must rank the input check first among four distractors, although the check never uses
"validate" (or any inflection) in its path, symbol or body; the Spanish variant's rank is
recorded. Run it with ``make test-integration``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
from fake_embedding import FakeEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import PrivateAttr

from src.errors import StaleIndexError
from src.indexing.build_index import build_index, build_index_with_report, load_index
from src.indexing.embeddings import EmbedderIdentity
from src.indexing.search import search
from src.indexing.vector_store import ChromaIndexStore, get_vector_store
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_repository

REPO_ROOT = Path(__file__).resolve().parents[1]

# Neutral names: nothing in a path or symbol gives the answer away. The target checks its
# input without saying "validate" anywhere.
TARGET = (
    "def ensure_payload(payload):\n"
    '    """Reject a payload that is missing fields or has the wrong types."""\n'
    "    if not isinstance(payload, dict):\n"
    "        raise TypeError('payload must be a mapping')\n"
    "    for field in ('name', 'age'):\n"
    "        if field not in payload:\n"
    "            raise ValueError(f'missing field: {field}')\n"
    "    if not isinstance(payload['age'], int) or payload['age'] < 0:\n"
    "        raise ValueError('age must be a non-negative integer')\n"
    "    return payload\n"
)
DISTRACTORS = {
    "pkg/alpha.py": (
        "def resize_image(image, max_side=1024):\n"
        '    """Shrink an image so its longest side is at most max_side pixels."""\n'
        "    scale = max_side / max(image.width, image.height)\n"
        "    return image.resize((int(image.width * scale), int(image.height * scale)))\n"
    ),
    "pkg/beta.py": (
        "def send_report(address, body):\n"
        '    """E-mail the weekly report."""\n'
        "    message = EmailMessage()\n    message['To'] = address\n"
        "    message.set_content(body)\n    smtp.send_message(message)\n"
    ),
    "pkg/gamma.py": (
        "def celsius_to_fahrenheit(celsius):\n"
        '    """Convert a temperature from Celsius to Fahrenheit."""\n'
        "    return celsius * 9 / 5 + 32\n"
    ),
    "pkg/delta.py": (
        "def paginate(items, page, per_page=20):\n"
        '    """Return one page of results, counting from 1."""\n'
        "    start = (page - 1) * per_page\n    return items[start : start + per_page]\n"
    ),
}
FILES = {"pkg/omega.py": TARGET, **DISTRACTORS, "docs/notes.md": "# Notes\n\nGeneral remarks.\n"}
QUESTION_EN = "how are input data validated?"
QUESTION_ES = "¿cómo se validan los datos de entrada?"


def _repo(root: Path) -> Path:
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _rules(tmp_path: Path) -> Path:
    path = tmp_path / "rules"
    path.write_text("", encoding="utf-8")
    return path


def test_the_target_never_names_the_question_verb() -> None:
    lowered = TARGET.lower() + " pkg/omega.py ensure_payload"
    assert "valid" not in lowered


# --- hermetic: the plumbing ------------------------------------------------------------------

_NEW_PROCESS = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path[:0] = [repo_root, repo_root + "/tests"]
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from fake_embedding import FakeEmbedding
        from src.indexing.build_index import load_index_with_report
        from src.indexing.search import search
        from src.indexing.vector_store import ChromaIndexStore
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_repository

        spy = FakeEmbedding()
        nodes = chunk_documents(load_repository(work / "repo", rules_file=work / "rules"))
        loaded = load_index_with_report(
            nodes, vector_store=ChromaIndexStore(work / "vs", "code"), embed_model=spy
        )
        embedded_on_load = spy.texts_embedded
        hits = search(loaded.index, "missing fields wrong types payload", top_k=3)
        loaded.close()
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "embedded_on_load": embedded_on_load,
        "files": [h.file_path for h in hits],
    }))
    """
)


def test_build_persist_reload_in_a_new_process_and_search(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    rules = _rules(tmp_path)
    build_index_with_report(
        repo, ChromaIndexStore(tmp_path / "vs", "code"), FakeEmbedding(), rules_file=rules
    )

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _NEW_PROCESS, str(REPO_ROOT), str(tmp_path)],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LC_ALL": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["embedded_on_load"] == 0
    assert report["files"][0] == "pkg/omega.py"


def test_the_phase_4_5_layout_with_load_index(tmp_path: Path) -> None:
    repo = _repo(tmp_path)  # the store lives inside the indexed repository
    store = get_vector_store(persist_dir=tmp_path / "vs", collection="test")
    build_index(tmp_path, vector_store=store, embed_model=FakeEmbedding())

    nodes = chunk_documents(load_repository(repo))
    index = load_index(nodes, vector_store=store, embed_model=FakeEmbedding())
    hits = search(index, "missing fields wrong types payload", top_k=1)

    assert hits[0].file_path == "pkg/omega.py"
    assert not any(str(n.metadata["file_path"]).startswith("vs/") for n in nodes)


def test_drift_and_the_language_filter(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    rules = _rules(tmp_path)
    store = ChromaIndexStore(tmp_path / "vs", "code")
    build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules)
    index = load_index(vector_store=store, embed_model=FakeEmbedding())

    assert {h.language for h in search(index, "notes", top_k=10, language="markdown")} == {
        "markdown"
    }
    (repo / "pkg" / "gamma.py").unlink()
    with pytest.raises(StaleIndexError):
        load_index(
            chunk_documents(load_repository(repo, rules_file=rules)),
            vector_store=store,
            embed_model=FakeEmbedding(),
        )


class _Inverted(BaseEmbedding):
    """A broken embedder: the target's chunk is pushed to the bottom of every ranking."""

    _inner: FakeEmbedding = PrivateAttr(default_factory=FakeEmbedding)

    @classmethod
    def class_name(cls) -> str:
        return "Inverted"

    def identity(self) -> EmbedderIdentity:
        return self._inner.identity()

    def _flip(self, text: str) -> Embedding:
        vector = self._inner.vector(text)
        return [-v for v in vector] if "ensure_payload" in text else vector

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._inner.vector(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._flip(text)


def _target_rank(tmp_path: Path, embedder: BaseEmbedding) -> int:
    repo = _repo(tmp_path / "repo")
    store = ChromaIndexStore(tmp_path / "vs", "code")
    build_index_with_report(repo, store, embedder, rules_file=_rules(tmp_path))
    index = load_index(vector_store=store, embed_model=embedder)
    files = [h.file_path for h in search(index, "missing fields wrong types payload", top_k=10)]
    return files.index("pkg/omega.py") + 1


def test_the_relevance_assertion_can_fail(tmp_path: Path) -> None:
    assert _target_rank(tmp_path / "good", FakeEmbedding()) == 1
    assert _target_rank(tmp_path / "broken", _Inverted()) > 1


# --- integration: the real model, the doc's question, no skip ---------------------------------

RecordProperty = Callable[[str, object], None]


@pytest.mark.integration
def test_the_real_model_ranks_the_input_check_first(
    tmp_path: Path, record_property: RecordProperty
) -> None:
    repo = _repo(tmp_path / "repo")
    store = ChromaIndexStore(tmp_path / "vs", "code")
    build_index_with_report(repo, store, rules_file=_rules(tmp_path))  # the real embedder
    index = load_index(vector_store=store)

    english = [h.file_path for h in search(index, QUESTION_EN, top_k=5)]
    spanish = [h.file_path for h in search(index, QUESTION_ES, top_k=5)]

    record_property(
        "en_rank", english.index("pkg/omega.py") + 1 if "pkg/omega.py" in english else None
    )
    record_property(
        "es_rank", spanish.index("pkg/omega.py") + 1 if "pkg/omega.py" in spanish else None
    )
    assert english[0] == "pkg/omega.py"
    assert "pkg/omega.py" in spanish  # recorded, not required to be first (decision P3)
