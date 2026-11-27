"""Phase 3, Step 6: similarity search, its CLI, the query recipe and the exact oracle.

Fake embedder (or the real adapter over ``httpx.MockTransport``) and real Chroma in
``tmp_path``. No model and no external network.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Iterator
from io import StringIO
from itertools import pairwise
from pathlib import Path

import httpx
import pytest
from fake_embedding import FakeEmbedding
from retrieval_oracle import disagreements, exact_ranking, hnsw_ranking

from src import config
from src.errors import EmbeddingInputTooLongError, InvalidQueryError
from src.indexing import loading
from src.indexing.build_index import build_index_with_report, load_index_with_report
from src.indexing.embeddings import (
    NOMIC_EMBED_TEXT_V15_DIGEST,
    RECIPES,
    OllamaEmbeddingAdapter,
)
from src.indexing.manifest import MANIFEST_NAME
from src.indexing.search import SearchHit, cosine_from_score, main, sanitize, search
from src.indexing.vector_store import ChromaIndexStore
from src.ollama_client import INPUT_TOO_LONG_MESSAGE, OllamaHttpClient

SENTINEL_CODE = "SENTINEL_CODE_IN_HIT_3f9b"
SENTINEL_QUERY = "SENTINEL_QUERY_TEXT_77c1"
ANSI = "\x1b[31m"


def _files() -> dict[str, str]:
    files = {
        "src/auth.py": (
            "def verify_token(token, key):\n"
            '    """Check the signature of an access token."""\n'
            f"    return token == key  # {SENTINEL_CODE} {ANSI}red\n"
        ),
        "docs/guide.md": "# Guide\n\nHow access tokens are checked.\n\n## Install\n\nRun make.\n",
        "config/app.yaml": "server:\n  port: 8080\n",
    }
    for i in range(12):  # enough nodes for a top-20 comparison
        files[f"src/mod{i}.py"] = (
            f"def task_{i}(value):\n    return value * {i}\n\n\n"
            f"def helper_{i}(items):\n    return [x for x in items if x > {i}]\n"
        )
    return files


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in _files().items():
        (tmp_path / "repo" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / rel).write_text(text, encoding="utf-8")
    return tmp_path / "repo"


@pytest.fixture
def rules(tmp_path: Path) -> Path:
    (tmp_path / "rules").write_text("", encoding="utf-8")
    return tmp_path / "rules"


@pytest.fixture
def store(tmp_path: Path, repo: Path, rules: Path) -> ChromaIndexStore:
    store = ChromaIndexStore(tmp_path / "vs", "code")
    build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules)
    return store


@pytest.fixture
def loaded(store: ChromaIndexStore) -> Iterator[loading.LoadResult]:
    result = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    yield result
    result.close()


# --- the search function ------------------------------------------------------------------------


def test_hits_carry_the_source_fields_rank_and_cosine(loaded: loading.LoadResult) -> None:
    hits = search(loaded.index, "check the access token signature", top_k=5)

    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    top = hits[0]
    assert isinstance(top, SearchHit)
    assert (top.file_path, top.symbol, top.start_line, top.end_line) == (
        "src/auth.py",
        "verify_token",
        1,
        3,
    )
    for hit in hits:
        assert hit.score is not None
        assert hit.cosine == pytest.approx(1 + math.log(hit.score))
    cosines = [h.cosine for h in hits if h.cosine is not None]
    assert all(a >= b for a, b in pairwise(cosines))


def test_the_cosine_formula_handles_a_missing_or_invalid_score() -> None:
    assert cosine_from_score(None) is None
    assert cosine_from_score(0.0) is None
    assert cosine_from_score(math.exp(-0.25)) == pytest.approx(0.75)


def test_the_language_filter_and_its_errors(loaded: loading.LoadResult) -> None:
    markdown = search(loaded.index, "tokens", top_k=10, language="markdown")
    rust = search(loaded.index, "tokens", top_k=10, language="rust")  # known label, no chunks

    assert markdown
    assert {h.language for h in markdown} == {"markdown"}
    assert rust == ()
    with pytest.raises(InvalidQueryError, match="Unknown language"):
        search(loaded.index, "tokens", language="klingon")


@pytest.mark.parametrize(
    ("query", "top_k", "match"),
    [
        ("", 5, "empty"),
        ("   \n", 5, "empty"),
        ("q", 0, "top_k"),
        ("q", -2, "top_k"),
        ("q", True, "top_k"),
    ],
    ids=["empty", "blank", "zero", "negative", "bool"],
)
def test_unusable_queries_are_refused_without_quoting_them(
    loaded: loading.LoadResult, query: str, top_k: int, match: str
) -> None:
    with pytest.raises(InvalidQueryError, match=match):
        search(loaded.index, query, top_k=top_k)


def test_top_k_larger_than_the_index_returns_every_chunk(loaded: loading.LoadResult) -> None:
    hits = search(loaded.index, "value", top_k=1000)

    assert len(hits) == loaded.manifest.nodes


def test_searching_changes_no_file(store: ChromaIndexStore, loaded: loading.LoadResult) -> None:
    before = {p.relative_to(store.persist_dir).as_posix() for p in store.persist_dir.rglob("*")}
    manifest_bytes = (store.persist_dir / MANIFEST_NAME).read_bytes()

    search(loaded.index, "helper", top_k=5)

    assert {
        p.relative_to(store.persist_dir).as_posix() for p in store.persist_dir.rglob("*")
    } == before
    assert (store.persist_dir / MANIFEST_NAME).read_bytes() == manifest_bytes


def test_the_query_text_never_reaches_logs_or_errors(
    loaded: loading.LoadResult, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    search(loaded.index, f"{SENTINEL_QUERY} token", top_k=3)
    with pytest.raises(InvalidQueryError) as raised:
        search(loaded.index, f"{SENTINEL_QUERY}", top_k=0)

    assert "search.done" in caplog.text
    assert SENTINEL_QUERY not in caplog.text
    assert SENTINEL_CODE not in caplog.text
    assert SENTINEL_QUERY not in str(raised.value)


# --- the exact oracle: HNSW returns what brute force returns ----------------------------------


def test_hnsw_matches_an_exact_cosine_knn(loaded: loading.LoadResult) -> None:
    generation = loaded._generation
    store = generation.vector_store()
    entries = list(generation.entries())
    assert len(entries) > 20
    fake = FakeEmbedding()
    for query in ("check the access token", "helper items", "server port", "multiply value"):
        vector = fake.get_query_embedding(query)
        exact = exact_ranking(entries, vector, 20)
        hnsw = hnsw_ranking(store, vector, 20)

        assert disagreements(exact, hnsw) == [], query


def test_the_oracle_comparison_can_fail() -> None:
    exact = [("a", 0.9), ("b", 0.8), ("c", 0.1)]

    assert disagreements(exact, [("a", 0.9, 0.0), ("c", 0.1, 0.0), ("b", 0.8, 0.0)])
    assert disagreements(exact, [("a", 0.9, 0.0), ("b", 0.8, 0.0)])


# --- the query recipe, through the real adapter over a mock transport ---------------------------


def _mock_adapter(requests: list[list[str]]) -> OllamaEmbeddingAdapter:
    recipe = dataclasses.replace(RECIPES["nomic-embed-text:latest"], dimension=16)
    vectorizer = FakeEmbedding(dimension=16)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            model = {
                "model": "nomic-embed-text:latest",
                "name": "nomic-embed-text:latest",
                "digest": NOMIC_EMBED_TEXT_V15_DIGEST,
                "size": 274_302_450,
            }
            return httpx.Response(200, json={"models": [model]})
        if path == "/api/show":
            info = {
                "general.architecture": "nomic-bert",
                "nomic-bert.context_length": 2048,
                "nomic-bert.embedding_length": 16,
            }
            return httpx.Response(200, json={"model_info": info, "capabilities": ["embedding"]})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.0"})
        inputs = json.loads(request.content)["input"]
        requests.append(list(inputs))
        if any(SENTINEL_QUERY in text and len(text) > 5000 for text in inputs):
            return httpx.Response(400, json={"error": INPUT_TOO_LONG_MESSAGE})
        return httpx.Response(200, json={"embeddings": [vectorizer.vector(t) for t in inputs]})

    client = OllamaHttpClient("http://localhost:11434", 5.0, transport=httpx.MockTransport(handler))
    return OllamaEmbeddingAdapter(client, recipe, sleep=lambda _: None)


def test_the_query_carries_the_query_prefix_once_and_hits_carry_no_prefix(
    repo: Path, rules: Path, tmp_path: Path
) -> None:
    requests: list[list[str]] = []
    adapter = _mock_adapter(requests)
    store = ChromaIndexStore(tmp_path / "vs-mock", "code")
    build_index_with_report(repo, store, adapter, rules_file=rules)
    documents = [text for batch in requests for text in batch]
    requests.clear()

    result = load_index_with_report(vector_store=store, embed_model=adapter)
    hits = search(result.index, "how is a token checked?", top_k=5)
    result.close()

    assert all(
        t.startswith("search_document: ") and not t.startswith("search_document: search_document: ")
        for t in documents
    )
    assert requests == [["search_query: how is a token checked?"]]
    assert hits
    assert not any(h.text.startswith(("search_document: ", "search_query: ")) for h in hits)


def test_an_over_long_query_raises_and_is_never_truncated(
    repo: Path, rules: Path, tmp_path: Path
) -> None:
    requests: list[list[str]] = []
    adapter = _mock_adapter(requests)
    store = ChromaIndexStore(tmp_path / "vs-mock", "code")
    build_index_with_report(repo, store, adapter, rules_file=rules)
    result = load_index_with_report(vector_store=store, embed_model=adapter)
    long_query = (SENTINEL_QUERY + " ") * 400

    with pytest.raises(EmbeddingInputTooLongError) as raised:
        search(result.index, long_query, top_k=3)
    result.close()

    assert SENTINEL_QUERY not in str(raised.value)
    assert requests[-1] == ["search_query: " + long_query]  # sent whole, never cut


# --- presentation ------------------------------------------------------------------------------


def test_sanitize_escapes_controls_and_ansi_but_keeps_newlines_and_tabs() -> None:
    assert sanitize("a\x1b[31mb\tc\nd\x07\x9b") == "a\\x1b[31mb\tc\nd\\x07\\x9b"


# --- CLI ---------------------------------------------------------------------------------------


def _cli(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    store_dir: Path,
    embedder: FakeEmbedding | None = None,
) -> tuple[int, str]:
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", store_dir)
    monkeypatch.setattr(loading, "get_embedding_model", lambda: embedder or FakeEmbedding())
    out = StringIO()
    code = main(argv, stdout=out, stdin=StringIO("verify the access token\n"))
    return code, out.getvalue()


def test_the_cli_prints_no_code_without_show(
    repo: Path, rules: Path, store: ChromaIndexStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(
        ["check the access token", "--repo", str(repo), "--rules-file", str(rules)],
        monkeypatch,
        store.persist_dir,
    )

    assert code == 0
    assert "src/auth.py:1-3" in output
    assert SENTINEL_CODE not in output
    assert "RESULT: OK" in output


def test_the_cli_shows_sanitized_code_only_on_request(
    repo: Path, rules: Path, store: ChromaIndexStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(
        ["-", "--top-k", "1", "--show", "--repo", str(repo), "--rules-file", str(rules)],
        monkeypatch,
        store.persist_dir,
    )

    assert code == 0
    assert SENTINEL_CODE in output  # stdin query, --show
    assert "\x1b" not in output
    assert "\\x1b[31m" in output


def test_the_cli_json_has_no_text(
    repo: Path, rules: Path, store: ChromaIndexStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(
        ["token", "--json", "--top-k", "3", "--repo", str(repo), "--rules-file", str(rules)],
        monkeypatch,
        store.persist_dir,
    )

    hits = json.loads(output)
    assert code == 0
    assert len(hits) == 3
    assert "text" not in hits[0]
    assert SENTINEL_CODE not in output


def test_the_cli_exit_codes(
    repo: Path,
    rules: Path,
    store: ChromaIndexStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = ["--repo", str(repo), "--rules-file", str(rules)]

    assert _cli(["   ", *args], monkeypatch, store.persist_dir)[0] == 2
    assert _cli(["q", "--top-k", "0", *args], monkeypatch, store.persist_dir)[0] == 2
    assert _cli(["q", "--language", "klingon", *args], monkeypatch, store.persist_dir)[0] == 2
    assert _cli(["q", *args], monkeypatch, tmp_path / "no-index")[0] == 33
    assert (
        _cli(["q", *args], monkeypatch, store.persist_dir, FakeEmbedding(digest="e" * 64))[0] == 34
    )
    (repo / "src" / "auth.py").write_text("def changed():\n    return 0\n", encoding="utf-8")
    stale_code, stale_output = _cli(["q", *args], monkeypatch, store.persist_dir)
    assert stale_code == 35
    assert "make index" in stale_output
    assert not (tmp_path / "no-index").exists()
