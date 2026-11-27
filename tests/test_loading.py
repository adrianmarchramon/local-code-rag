"""Phase 3, Step 5: verified reload (fake embedder, real Chroma in ``tmp_path``).

Every way a persisted index can be missing, unreadable, incompatible or stale; the
reader/writer race with forced interleavings; zero document embeddings on load (in a new
process); no file created or deleted; the replaced-generation check (Gate 1 Q-F).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
from fake_embedding import FakeEmbedding
from llama_index.core import MockEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode

from src.errors import (
    IncompatibleIndexError,
    IndexNotFoundError,
    OllamaUnavailableError,
    StaleIndexError,
    VectorStoreError,
)
from src.indexing.build_index import (
    ExitCode,
    build_index_with_report,
    exit_code_for,
    load_index,
    load_index_with_report,
)
from src.indexing.embeddings import RECIPES, OllamaEmbeddingAdapter
from src.indexing.loading import LoadResult, compatibility_problems, drift_problems
from src.indexing.manifest import MANIFEST_NAME, RepoIdentity, read_manifest, write_manifest
from src.indexing.vector_store import SEARCH_EF_FLOOR, ChromaIndexStore
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_repository
from src.ollama_client import EmbeddingResult, LocalModel, ModelInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "SENTINEL_LOAD_TEXT_8e2a"
FILES = {
    "src/auth.py": f"def verify_token(token):\n    return token == '{SENTINEL}'\n",
    "src/rows.py": "def read_rows(path):\n    return open(path).read().splitlines()\n",
    "README.md": "# Notes\n\nHow tokens are checked.\n",
}


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def rules(tmp_path: Path) -> Path:
    path = tmp_path / "rules"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _write(tmp_path / "repo", FILES)


@pytest.fixture
def store(tmp_path: Path) -> ChromaIndexStore:
    return ChromaIndexStore(tmp_path / "vs", "code")


@pytest.fixture
def built(repo: Path, store: ChromaIndexStore, rules: Path) -> str:
    return build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules).report.generation


def _nodes(repo: Path, rules: Path) -> list[BaseNode]:
    return chunk_documents(load_repository(repo, rules_file=rules))


def _load(
    store: ChromaIndexStore,
    nodes: list[BaseNode] | None = None,
    *,
    embed_model: BaseEmbedding | None = None,
    repo_path: Path | None = None,
    attempts: int = 3,
) -> LoadResult:
    return load_index_with_report(
        nodes,
        vector_store=store,
        embed_model=embed_model if embed_model is not None else FakeEmbedding(),
        repo_path=repo_path,
        attempts=attempts,
    )


def _tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


# --- the happy path ----------------------------------------------------------------------------


def test_a_built_index_loads_verified_and_answers(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str
) -> None:
    fake = FakeEmbedding()
    result = load_index_with_report(_nodes(repo, rules), vector_store=store, embed_model=fake)

    assert (result.generation, result.drift_checked, result.attempts) == (built, True, 1)
    assert fake.texts_embedded == 0  # loading embeds no document
    hits = result.index.as_retriever(similarity_top_k=1).retrieve("verify the token")
    assert hits[0].node.metadata["file_path"] == "src/auth.py"
    assert fake.batches == [1]  # only the query
    result.close()


def test_the_contract_function_returns_the_index(store: ChromaIndexStore, built: str) -> None:
    index = load_index(vector_store=store, embed_model=FakeEmbedding())

    assert index.as_retriever(similarity_top_k=2).retrieve("rows")


def test_loading_creates_and_deletes_nothing_and_keeps_the_manifest(
    store: ChromaIndexStore, built: str
) -> None:
    before = _tree(store.persist_dir)
    manifest_bytes = (store.persist_dir / MANIFEST_NAME).read_bytes()

    result = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    result.index.as_retriever(similarity_top_k=2).retrieve("token")
    result.close()

    assert _tree(store.persist_dir) == before
    assert (store.persist_dir / MANIFEST_NAME).read_bytes() == manifest_bytes


def test_loading_does_not_take_the_build_lock(store: ChromaIndexStore, built: str) -> None:
    with store.lock():
        result = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    result.close()


def test_two_consecutive_loads_and_closing(store: ChromaIndexStore, built: str) -> None:
    first = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    second = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())

    assert first.index.as_retriever(similarity_top_k=1).retrieve("rows")
    first.close()
    assert second.index.as_retriever(similarity_top_k=1).retrieve("rows")  # still usable
    second.close()


def test_without_nodes_drift_is_skipped_and_logged(
    store: ChromaIndexStore, built: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    result.close()

    assert result.drift_checked is False
    assert "index.drift_unchecked" in caplog.text
    assert "index.loaded" in caplog.text
    assert SENTINEL not in caplog.text


# --- missing or unreadable ---------------------------------------------------------------------


def test_a_missing_index_or_directory_is_not_created(tmp_path: Path) -> None:
    for root in (tmp_path / "absent", tmp_path / "empty"):
        if root.name == "empty":
            root.mkdir()
        with pytest.raises(IndexNotFoundError, match="make index"):
            _load(ChromaIndexStore(root, "code"))
    assert not (tmp_path / "absent").exists()
    assert list((tmp_path / "empty").iterdir()) == []


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{broken", "not valid JSON"),
        ('{"format": "local-code-rag/index-manifest", "schema_version": 9}', "schema version 9"),
    ],
    ids=["corrupt", "schema"],
)
def test_an_unusable_manifest_is_an_error(
    store: ChromaIndexStore, built: str, text: str, match: str
) -> None:
    (store.persist_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")

    with pytest.raises(VectorStoreError, match=match):
        _load(store)


def test_a_missing_generation_is_an_error_without_retrying_forever(
    store: ChromaIndexStore, built: str
) -> None:
    manifest = read_manifest(store.persist_dir)
    assert manifest is not None
    write_manifest(
        store.persist_dir,
        dataclasses.replace(manifest, generation="gen-20261127T000000000000Z-00000000"),
    )

    with pytest.raises(VectorStoreError, match="never recreated silently"):
        _load(store)
    assert not (store.persist_dir / "gen-20261127T000000000000Z-00000000").exists()


# --- incompatible ------------------------------------------------------------------------------


def _rewrite(store: ChromaIndexStore, **changes: object) -> None:
    manifest = read_manifest(store.persist_dir)
    assert manifest is not None
    write_manifest(store.persist_dir, dataclasses.replace(manifest, **changes))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("embedder", "tamper", "expected"),
    [
        (FakeEmbedding(digest="e" * 64), None, ["model digest"]),
        (FakeEmbedding(dimension=32), None, ["recipe fingerprint", "dimension 64"]),
        (FakeEmbedding(), {"metric": "l2"}, ["metric 'l2'"]),
        (FakeEmbedding(), {"nodes": 99}, ["vectors stored for 99 nodes"]),
    ],
    ids=["model_digest", "recipe_and_dimension", "metric", "count"],
)
def test_an_incompatible_index_lists_every_difference(
    store: ChromaIndexStore,
    built: str,
    embedder: FakeEmbedding,
    tamper: dict[str, object] | None,
    expected: list[str],
) -> None:
    if tamper:
        _rewrite(store, **tamper)

    with pytest.raises(IncompatibleIndexError, match="make index") as raised:
        _load(store, embed_model=embedder)

    for fragment in expected:
        assert fragment in str(raised.value)


def test_an_index_built_by_an_unverified_embedder_is_not_loaded(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    build_index_with_report(repo, store, MockEmbedding(embed_dim=64), rules_file=rules)

    with pytest.raises(IncompatibleIndexError, match="unverified embedder"):
        _load(store)


def test_an_embedder_without_identity_cannot_load(store: ChromaIndexStore, built: str) -> None:
    with pytest.raises(IncompatibleIndexError, match="cannot prove"):
        _load(store, embed_model=MockEmbedding(embed_dim=64))


class _DownClient:
    def version(self) -> str:
        raise OllamaUnavailableError("down")

    def list_models(self) -> list[LocalModel]:
        raise OllamaUnavailableError("down")

    def show(self, model: str) -> ModelInfo:
        raise OllamaUnavailableError("down")

    def embed(self, model: str, inputs: object, *, truncate: bool | None = None) -> EmbeddingResult:
        raise OllamaUnavailableError("down")


def test_the_model_check_is_never_skipped_when_ollama_is_down(
    store: ChromaIndexStore, built: str
) -> None:
    adapter = OllamaEmbeddingAdapter(_DownClient(), RECIPES["nomic-embed-text:latest"])

    with pytest.raises(OllamaUnavailableError):
        _load(store, embed_model=adapter)


# --- stale -------------------------------------------------------------------------------------


@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_a_changed_repository_is_stale(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str, change: str
) -> None:
    if change == "modify":
        (repo / "src" / "rows.py").write_text(
            "def read_rows(path):\n    return []\n", encoding="utf-8"
        )
    elif change == "delete":
        (repo / "src" / "rows.py").unlink()
    else:
        (repo / "src" / "new.py").write_text("def new():\n    return 1\n", encoding="utf-8")

    with pytest.raises(StaleIndexError, match="no longer match") as raised:
        _load(store, _nodes(repo, rules))
    assert "make index" in str(raised.value)
    assert SENTINEL not in str(raised.value)


def test_other_ingestion_rules_are_drift_too(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str, tmp_path: Path
) -> None:
    stricter = tmp_path / "stricter"
    stricter.write_text("README.md\n", encoding="utf-8")

    with pytest.raises(StaleIndexError, match="ingestion or chunking rules"):
        _load(store, _nodes(repo, stricter))


def test_empty_nodes_are_checked_not_skipped(store: ChromaIndexStore, built: str) -> None:
    with pytest.raises(StaleIndexError):
        _load(store, [])


def _git_head(repo: Path, sha: str) -> None:
    (repo / ".git").mkdir(exist_ok=True)
    (repo / ".git" / "HEAD").write_text(sha + "\n", encoding="ascii")


def test_a_moved_head_is_stale_even_with_identical_nodes(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    _git_head(repo, "a" * 40)
    build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules)
    nodes = _nodes(repo, rules)
    _git_head(repo, "b" * 40)  # Gate 1 Q-E: a new commit is drift

    with pytest.raises(StaleIndexError, match="HEAD moved from aaaaaaaaaaaa to bbbbbbbbbbbb"):
        _load(store, nodes)


def test_a_head_that_can_no_longer_be_read_is_stale(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    _git_head(repo, "a" * 40)
    build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules)
    (repo / ".git" / "HEAD").write_text("garbage\n", encoding="ascii")

    with pytest.raises(StaleIndexError, match="can no longer be read"):
        _load(store, _nodes(repo, rules))


def test_a_repository_without_git_checks_the_digest_only(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str
) -> None:
    result = _load(store, _nodes(repo, rules), repo_path=repo)

    assert result.drift_checked is True
    result.close()


# --- pure checks -------------------------------------------------------------------------------


def test_the_pure_checks_report_nothing_for_a_matching_index(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str
) -> None:
    manifest = read_manifest(store.persist_dir)
    assert manifest is not None
    identity = FakeEmbedding().identity()

    assert (
        compatibility_problems(
            manifest, identity, manifest.nodes, "cosine", search_breadth=manifest.nodes
        )
        == []
    )
    approximate = compatibility_problems(
        manifest, identity, manifest.nodes, "cosine", search_breadth=manifest.nodes - 1
    )
    assert len(approximate) == 1
    assert "approximate" in approximate[0]
    assert drift_problems(manifest, _nodes(repo, rules), manifest.repo) == []
    other = RepoIdentity(manifest.repo.path, "git", "c" * 40, None)
    assert drift_problems(manifest, _nodes(repo, rules), other) == []  # recorded: not a git repo


# --- the generation is replaced under a reader (Gate 1 Q-F) -------------------------------------


def test_a_reader_learns_that_its_generation_was_replaced(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str
) -> None:
    result = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    assert result.is_current()
    (repo / "src" / "new.py").write_text("def new():\n    return 1\n", encoding="utf-8")

    build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules)

    assert not result.is_current()
    fresh = load_index_with_report(vector_store=store, embed_model=FakeEmbedding())
    assert fresh.is_current()
    assert fresh.generation != result.generation
    result.close()
    fresh.close()


# --- the reader/writer race, with forced interleavings -----------------------------------------


def _rebuild(repo: Path, store: ChromaIndexStore, rules: Path) -> None:
    marker = repo / "src" / f"race_{len(list((repo / 'src').iterdir()))}.py"
    marker.write_text("def race():\n    return 0\n", encoding="utf-8")
    build_index_with_report(repo, store, FakeEmbedding(), rules_file=rules)


def test_a_generation_deleted_between_reading_and_opening_is_retried(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ChromaIndexStore._open
    calls = {"n": 0}

    def interleaved(self: ChromaIndexStore, name: str, path: Path, *, create: bool) -> object:
        if not create and calls["n"] == 0:
            calls["n"] += 1
            _rebuild(repo, store, rules)  # activates a new generation, deletes `name`
        return original(self, name, path, create=create)

    monkeypatch.setattr(ChromaIndexStore, "_open", interleaved)

    result = _load(store)

    assert result.attempts == 2
    assert result.generation != built
    assert not (store.persist_dir / built).exists()
    result.close()


def test_a_generation_replaced_right_after_opening_is_retried(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ChromaIndexStore.open_active
    calls = {"n": 0}

    def interleaved(self: ChromaIndexStore) -> object:
        opened = original(self)
        if calls["n"] == 0:
            calls["n"] += 1
            _rebuild(repo, store, rules)
        return opened

    monkeypatch.setattr(ChromaIndexStore, "open_active", interleaved)

    result = _load(store)

    assert result.attempts == 2
    assert result.is_current()
    result.close()


def test_an_index_that_keeps_changing_fails_after_bounded_attempts(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ChromaIndexStore.open_active

    def always_rebuilt(self: ChromaIndexStore) -> object:
        opened = original(self)
        _rebuild(repo, store, rules)
        return opened

    monkeypatch.setattr(ChromaIndexStore, "open_active", always_rebuilt)

    with pytest.raises(VectorStoreError, match="kept changing"):
        _load(store, attempts=3)


# --- hostile environment and a new process -------------------------------------------------------


def test_a_hostile_env_file_does_not_change_where_loading_reads(
    store: ChromaIndexStore, built: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "CHROMA_API_IMPL=chromadb.api.fastapi.FastAPI\nCHROMA_SERVER_HOST=203.0.113.10\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_SERVER_HOST", "203.0.113.10")

    result = _load(store)

    assert result.index.as_retriever(similarity_top_k=1).retrieve("token")
    result.close()


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
        from src.indexing.vector_store import ChromaIndexStore
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_repository

        spy = FakeEmbedding()
        nodes = chunk_documents(load_repository(work / "repo", rules_file=work / "rules"))
        result = load_index_with_report(
            nodes, vector_store=ChromaIndexStore(work / "vs", "code"), embed_model=spy
        )
        embedded_on_load = spy.texts_embedded
        hits = result.index.as_retriever(similarity_top_k=1).retrieve("verify the token")
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "embedded_on_load": embedded_on_load,
        "batches_after_query": spy.batches,
        "top": hits[0].node.metadata["file_path"],
        "home": sorted(p.name for p in (work / "home").iterdir()),
    }))
    """
)


def test_a_new_process_loads_without_embedding_any_document(
    repo: Path, store: ChromaIndexStore, rules: Path, built: str, tmp_path: Path
) -> None:
    (tmp_path / "home").mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "XDG_CACHE_HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path / "home"),
        "LC_ALL": "C.UTF-8",
    }

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _NEW_PROCESS, str(REPO_ROOT), str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["embedded_on_load"] == 0
    assert report["batches_after_query"] == [1]
    assert report["top"] == "src/auth.py"
    assert report["home"] == []


# --- exhaustive search (the Step 2 fix of 2026-11-27) -----------------------------------------


def test_a_build_makes_the_search_exhaustive(store: ChromaIndexStore, built: str) -> None:
    _, active = store.open_active()
    try:
        assert active.search_breadth() == max(SEARCH_EF_FLOOR, active.count())
    finally:
        active.close()


def test_an_index_whose_search_is_approximate_is_refused_with_exit_34(
    store: ChromaIndexStore, built: str
) -> None:
    _, active = store.open_active()
    try:  # what an index built before the fix looks like: ef_search below its vector count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # Chroma's "legacy" EF notice
            active._live().modify(configuration={"hnsw": {"ef_search": 1}})
    finally:
        active.close()

    with pytest.raises(IncompatibleIndexError, match="approximate") as raised:
        _load(store)

    assert "make index" in str(raised.value)
    assert exit_code_for(raised.value) == ExitCode.INDEX_INCOMPATIBLE
