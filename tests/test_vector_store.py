"""Phase 3, Step 3: the hardened Chroma store, generations, manifest and static audit.

Real Chroma in ``tmp_path``, real Phase 2 nodes from synthetic documents, and the
deterministic fake embedder. No model and no external network.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import stat
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_embedding import FakeEmbedding
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from src.errors import IndexLockedError, IndexNotFoundError, VectorStoreError
from src.indexing import vector_store as vs
from src.indexing.embeddings import EmbedderIdentity
from src.indexing.manifest import (
    MANIFEST_NAME,
    IndexManifest,
    RepoIdentity,
    format_manifest,
    parse_manifest,
    read_manifest,
    write_manifest,
)
from src.indexing.vector_store import (
    ChromaGeneration,
    ChromaIndexStore,
    chroma_settings,
    get_vector_store,
    open_chroma_client,
)
from src.ingestion.chunker import chunk_documents

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SENTINEL = "SENTINEL_CODE_TEXT_6b2e"


def _document(text: str, path: str, language: str) -> Document:
    return Document(
        id_=path,
        text=text,
        metadata={
            "file_path": path,
            "language": language,
            "size_bytes": len(text.encode("utf-8")),
            "num_lines": text.count("\n") + 1,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "directory": path.rpartition("/")[0],
        },
    )


DOCUMENTS = [
    _document(
        "import hmac\n\n\ndef verify_token(token):\n"
        f"    return hmac.compare_digest(token, '{SENTINEL}')\n\n\n"
        "class Store:\n    def save(self, item):\n        return item\n",
        "src/auth.py",
        "python",
    ),
    _document(
        "# Guide\n\nHow tokens work.\n\n## Usage\n\nRun the tool.\n", "README.md", "markdown"
    ),
    _document("jobs:\n  test:\n    runs-on: ubuntu\n", ".github/ci.yml", "yaml"),
]


def _nodes() -> list[BaseNode]:
    return chunk_documents(DOCUMENTS)


def _with_vectors(nodes: list[BaseNode], embedder: FakeEmbedding) -> list[BaseNode]:
    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    copies = []
    for node, vector in zip(nodes, embedder.get_text_embedding_batch(texts), strict=True):
        copy = node.model_copy()
        copy.embedding = vector
        copies.append(copy)
    return copies


def _manifest(generation: str, collection: str = "code", **changes: object) -> IndexManifest:
    base = IndexManifest(
        generation=generation,
        collection=collection,
        created_utc="2026-11-27T00:00:00Z",
        repo=RepoIdentity("/repo", "not_a_git_repo", None, None),
        documents=3,
        nodes=5,
        node_digest="a" * 64,
        embed_digest="b" * 64,
        embedder=EmbedderIdentity("fake-embedding:latest", "f" * 64, "c" * 64, 64),
        recipe=None,
        ollama_version=None,
        metric="cosine",
        max_chunk_chars=1500,
        rules_file_sha256=None,
        libraries={"chromadb": "1.5.9"},
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> ChromaIndexStore:
    return ChromaIndexStore(tmp_path / "vs", "code")


@pytest.fixture
def staged(store: ChromaIndexStore) -> Iterator[ChromaGeneration]:
    generation = store.create_staging()
    generation.add(_with_vectors(_nodes(), FakeEmbedding()))
    yield generation
    generation.close()


def _tree(root: Path) -> dict[str, tuple[int, int, int]]:
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): (
            p.lstat().st_mode,
            p.lstat().st_size,
            p.lstat().st_mtime_ns,
        )
        for p in sorted(root.rglob("*"))
    }


# --- the single client factory: local, verified, fail-closed ----------------------------------


def test_settings_pin_every_field_that_decides_the_destination(tmp_path: Path) -> None:
    settings = chroma_settings(tmp_path)

    assert settings.chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
    assert (settings.is_persistent, settings.persist_directory) == (True, str(tmp_path))
    assert (settings.anonymized_telemetry, settings.allow_reset) == (False, False)
    assert settings.chroma_server_host is None
    assert settings.chroma_otel_collection_endpoint == ""
    assert settings.chroma_otel_granularity is None


def test_a_hostile_env_file_and_environment_cannot_redirect_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "CHROMA_API_IMPL=chromadb.api.fastapi.FastAPI\nCHROMA_SERVER_HOST=203.0.113.10\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "CHROMA_API_IMPL": "chromadb.api.fastapi.FastAPI",
        "CHROMA_SERVER_HOST": "203.0.113.10",
        "CHROMA_SERVER_HTTP_PORT": "8000",
        "CHROMA_OTEL_COLLECTION_ENDPOINT": "http://203.0.113.10:4317",
        "CHROMA_OTEL_GRANULARITY": "all",
        "ALLOW_RESET": "TRUE",
        "ANONYMIZED_TELEMETRY": "TRUE",
    }.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "db").mkdir()

    client = open_chroma_client(tmp_path / "db")
    try:
        effective = client.get_settings()
        assert effective.chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
        assert effective.chroma_server_host is None
        assert effective.chroma_otel_collection_endpoint == ""
        assert (effective.allow_reset, effective.anonymized_telemetry) == (False, False)
    finally:
        client.close()


def test_effective_settings_that_differ_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = vs.chroma_settings

    def telemetry_on(path: Path) -> object:
        settings = original(path)
        settings.anonymized_telemetry = True
        return settings

    monkeypatch.setattr(vs, "chroma_settings", telemetry_on)
    (tmp_path / "db").mkdir()

    with pytest.raises(VectorStoreError, match="anonymized_telemetry"):
        open_chroma_client(tmp_path / "db")


def test_the_factory_never_creates_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="never opened for reading"):
        open_chroma_client(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


# --- generations: create, round trip, metric, filter, score ------------------------------------


def test_staging_uses_cosine_and_no_embedding_function(staged: ChromaGeneration) -> None:
    assert staged.metric() == "cosine"
    assert staged._live()._embedding_function is None
    assert "onnxruntime" not in sys.modules


def test_nodes_come_back_exactly_as_written(staged: ChromaGeneration) -> None:
    originals = {n.node_id: n for n in _nodes()}
    seen = 0
    for node, vector in staged.entries():
        original = originals[node.node_id]
        assert isinstance(node, TextNode)
        assert isinstance(original, TextNode)
        assert node.text == original.text
        assert list(node.metadata.items()) == list(original.metadata.items())
        assert [type(v) for v in node.metadata.values()] == [
            type(v) for v in original.metadata.values()
        ]
        assert ("symbol" in node.metadata) == ("symbol" in original.metadata)
        assert node.ref_doc_id == original.ref_doc_id
        assert (node.start_char_idx, node.end_char_idx) == (
            original.start_char_idx,
            original.end_char_idx,
        )
        assert node.excluded_embed_metadata_keys == original.excluded_embed_metadata_keys
        assert node.excluded_llm_metadata_keys == original.excluded_llm_metadata_keys
        assert node.hash == original.hash
        assert node.get_content(MetadataMode.EMBED) == original.get_content(MetadataMode.EMBED)
        assert len(vector) == 64
        seen += 1
    assert seen == len(originals) == staged.count()


def test_the_language_filter_works_through_the_retriever(staged: ChromaGeneration) -> None:
    fake = FakeEmbedding()
    index = VectorStoreIndex.from_vector_store(staged.vector_store(), embed_model=fake)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="language", value="markdown", operator=FilterOperator.EQ)]
    )

    hits = index.as_retriever(similarity_top_k=10, filters=filters).retrieve("tokens")

    assert hits
    assert {h.node.metadata["language"] for h in hits} == {"markdown"}


def test_the_score_is_exp_of_cosine_minus_one(staged: ChromaGeneration) -> None:
    fake = FakeEmbedding()
    query = fake.get_query_embedding("verify the token")
    stored = {node.node_id: vector for node, vector in staged.entries()}

    result = staged.vector_store().query(
        VectorStoreQuery(query_embedding=query, similarity_top_k=5)
    )

    assert result.nodes
    assert result.similarities
    for node, score in zip(result.nodes, result.similarities, strict=True):
        vector = stored[node.node_id]
        cosine = sum(q * v for q, v in zip(query, vector, strict=True)) / math.sqrt(
            sum(v * v for v in vector)
        )
        assert math.isclose(1 + math.log(score), cosine, abs_tol=1e-5)  # float32 storage


def test_writes_are_batched_under_the_store_maximum(
    store: ChromaIndexStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = store.create_staging()
    try:
        monkeypatch.setattr(generation, "max_batch_size", lambda: 2)
        generation.add(_with_vectors(_nodes(), FakeEmbedding()))
        assert generation.count() == len(_nodes())
    finally:
        generation.close()


# --- logs never carry code, whatever the root level ---------------------------------------------


def test_node_text_never_reaches_the_logs_even_with_the_root_at_debug(
    staged: ChromaGeneration, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)  # the root logger only
    query = FakeEmbedding().get_query_embedding("token")

    staged.vector_store().query(VectorStoreQuery(query_embedding=query, similarity_top_k=5))
    index = VectorStoreIndex.from_vector_store(staged.vector_store(), embed_model=FakeEmbedding())
    index.as_retriever(similarity_top_k=5).retrieve("token")

    assert SENTINEL not in caplog.text


def test_the_log_check_can_fail(staged: ChromaGeneration, caplog: pytest.LogCaptureFixture) -> None:
    """Negative control: with the library logger opened up, the same query leaks."""
    library = logging.getLogger("llama_index")
    caplog.set_level(logging.DEBUG)
    library.setLevel(logging.DEBUG)
    try:
        query = FakeEmbedding().get_query_embedding("token")
        staged.vector_store().query(VectorStoreQuery(query_embedding=query, similarity_top_k=5))
    finally:
        library.setLevel(logging.WARNING)

    assert SENTINEL in caplog.text


# --- the location handle: reading never writes, errors are explicit ----------------------------


def test_constructing_and_reading_write_nothing(tmp_path: Path) -> None:
    store = ChromaIndexStore(tmp_path / "vs", "code")

    assert store.read_manifest() is None
    with pytest.raises(IndexNotFoundError, match="make index"):
        store.open_active()
    assert not (tmp_path / "vs").exists()


def test_reading_an_active_index_creates_or_deletes_nothing(store: ChromaIndexStore) -> None:
    """Measured: Chroma has no read-only mode; opening a generation rewrites the content of
    its own sqlite and HNSW files (no file is created or deleted). The handle guarantees
    the rest: no path is created, nothing is removed, the manifest is untouched."""
    generation = store.create_staging()
    generation.add(_with_vectors(_nodes(), FakeEmbedding()))
    store.activate(generation, _manifest(generation.name))
    generation.close()
    manifest_bytes = (store.persist_dir / MANIFEST_NAME).read_bytes()
    before = _tree(store.persist_dir)

    manifest, active = store.open_active()
    count = active.count()
    list(active.entries())
    active.close()

    assert (manifest.generation, count) == (generation.name, len(_nodes()))
    assert set(_tree(store.persist_dir)) == set(before)
    assert (store.persist_dir / MANIFEST_NAME).read_bytes() == manifest_bytes


def test_a_manifest_pointing_to_a_missing_generation_is_an_error(store: ChromaIndexStore) -> None:
    store.persist_dir.mkdir(mode=0o700)
    write_manifest(store.persist_dir, _manifest("gen-20261127T000000000000Z-00000000"))

    with pytest.raises(VectorStoreError, match="never recreated silently"):
        store.open_active()
    assert not (store.persist_dir / "gen-20261127T000000000000Z-00000000").exists()


def test_another_collection_name_is_an_error(store: ChromaIndexStore) -> None:
    generation = store.create_staging()
    store.activate(generation, _manifest(generation.name))
    generation.close()

    with pytest.raises(VectorStoreError, match="holds collection"):
        ChromaIndexStore(store.persist_dir, "test").open_active()


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{not json", "not valid JSON"),
        ("[]", "not a JSON object"),
        ('{"format": "other"}', "unknown format"),
        ('{"format": "local-code-rag/index-manifest", "schema_version": 2}', "schema version 2"),
    ],
    ids=["json", "object", "format", "schema"],
)
def test_an_unusable_manifest_is_never_repaired(
    store: ChromaIndexStore, text: str, match: str
) -> None:
    store.persist_dir.mkdir(mode=0o700)
    (store.persist_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")

    with pytest.raises(VectorStoreError, match=match):
        store.read_manifest()


def test_the_manifest_round_trips_and_is_written_atomically(tmp_path: Path) -> None:
    manifest = _manifest("gen-20261127T000000000000Z-0000abcd")

    write_manifest(tmp_path, manifest)

    assert read_manifest(tmp_path) == manifest
    assert parse_manifest(format_manifest(manifest), tmp_path) == manifest
    assert stat.S_IMODE((tmp_path / MANIFEST_NAME).stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST_NAME]


def test_the_manifest_rejects_extra_or_missing_keys(tmp_path: Path) -> None:
    data = json.loads(format_manifest(_manifest("gen-20261127T000000000000Z-0000abcd")))
    data["extra"] = 1

    with pytest.raises(VectorStoreError, match="unexpected keys"):
        parse_manifest(json.dumps(data), tmp_path)


# --- permissions, safe deletion, lock ------------------------------------------------------------


def test_the_root_is_private_whatever_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        generation = ChromaIndexStore(tmp_path / "vs", "code").create_staging()
        generation.close()
    finally:
        os.umask(previous)

    assert stat.S_IMODE((tmp_path / "vs").stat().st_mode) == 0o700
    assert stat.S_IMODE(generation.path.stat().st_mode) == 0o700


def test_a_permissive_root_is_fixed_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "vs"
    root.mkdir()
    root.chmod(0o755)

    with ChromaIndexStore(root, "code").lock():
        pass

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert "vector_store.permissions_fixed" in caplog.text


def test_only_this_projects_generations_are_ever_deleted(
    store: ChromaIndexStore, tmp_path: Path
) -> None:
    ours = store.create_staging()
    ours.close()
    keep = store.create_staging()
    keep.close()
    root = store.persist_dir
    (root / "notes").mkdir()
    (root / "gen-20261127T000000000000Z-deadbeef").mkdir()  # right name, no marker
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / vs.MARKER_NAME).write_text("local-code-rag index generation\n", encoding="ascii")
    (root / "gen-20261127T000000000001Z-cafebabe").symlink_to(outside, target_is_directory=True)

    removed = store.discard_leftovers(frozenset({keep.name}))

    assert removed == [ours.name]
    assert sorted(p.name for p in root.iterdir() if p.name not in {vs.LOCK_NAME}) == sorted(
        [
            keep.name,
            "notes",
            "gen-20261127T000000000000Z-deadbeef",
            "gen-20261127T000000000001Z-cafebabe",
        ]
    )
    assert outside.exists()
    with pytest.raises(VectorStoreError, match="not a generation of this project"):
        store._delete("gen-20261127T000000000000Z-deadbeef")


def test_a_discarded_generation_leaves_no_trace(store: ChromaIndexStore) -> None:
    generation = store.create_staging()
    generation.add(_with_vectors(_nodes(), FakeEmbedding()))

    store.discard(generation)

    assert not generation.path.exists()
    for path in store.persist_dir.rglob("*"):
        if path.is_file():
            assert SENTINEL.encode() not in path.read_bytes()


def test_a_second_build_cannot_take_the_lock(store: ChromaIndexStore) -> None:
    second = ChromaIndexStore(store.persist_dir, "code")
    with store.lock(), pytest.raises(IndexLockedError, match="holds the lock"):  # noqa: SIM117
        with second.lock():
            pass
    with store.lock():  # released
        pass


# --- the contract entry point ------------------------------------------------------------------


def test_get_vector_store_is_a_handle_built_without_io(tmp_path: Path) -> None:
    store = get_vector_store(persist_dir=tmp_path / "vs", collection="test")

    assert (store.persist_dir, store.collection_name) == (tmp_path / "vs", "test")
    assert not (tmp_path / "vs").exists()


def test_get_vector_store_defaults_come_from_configuration() -> None:
    from src import config

    store = get_vector_store()

    assert (store.persist_dir, store.collection_name) == (
        config.VECTOR_STORE_DIR,
        config.COLLECTION_NAME,
    )


@pytest.mark.parametrize("name", ["ab", "-x-", "a b", "c" * 513])
def test_invalid_collection_names_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(VectorStoreError, match="collection name"):
        ChromaIndexStore(tmp_path, name)


def test_the_store_and_the_configuration_share_chromas_naming_rule() -> None:
    from src import config

    assert vs.COLLECTION_NAME_RE.pattern == config._COLLECTION_NAME_RE.pattern


# --- a fresh interpreter with a hostile .env and environment stays local -----------------------

_HOSTILE_CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path[:0] = [repo_root, repo_root + "/tests"]
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from fake_embedding import FakeEmbedding
        from llama_index.core.schema import MetadataMode, TextNode
        from src.indexing.vector_store import ChromaIndexStore

        store = ChromaIndexStore(work / "vs", "code")
        generation = store.create_staging()
        node = TextNode(
            id_="n1", text="def f():\\n    return 1\\n", metadata={"language": "python"}
        )
        node.embedding = FakeEmbedding().vector(node.get_content(MetadataMode.EMBED))
        generation.add([node])
        effective = generation._client.get_settings()
        count = generation.count()
        generation.close()
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "api_impl": effective.chroma_api_impl,
        "host": effective.chroma_server_host,
        "count": count,
        "onnxruntime": "onnxruntime" in sys.modules,
        "home": sorted(p.name for p in (work / "home").iterdir()),
    }))
    """
)


def test_a_fresh_interpreter_with_a_hostile_env_stays_local(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "home").mkdir(parents=True)
    (work / ".env").write_text(
        "CHROMA_API_IMPL=chromadb.api.fastapi.FastAPI\nCHROMA_SERVER_HOST=203.0.113.10\n"
        "CHROMA_SERVER_HTTP_PORT=8000\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work / "home"),
        "XDG_CACHE_HOME": str(work / "home"),
        "TMPDIR": str(work / "home"),
        "CHROMA_API_IMPL": "chromadb.api.fastapi.FastAPI",
        "CHROMA_SERVER_HOST": "203.0.113.10",
        "CHROMA_OTEL_COLLECTION_ENDPOINT": "http://203.0.113.10:4317",
        "HTTPS_PROXY": "http://203.0.113.10:3128",
        "LC_ALL": "C.UTF-8",
    }

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _HOSTILE_CHILD, str(REPO_ROOT), str(work)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert (report["api_impl"], report["host"]) == ("chromadb.api.rust.RustBindingsAPI", None)
    assert (report["count"], report["onnxruntime"], report["home"]) == (1, False, [])


# --- static audit: one factory, no unsafe Chroma calls anywhere in src/ ------------------------

_CLIENT_CONSTRUCTORS = {
    "PersistentClient",
    "Client",
    "HttpClient",
    "AsyncHttpClient",
    "EphemeralClient",
    "CloudClient",
    "RustClient",
}
_FORBIDDEN_ATTRIBUTES = {"get_or_create_collection", "from_params"}


def _findings(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    factory_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "open_chroma_client":
            factory_lines = {n.lineno for n in ast.walk(node) if hasattr(n, "lineno")}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            found.append(f"{filename}:{node.lineno}: {node.attr}")
        if isinstance(node, ast.keyword) and node.arg == "query_texts":
            found.append(f"{filename}:{node.value.lineno}: query_texts")
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "chromadb"
            and name in _CLIENT_CONSTRUCTORS
            and node.lineno not in factory_lines
        ):
            found.append(f"{filename}:{node.lineno}: chromadb.{name} outside the factory")
        if name == "ChromaVectorStore" and "chroma_collection" not in {
            k.arg for k in node.keywords
        }:
            found.append(f"{filename}:{node.lineno}: ChromaVectorStore without chroma_collection")
    return found


def test_src_creates_chroma_clients_only_in_the_factory_and_never_calls_unsafe_apis() -> None:
    findings = [
        finding
        for path in sorted(SRC_DIR.rglob("*.py"))
        for finding in _findings(
            path.read_text(encoding="utf-8"), path.relative_to(REPO_ROOT).as_posix()
        )
    ]

    assert findings == []


def test_the_static_audit_can_fail() -> None:
    source = textwrap.dedent(
        """
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore
        client = chromadb.PersistentClient(path="x")
        collection = client.get_or_create_collection("c")
        collection.query(query_texts=["q"])
        store = ChromaVectorStore(collection_name="c")
        other = ChromaVectorStore.from_params(collection_name="c")
        """
    )

    assert len(_findings(source, "bad.py")) == 5


# --- exhaustive search (the Step 2 fix of 2026-11-27) -----------------------------------------
# Below Chroma's sync_threshold (1000) the HNSW graph is never written to disk: every process
# rebuilds it, and with the default ef_search (100) the misses differed from run to run.

SYNTH_N, SYNTH_DIM, SYNTH_K, SYNTH_QUERIES = 658, 256, 20, 20

_QUERY_PROCESS = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    sys.path.insert(0, sys.argv[1])
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from src.indexing.vector_store import open_chroma_client

        queries = json.loads(sys.stdin.read())
        client = open_chroma_client(Path(sys.argv[2]))
        collection = client.get_collection("code", embedding_function=None)
        found = collection.query(query_embeddings=queries, n_results=int(sys.argv[3]), include=[])
        client.close()
    print(json.dumps({"violations": len(guard.violations), "ids": found["ids"]}))
    """
)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def _synthetic() -> tuple[list[list[float]], list[list[float]]]:
    """Clustered unit vectors, like embeddings: the shape that showed per-process misses."""
    rng = random.Random(1234)  # noqa: S311 (seeded test data, not cryptography)
    centers = [[rng.gauss(0, 1) for _ in range(SYNTH_DIM)] for _ in range(25)]
    points = [_unit([c + rng.gauss(0, 0.9) for c in centers[i % 25]]) for i in range(SYNTH_N)]
    queries = [
        _unit([x + rng.gauss(0, 0.5) for x in points[rng.randrange(SYNTH_N)]])
        for _ in range(SYNTH_QUERIES)
    ]
    return points, queries


def _exact(points: list[list[float]], query: list[float]) -> list[str]:
    scores = [sum(a * b for a, b in zip(query, p, strict=True)) for p in points]
    order = sorted(range(len(points)), key=lambda i: (-scores[i], i))
    return [f"n{i:04d}" for i in order[:SYNTH_K]]


def _query_in_a_new_process(path: Path, queries: list[list[float]], home: Path) -> list[list[str]]:
    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _QUERY_PROCESS, str(REPO_ROOT), str(path), str(SYNTH_K)],
        input=json.dumps(queries),
        cwd=home,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(home), "LC_ALL": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["violations"] == 0
    ids: list[list[str]] = report["ids"]
    return ids


def test_an_exhaustive_generation_is_exact_and_identical_in_every_new_process(
    store: ChromaIndexStore, tmp_path: Path
) -> None:
    points, queries = _synthetic()
    staging = store.create_staging()
    try:
        staging.add(
            [TextNode(id_=f"n{i:04d}", text=f"node {i}", embedding=p) for i, p in enumerate(points)]
        )
        assert staging.search_breadth() < SYNTH_N  # Chroma's default: an approximate search
        assert staging.make_search_exhaustive() == max(vs.SEARCH_EF_FLOOR, SYNTH_N)
        path = staging.path
    finally:
        staging.close()
    exact = [_exact(points, q) for q in queries]

    first = _query_in_a_new_process(path, queries, tmp_path)
    second = _query_in_a_new_process(path, queries, tmp_path)

    assert first == exact
    assert second == exact
