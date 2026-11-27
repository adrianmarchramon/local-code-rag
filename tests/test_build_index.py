"""Phase 3, Step 4: the indexing pipeline (fake embedder, real Chroma in ``tmp_path``).

Covers the end-to-end run, reuse, re-indexing after every kind of file change, atomicity
under injected failures (KeyboardInterrupt included), repository replacement, the Phase
4-5 layout with the store inside the repository, repository identity, content-free reports
and logs, and the CLI. No model and no external network.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from fake_embedding import FakeEmbedding
from llama_index.core import MockEmbedding
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from src import config
from src.errors import (
    ConfigurationError,
    EmbeddingInputTooLongError,
    IndexBuildError,
    IndexLockedError,
    NoDocumentsError,
    OllamaUnavailableError,
    VectorStoreError,
)
from src.indexing import build_index as bi
from src.indexing.build_index import (
    ExitCode,
    build_index,
    build_index_with_report,
    exit_code_for,
    repo_identity,
)
from src.indexing.manifest import MANIFEST_NAME, read_manifest
from src.indexing.vector_store import GENERATION_RE, ChromaIndexStore, get_vector_store
from src.ingestion.chunker import chunk_documents, chunk_documents_with_report
from src.ingestion.digest import nodes_digest
from src.ingestion.loader import load_repository

SENTINEL = "SENTINEL_IN_CODE_4d71"
FILES: dict[str, str] = {
    "src/auth.py": (
        "import hmac\n\n\n"
        "def verify_token(token, key):\n"
        '    """Check the signature of an access token."""\n'
        f"    return hmac.compare_digest(token, key)  # {SENTINEL}\n\n\n"
        "def revoke_token(token):\n    return None\n"
    ),
    "src/csv_tools.py": (
        "import csv\n\n\ndef read_rows(path):\n"
        '    """Read a CSV file into dictionaries."""\n'
        "    with open(path) as handle:\n        return list(csv.DictReader(handle))\n"
    ),
    "src/retry.py": (
        "import time\n\n\ndef retry(call, attempts=3):\n"
        "    for attempt in range(attempts):\n        try:\n            return call()\n"
        "        except ConnectionError:\n            time.sleep(2 ** attempt)\n"
    ),
    "docs/guide.md": "# Guide\n\nHow access tokens work.\n\n## Install\n\nRun make install.\n",
    "config/app.yaml": "server:\n  port: 8080\n  host: localhost\n",
}


# --- helpers -----------------------------------------------------------------------------------


def _write_repo(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def rules(tmp_path: Path) -> Path:
    path = tmp_path / "ingest.ignore"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _write_repo(tmp_path / "repo", FILES)


@pytest.fixture
def store(tmp_path: Path) -> ChromaIndexStore:
    return ChromaIndexStore(tmp_path / "vs", "code")


def _fresh_nodes(repo: Path, rules: Path) -> list[BaseNode]:
    return chunk_documents(load_repository(repo, rules_file=rules))


def _stored_ids(store: ChromaIndexStore) -> set[str]:
    _, active = store.open_active()
    try:
        return {node.node_id for node, _ in active.entries()}
    finally:
        active.close()


def _generations(store: ChromaIndexStore) -> list[str]:
    return sorted(p.name for p in store.persist_dir.iterdir() if GENERATION_RE.fullmatch(p.name))


def _build(
    repo: Path,
    store: ChromaIndexStore,
    rules: Path,
    fake: FakeEmbedding | None = None,
    **kw: object,
) -> bi.BuildResult:
    return build_index_with_report(
        repo,
        store,
        fake if fake is not None else FakeEmbedding(),
        rules_file=rules,
        **kw,  # type: ignore[arg-type]
    )


# --- end to end ----------------------------------------------------------------------------


def test_a_repository_is_indexed_end_to_end(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    fake = FakeEmbedding()
    result = _build(repo, store, rules, fake)
    nodes = _fresh_nodes(repo, rules)

    report = result.report
    assert (report.documents, report.nodes) == (5, len(nodes))
    assert (report.reused, report.embedded, report.removed) == (0, len(nodes), 0)
    assert report.node_digest == nodes_digest(nodes) == result.manifest.node_digest
    assert result.manifest.embedder.verified
    assert result.manifest.metric == "cosine"
    assert fake.texts_embedded == len(nodes)
    assert _stored_ids(store) == {n.node_id for n in nodes}
    assert read_manifest(store.persist_dir) == result.manifest
    assert _generations(store) == [result.report.generation]


def test_the_returned_index_retrieves_and_filters_by_language(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    index = _build(repo, store, rules).index

    top = index.as_retriever(similarity_top_k=1).retrieve("verify the access token signature")
    filters = MetadataFilters(
        filters=[MetadataFilter(key="language", value="markdown", operator=FilterOperator.EQ)]
    )
    markdown = index.as_retriever(similarity_top_k=10, filters=filters).retrieve("tokens")

    assert top[0].node.metadata["file_path"] == "src/auth.py"
    assert markdown
    assert {hit.node.metadata["language"] for hit in markdown} == {"markdown"}


def test_the_contract_function_returns_the_index(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    index = build_index(repo, store, FakeEmbedding(), rules_file=rules)

    assert index.as_retriever(similarity_top_k=3).retrieve("csv rows")


# --- reuse and re-indexing -----------------------------------------------------------------


def test_a_second_run_embeds_nothing_and_leaves_one_generation(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    first = _build(repo, store, rules)
    fake = FakeEmbedding()

    second = _build(repo, store, rules, fake)

    assert fake.texts_embedded == 0
    assert (second.report.reused, second.report.embedded, second.report.removed) == (
        first.report.nodes,
        0,
        0,
    )
    assert second.report.node_digest == first.report.node_digest
    assert second.report.previous_generation == first.report.generation
    assert _generations(store) == [second.report.generation]
    assert _stored_ids(store) == {n.node_id for n in _fresh_nodes(repo, rules)}


@pytest.mark.parametrize(
    "change",
    ["modify", "delete", "rename", "shift_lines", "same_id_new_text"],
)
def test_after_any_file_change_only_what_changed_is_embedded_and_nothing_stale_survives(
    repo: Path, store: ChromaIndexStore, rules: Path, change: str
) -> None:
    before = {(n.node_id, bi.embed_string_sha256(n)) for n in _fresh_nodes(repo, rules)}
    _build(repo, store, rules)
    auth = repo / "src" / "auth.py"
    if change == "modify":
        auth.write_text(auth.read_text(encoding="utf-8") + "\n\ndef added():\n    return 2\n")
    elif change == "delete":
        (repo / "src" / "retry.py").unlink()
    elif change == "rename":
        (repo / "src" / "csv_tools.py").rename(repo / "src" / "tables.py")
    elif change == "shift_lines":
        auth.write_text("# a new header line\n\n" + auth.read_text(encoding="utf-8"))
    else:  # same offsets, different text: the node ID stays, the EMBED string changes
        text = auth.read_text(encoding="utf-8")
        auth.write_text(text.replace("return None", "return True"), encoding="utf-8")
    after_nodes = _fresh_nodes(repo, rules)
    after = {(n.node_id, bi.embed_string_sha256(n)) for n in after_nodes}
    fake = FakeEmbedding()

    report = _build(repo, store, rules, fake).report

    assert report.embedded == len(after - before) == fake.texts_embedded
    assert report.reused == len(after & before)
    assert _stored_ids(store) == {n.node_id for n in after_nodes}
    assert report.removed == len({i for i, _ in before} - {n.node_id for n in after_nodes})
    if change == "same_id_new_text":
        assert {i for i, _ in after - before} <= {i for i, _ in before}  # same ID, re-embedded


@pytest.mark.parametrize("what", ["model_digest", "recipe"])
def test_a_new_model_or_recipe_reuses_nothing(
    repo: Path, store: ChromaIndexStore, rules: Path, what: str
) -> None:
    first = _build(repo, store, rules)
    fake = FakeEmbedding(digest="e" * 64) if what == "model_digest" else FakeEmbedding(dimension=32)

    report = _build(repo, store, rules, fake).report

    assert (report.reused, report.embedded) == (0, first.report.nodes)


def test_an_unverified_embedder_is_indexed_but_never_reused(
    repo: Path, store: ChromaIndexStore, rules: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    first = _build(repo, store, rules, MockEmbedding(embed_dim=8))  # type: ignore[arg-type]
    second = _build(repo, store, rules, MockEmbedding(embed_dim=8))  # type: ignore[arg-type]

    assert not first.manifest.embedder.verified
    assert first.manifest.embedder.dimension == 8
    assert first.manifest.recipe is None
    assert (second.report.reused, second.report.embedded) == (0, first.report.nodes)
    assert "index.embedder_unverified" in caplog.text


# --- atomicity: any failure before activation leaves the previous index serving -------------


def _fail_embedding(fake: FakeEmbedding, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.fail_on_call(1, OllamaUnavailableError("down"))


def _fail_interrupt(fake: FakeEmbedding, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.fail_on_call(1, KeyboardInterrupt())


def _fail_writing(fake: FakeEmbedding, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(bi, "_write", broken)


def _fail_verification(fake: FakeEmbedding, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise IndexBuildError("verification failed")

    monkeypatch.setattr(bi, "_verify", broken)


def _fail_manifest(fake: FakeEmbedding, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("cannot write the manifest")

    monkeypatch.setattr("src.indexing.vector_store.write_manifest", broken)


@pytest.mark.parametrize(
    ("inject", "error"),
    [
        (_fail_embedding, OllamaUnavailableError),
        (_fail_interrupt, KeyboardInterrupt),
        (_fail_writing, OSError),
        (_fail_verification, IndexBuildError),
        (_fail_manifest, OSError),
    ],
    ids=["embedding", "interrupt", "writing", "verification", "manifest"],
)
def test_a_failure_before_activation_keeps_the_previous_index(
    repo: Path,
    store: ChromaIndexStore,
    rules: Path,
    monkeypatch: pytest.MonkeyPatch,
    inject: Callable[[FakeEmbedding, pytest.MonkeyPatch], None],
    error: type[BaseException],
) -> None:
    first = _build(repo, store, rules)
    first_ids = _stored_ids(store)
    manifest_bytes = (store.persist_dir / MANIFEST_NAME).read_bytes()
    (repo / "src" / "new.py").write_text("def brand_new():\n    return 3\n", encoding="utf-8")
    fake = FakeEmbedding()
    inject(fake, monkeypatch)

    with pytest.raises(error):
        _build(repo, store, rules, fake)

    assert (store.persist_dir / MANIFEST_NAME).read_bytes() == manifest_bytes
    assert _generations(store) == [first.report.generation]  # staging discarded
    assert _stored_ids(store) == first_ids  # the previous index still serves, unchanged
    monkeypatch.undo()
    recovered = _build(repo, store, rules)
    assert _generations(store) == [recovered.report.generation]
    assert recovered.report.nodes == first.report.nodes + 1


def test_leftovers_of_a_crashed_run_are_removed_by_the_next_one(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    first = _build(repo, store, rules)
    crashed = store.create_staging()  # as if the process had died mid-run
    crashed.close()

    second = _build(repo, store, rules)

    assert crashed.name not in _generations(store)
    assert _generations(store) == [second.report.generation]
    assert second.report.previous_generation == first.report.generation


def test_an_unreadable_manifest_is_replaced_without_deleting_first(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    first = _build(repo, store, rules)
    (store.persist_dir / MANIFEST_NAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(VectorStoreError):
        store.open_active()

    second = _build(repo, store, rules)

    assert (second.report.reused, second.report.previous_generation) == (0, None)
    assert _generations(store) == [second.report.generation]  # the old one, after activation
    assert first.report.generation != second.report.generation


# --- inputs that must fail without touching the index ---------------------------------------


def test_an_empty_repository_fails_and_keeps_the_index(
    repo: Path, store: ChromaIndexStore, rules: Path, tmp_path: Path
) -> None:
    _build(repo, store, rules)
    manifest_bytes = (store.persist_dir / MANIFEST_NAME).read_bytes()
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(NoDocumentsError):
        _build(empty, store, rules)
    assert (store.persist_dir / MANIFEST_NAME).read_bytes() == manifest_bytes


def test_documents_from_inside_the_store_are_refused(tmp_path: Path, rules: Path) -> None:
    repo = _write_repo(tmp_path / "repo", {"main.py": "def f():\n    return 1\n"})
    store = ChromaIndexStore(repo / "vs", "code")
    _write_repo(repo, {"vs/planted.py": "def planted():\n    return 0\n"})

    with pytest.raises(IndexBuildError, match="inside the vector store"):
        _build(repo, store, rules)


def test_repeated_node_ids_are_refused(
    repo: Path, store: ChromaIndexStore, rules: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = chunk_documents_with_report

    def duplicated(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs)  # type: ignore[arg-type]
        return type(result)(
            result.nodes + result.nodes[:1], result.spans + result.spans[:1], result.report
        )

    monkeypatch.setattr(bi, "chunk_documents_with_report", duplicated)

    with pytest.raises(IndexBuildError, match="repeated"):
        _build(repo, store, rules)
    assert not store.persist_dir.joinpath(MANIFEST_NAME).exists()


def test_a_held_lock_refuses_a_second_build(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    with store.lock(), pytest.raises(IndexLockedError):
        _build(repo, store, rules)


def test_a_store_that_is_not_a_location_handle_is_refused(repo: Path, rules: Path) -> None:
    with pytest.raises(IndexBuildError, match="location handle"):
        build_index_with_report(repo, object(), FakeEmbedding(), rules_file=rules)  # type: ignore[arg-type]


def test_an_over_long_node_is_named_by_path_and_lines_not_text(
    repo: Path, store: ChromaIndexStore, rules: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEmbedding()
    fake.fail_on_call(1, EmbeddingInputTooLongError("too long", index=0, length=99))

    with pytest.raises(EmbeddingInputTooLongError) as raised:
        _build(repo, store, rules, fake)

    message = str(raised.value)
    assert ".md:" in message or ".py:" in message or ".yaml:" in message
    assert SENTINEL not in message


# --- writes larger than one batch, replacement by another repository ------------------------


def test_more_nodes_than_one_write_batch(repo: Path, store: ChromaIndexStore, rules: Path) -> None:
    report = _build(repo, store, rules, write_batch_size=2).report

    assert len(_stored_ids(store)) == report.nodes > 2


def test_another_repository_replaces_the_first_without_leftovers(
    repo: Path, store: ChromaIndexStore, rules: Path, tmp_path: Path
) -> None:
    _build(repo, store, rules)
    other = _write_repo(tmp_path / "other", {"lib/only.py": "def only():\n    return 1\n"})

    report = _build(other, store, rules).report

    assert _stored_ids(store) == {n.node_id for n in _fresh_nodes(other, rules)}
    assert report.repo.path == str(other.resolve())
    assert report.removed == len({n.node_id for n in _fresh_nodes(repo, rules)})


# --- the Phase 4-5 layout: the store lives inside the indexed repository -------------------


def test_the_phase4_layout_indexes_twice_and_never_itself(repo: Path) -> None:
    store = get_vector_store(persist_dir=repo / "vs", collection="test")

    first = build_index_with_report(repo, vector_store=store, embed_model=FakeEmbedding())
    second = build_index_with_report(repo, vector_store=store, embed_model=FakeEmbedding())

    ids = _stored_ids(store)
    assert first.report.node_digest == second.report.node_digest
    assert (second.report.embedded, len(ids)) == (0, first.report.nodes)
    assert not any(
        str(n.metadata["file_path"]).startswith("vs/")
        for n in _fresh_nodes(repo, config.INGEST_IGNORE_FILE)
    )


# --- repository identity ---------------------------------------------------------------------


def test_repository_identity_in_its_three_states(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    git = tmp_path / "git"
    (git / ".git").mkdir(parents=True)
    (git / ".git" / "HEAD").write_text("a" * 40 + "\n", encoding="ascii")
    broken = tmp_path / "broken"
    (broken / ".git").mkdir(parents=True)
    (broken / ".git" / "HEAD").write_text("garbage\n", encoding="ascii")

    assert repo_identity(plain).state == "not_a_git_repo"
    assert (repo_identity(git).state, repo_identity(git).head) == ("git", "a" * 40)
    unresolvable = repo_identity(broken)
    assert (unresolvable.state, unresolvable.head) == ("head_unresolvable", None)
    assert unresolvable.reason is not None
    assert "neither a SHA" in unresolvable.reason


# --- no content in reports, manifest or logs -------------------------------------------------


def test_reports_manifest_and_logs_never_carry_code(
    repo: Path, store: ChromaIndexStore, rules: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = _build(repo, store, rules)
    result.index.as_retriever(similarity_top_k=3).retrieve("token")

    assert SENTINEL not in json.dumps(result.report.to_dict())
    assert SENTINEL not in result.report.render_text()
    assert SENTINEL not in (store.persist_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    assert SENTINEL not in caplog.text
    assert "index.done" in caplog.text
    assert "index.embedding_progress" in caplog.text


# --- CLI -------------------------------------------------------------------------------------


def _cli(argv: list[str], monkeypatch: pytest.MonkeyPatch, store_dir: Path) -> tuple[int, str]:
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", store_dir)
    monkeypatch.setattr(bi, "get_embedding_model", FakeEmbedding)
    out = StringIO()
    code = bi.main(argv, stdout=out)
    return code, out.getvalue()


def test_the_cli_indexes_and_reports_json(
    repo: Path, rules: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli(
        ["--repo", str(repo), "--rules-file", str(rules), "--json"],
        monkeypatch,
        tmp_path / "cli-vs",
    )

    report = json.loads(output)
    assert code == ExitCode.OK
    assert (report["documents"], report["embedded"], report["reused"]) == (5, report["nodes"], 0)
    assert SENTINEL not in output


def test_the_cli_maps_failures_to_exit_codes(
    tmp_path: Path, rules: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    code, output = _cli(
        ["--repo", str(empty), "--rules-file", str(rules)], monkeypatch, tmp_path / "vs"
    )

    assert code == ExitCode.NO_DOCUMENTS
    assert "RESULT: FAIL (exit 21 NO_DOCUMENTS)" in output


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigurationError("x"), 10),
        (OllamaUnavailableError("x"), 11),
        (EmbeddingInputTooLongError("x"), 13),
        (NoDocumentsError("x"), 21),
        (IndexLockedError("x"), 30),
        (IndexBuildError("x"), 31),
        (VectorStoreError("x"), 32),
    ],
)
def test_exit_codes(error: Exception, code: int) -> None:
    assert exit_code_for(error) == code  # type: ignore[arg-type]


def test_the_store_directory_holds_only_known_entries(
    repo: Path, store: ChromaIndexStore, rules: Path
) -> None:
    _build(repo, store, rules)
    names = sorted(p.name for p in store.persist_dir.iterdir())

    assert names[0] == ".lock"
    assert MANIFEST_NAME in names
    assert len([n for n in names if GENERATION_RE.fullmatch(n)]) == 1
    shutil.rmtree(store.persist_dir)  # the index is disposable: a rebuild recreates it
    assert _build(repo, store, rules).report.reused == 0
