"""Phase 3 key test (decision P8, Gate 1 Q-H): index the frozen corpus and prove it.

Builds the whole corpus into a temporary ``VECTOR_STORE_DIR`` (never the owner's index)
with the real local model, twice, and checks:

* 658 vectors whose node digest is the Phase 2 manifest's (existing constants);
* the second build reuses all 658 and embeds none, with identical IDs;
* a new process loads the index with zero document embeddings, verified (P5);
* Chroma's HNSW top-20 equals an exact cosine kNN for generic queries written for this
  test (not from the corpus), and cosine = 1 + ln(score);
* the corpus tree is unchanged and nothing tried to leave the machine.

It skips only when the corpus is absent or its HEAD is not the frozen commit (a dirty
corpus fails), and a skip is never evidence. Assertions compare counts, IDs and digests,
never corpus text. Never run with ``-vv``, ``-l`` or ``--showlocals``. About 6 minutes:
``make test-index-corpus`` (excluded from ``make test-integration``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
from chunk_manifest import MANIFEST_PATH, parse_manifest
from chunk_validation import Snapshot, tree_snapshot
from corpus_manifest import corpus_skip_reason
from retrieval_oracle import disagreements, exact_ranking, hnsw_ranking

from src import config
from src.indexing.build_index import IndexReport, build_index_with_report, load_index_with_report
from src.indexing.embeddings import get_embedding_model
from src.indexing.vector_store import ChromaIndexStore
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_repository
from src.network_guard import NetworkGuard, NetworkViolation

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_NODES = 658
# Generic questions written for this test, about any software project (not the corpus).
GENERIC_QUERIES = (
    "how is the configuration loaded?",
    "where are the tests?",
    "how are errors handled?",
    "how is logging set up?",
    "¿cómo se procesan los datos?",
)


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    root: Path
    store_dir: Path
    first: IndexReport
    second: IndexReport
    first_ids: frozenset[str]
    second_ids: frozenset[str]
    chunk_ids: frozenset[str]
    oracle_problems: tuple[str, ...]
    cosine_formula_problems: int
    before: Snapshot
    after: Snapshot
    violations: tuple[NetworkViolation, ...]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> CorpusIndex:
    repo = config.REPO_PATH
    reason = corpus_skip_reason(repo)
    if reason is not None:
        pytest.skip(reason)
    store_dir = tmp_path_factory.mktemp("corpus-index") / "vs"
    before = tree_snapshot(repo)
    with NetworkGuard() as guard:  # a module fixture runs before the per-test guard
        store = ChromaIndexStore(store_dir, "code")
        first = build_index_with_report(repo, store).report
        first_ids = _ids(store)
        second = build_index_with_report(repo, store).report
        second_ids = _ids(store)
        nodes = chunk_documents(load_repository(repo))
        loaded = load_index_with_report(nodes, vector_store=store, repo_path=repo)
        try:
            entries = list(loaded._generation.entries())
            vector_store = loaded._generation.vector_store()
            embedder = get_embedding_model()
            problems: list[str] = []
            formula = 0
            for position, query in enumerate(GENERIC_QUERIES, start=1):
                vector = embedder.get_query_embedding(query)
                exact = exact_ranking(entries, vector, 20)
                hnsw = hnsw_ranking(vector_store, vector, 20)
                problems += [f"query {position}: {p}" for p in disagreements(exact, hnsw)]
                formula += sum(1 for _, cosine, score in hnsw if score <= 0 or cosine > 1 + 1e-6)
        finally:
            loaded.close()
    return CorpusIndex(
        repo,
        store_dir,
        first,
        second,
        first_ids,
        second_ids,
        frozenset(n.node_id for n in nodes),
        tuple(problems),
        formula,
        before,
        tree_snapshot(repo),
        guard.violations,
    )


def _ids(store: ChromaIndexStore) -> frozenset[str]:
    _, active = store.open_active()
    try:
        return frozenset(node.node_id for node, _ in active.entries())
    finally:
        active.close()


def test_the_corpus_is_indexed_with_the_phase2_digest(corpus: CorpusIndex) -> None:
    expected = parse_manifest(MANIFEST_PATH.read_bytes())["digest"]
    assert isinstance(expected, dict)

    assert (corpus.first.documents, corpus.first.nodes) == (88, CORPUS_NODES)
    assert (corpus.first.embedded, corpus.first.reused) == (CORPUS_NODES, 0)
    assert corpus.first.node_digest == expected["corpus"]
    assert corpus.first.embedder.verified
    assert corpus.first_ids == corpus.chunk_ids


def test_a_second_build_reuses_everything_with_identical_ids(corpus: CorpusIndex) -> None:
    assert (corpus.second.reused, corpus.second.embedded, corpus.second.removed) == (
        CORPUS_NODES,
        0,
        0,
    )
    assert corpus.second_ids == corpus.first_ids
    assert corpus.second.node_digest == corpus.first.node_digest
    assert corpus.second.embed_digest == corpus.first.embed_digest


def test_hnsw_equals_an_exact_cosine_knn_on_generic_queries(corpus: CorpusIndex) -> None:
    assert corpus.oracle_problems == ()
    assert corpus.cosine_formula_problems == 0


_NEW_PROCESS = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, store_dir, repo = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    sys.path.insert(0, repo_root)
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from src.indexing import embeddings
        from src.indexing.build_index import load_index_with_report
        from src.indexing.vector_store import ChromaIndexStore
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_repository

        documents_embedded = []
        original = embeddings.OllamaEmbeddingAdapter._get_text_embeddings

        def spy(self, texts):
            documents_embedded.append(len(texts))
            return original(self, texts)

        embeddings.OllamaEmbeddingAdapter._get_text_embeddings = spy
        nodes = chunk_documents(load_repository(repo))
        loaded = load_index_with_report(
            nodes, vector_store=ChromaIndexStore(store_dir, "code"), repo_path=repo
        )
        hits = loaded.index.as_retriever(similarity_top_k=3).retrieve("how is logging set up?")
        result = {"nodes": loaded.manifest.nodes, "drift_checked": loaded.drift_checked}
        loaded.close()
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "documents_embedded": sum(documents_embedded),
        "hits": len(hits),
        **result,
    }))
    """
)


def test_a_new_process_loads_without_embedding_any_document(
    corpus: CorpusIndex, tmp_path: Path
) -> None:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LC_ALL": "C.UTF-8"}

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _NEW_PROCESS,
            str(REPO_ROOT),
            str(corpus.store_dir),
            str(corpus.root),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert result.returncode == 0, "the loading subprocess failed (see its exit code)"
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert (report["documents_embedded"], report["nodes"], report["drift_checked"]) == (
        0,
        CORPUS_NODES,
        True,
    )
    assert report["hits"] == 3


def test_the_corpus_is_unchanged_and_nothing_left_the_machine(corpus: CorpusIndex) -> None:
    assert corpus.violations == ()
    assert corpus.after == corpus.before
