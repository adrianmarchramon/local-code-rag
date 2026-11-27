"""Phase 3, Step 4: the indexing pipeline, atomic and repeatable.

``build_index(repo_path, vector_store=None, embed_model=None) -> VectorStoreIndex`` (the
contract with Phases 4-7) runs, in this order:

1. take the store lock (``IndexLockedError`` if another build holds it); remove this
   project's leftovers from crashed runs, unless the manifest is unreadable (then nothing
   is deleted until a new index is active);
2. load and chunk the repository with the Phase 1-2 public API: zero documents raise
   without touching the index; no document may come from inside the store; node IDs must
   be unique (Chroma silently ignores a repeated ID);
3. compute the repository identity (decision P7), the Phase 2 node digest and the sha256
   of every EMBED string (exactly what is embedded, after the recipe's prefix);
4. verify the embedder's identity (model, digest, recipe) before creating anything;
5. reuse a stored vector only when recipe fingerprint and model digest match the active
   manifest and, per node, the ID and the EMBED-string sha256 match (decision P4);
   nothing is ever added to ``node.metadata`` (it would change the node hash);
6. create a staging generation; embed only what was not reused, in batches, logging
   progress without content; write in batches Chroma accepts;
7. verify the staging generation: count, ID set, metric, dimension, valid vectors, and
   the hash of every reconstructed node;
8. check the model digest again (a ``pull`` during the run would mix vector spaces);
9. activate: write the manifest, atomically (``src.indexing.manifest``);
10. discard the previous generation (a failure is a WARNING; the next run retries).

Any exception before step 9, ``KeyboardInterrupt`` included, discards the staging
generation and leaves the previous index active. Staging only ever holds the current
nodes, so nothing from a modified, renamed or deleted file survives (the Phase 2 handoff's
"delete by ref_doc_id before re-indexing", by construction).

The returned index reads the generation just activated, with the same ``embed_model``; it
stays valid until the next activation deletes that generation.

CLI: ``python -m src.indexing.build_index [--repo PATH] [--rules-file PATH] [--json]``
(``make index``), under :class:`NetworkGuard`; it prints counts, digests and paths, never
code. Exit codes: 0 OK; 1 unexpected; 10 configuration (or no recipe for the model); 11
Ollama unavailable; 12 model missing; 13 verification failed (unverified model, invalid
vector, input too long); 14 external network access attempted; 20 ingestion; 21 no
documents; 30 index locked; 31 build failed; 32 vector store unusable; 33 no index.
Step 5 (loading and search CLIs) adds 34 incompatible index and 35 stale index.

``load_index`` and ``load_index_with_report`` (Step 5) live in :mod:`src.indexing.loading`
and are re-exported here, where Phase 6 imports them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, TextIO

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, MetadataMode

from src.errors import (
    ConfigurationError,
    EmbeddingInputTooLongError,
    ExternalNetworkAccessError,
    IncompatibleIndexError,
    IndexBuildError,
    IndexingError,
    IndexLockedError,
    IndexNotFoundError,
    IngestionError,
    LocalStackError,
    ModelNotFoundError,
    NoDocumentsError,
    OllamaUnavailableError,
    StaleIndexError,
    VectorStoreError,
    VerificationError,
)
from src.indexing.embeddings import (
    EmbedderIdentity,
    IdentifiedEmbedding,
    ServedEmbedding,
    get_embedding_model,
    validate_vectors,
)
from src.indexing.manifest import IndexManifest, RepoIdentity
from src.indexing.vector_store import METRIC, IndexStore, StoredGeneration, get_vector_store
from src.ingestion.chunker import ChunkingRules, chunk_documents_with_report
from src.ingestion.digest import nodes_digest
from src.ingestion.loader import load_repository_with_report
from src.ingestion.repo_head import RepositoryHeadError, read_head
from src.ingestion.rules import IngestionRules
from src.logging_config import configure_logging
from src.network_guard import NetworkGuard

# Fixed name: under `python -m src.indexing.build_index`, __name__ would be "__main__".
logger = logging.getLogger("src.indexing.build_index")

_COMPONENT: Final = "indexing"
LIBRARIES: Final = (
    "chromadb",
    "llama-index-core",
    "llama-index-vector-stores-chroma",
    "ollama",
    "tree-sitter",
    "tree-sitter-python",
)


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one indexing run did: counts, digests, identity, timing. No content."""

    documents: int
    nodes: int
    reused: int
    embedded: int
    removed: int
    node_digest: str
    embed_digest: str
    repo: RepoIdentity
    generation: str
    previous_generation: str | None
    embedder: EmbedderIdentity
    leftovers_removed: int
    previous_discarded: bool
    duration_ms: float
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "nodes": self.nodes,
            "reused": self.reused,
            "embedded": self.embedded,
            "removed": self.removed,
            "node_digest": self.node_digest,
            "embed_digest": self.embed_digest,
            "repo": self.repo.to_dict(),
            "generation": self.generation,
            "previous_generation": self.previous_generation,
            "embedder": self.embedder.to_dict(),
            "leftovers_removed": self.leftovers_removed,
            "previous_discarded": self.previous_discarded,
            "duration_ms": round(self.duration_ms, 1),
            "size_bytes": self.size_bytes,
        }

    def render_text(self) -> str:
        head = self.repo.head or self.repo.state
        return (
            f"Indexed: {self.documents} documents -> {self.nodes} nodes "
            f"({self.reused} reused, {self.embedded} embedded, {self.removed} removed)\n"
            f"Repository: {self.repo.path} @ {head}\n"
            f"Node digest: {self.node_digest}\n"
            f"Embedder: {self.embedder.model} ({(self.embedder.digest or 'unverified')[:12]}), "
            f"dimension {self.embedder.dimension}\n"
            f"Generation: {self.generation} (previous: {self.previous_generation or '-'}, "
            f"{'discarded' if self.previous_discarded else 'kept for the next run'})\n"
            f"Size on disk: {self.size_bytes:,} B; {self.duration_ms / 1000:.1f} s\n"
        )


@dataclass(frozen=True, slots=True)
class BuildResult:
    index: VectorStoreIndex
    report: IndexReport
    manifest: IndexManifest


# --- pure helpers ------------------------------------------------------------------------------


def repo_identity(root: Path) -> RepoIdentity:
    """Resolved path plus HEAD, "not a git repo", or "HEAD unresolvable" with its reason."""
    resolved = root.resolve()
    try:
        (resolved / ".git").lstat()
    except FileNotFoundError:
        return RepoIdentity(str(resolved), "not_a_git_repo", None, None)
    try:
        return RepoIdentity(str(resolved), "git", read_head(resolved), None)
    except RepositoryHeadError as error:
        return RepoIdentity(str(resolved), "head_unresolvable", None, str(error))


def embed_string_sha256(node: BaseNode) -> str:
    """sha256 of what the embedder receives for ``node`` (before the recipe's prefix)."""
    return hashlib.sha256(
        node.get_content(metadata_mode=MetadataMode.EMBED).encode("utf-8")
    ).hexdigest()


def embed_strings_digest(nodes: Sequence[BaseNode], shas: Sequence[str]) -> str:
    lines = (f"{node.node_id}:{sha}" for node, sha in zip(nodes, shas, strict=True))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _libraries() -> dict[str, str]:
    found: dict[str, str] = {}
    for package in LIBRARIES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not installed"
    return found


def _identity(embed_model: BaseEmbedding) -> EmbedderIdentity:
    if isinstance(embed_model, IdentifiedEmbedding):
        return embed_model.identity()
    logger.warning(
        "index.embedder_unverified",
        extra={"component": _COMPONENT, "embedder": type(embed_model).__name__},
    )
    return EmbedderIdentity(str(embed_model.model_name), None, None, None)


def _server_version(embed_model: BaseEmbedding) -> str | None:
    return embed_model.server_version() if isinstance(embed_model, ServedEmbedding) else None


def _reusable(previous: IndexManifest, identity: EmbedderIdentity, collection: str) -> bool:
    return (
        identity.verified
        and previous.embedder.verified
        and previous.embedder.digest == identity.digest
        and previous.embedder.recipe_fingerprint == identity.recipe_fingerprint
        and previous.embedder.dimension == identity.dimension
        and previous.collection == collection
        and previous.metric == METRIC
    )


# --- the orchestrator ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Previous:
    manifest: IndexManifest | None
    unreadable: bool
    vectors: dict[str, tuple[str, list[float]]]  # id -> (EMBED sha, vector), reusable only
    ids: frozenset[str]


def _read_previous_manifest(store: IndexStore) -> tuple[IndexManifest | None, bool]:
    """The active manifest, and whether it was unreadable (then nothing is deleted)."""
    try:
        return store.read_manifest(), False
    except VectorStoreError:
        logger.warning("index.previous_unreadable", extra={"component": _COMPONENT})
        return None, True


def _read_previous(
    store: IndexStore, manifest: IndexManifest | None, unreadable: bool, identity: EmbedderIdentity
) -> _Previous:
    if manifest is None:
        return _Previous(None, unreadable, {}, frozenset())
    reusable = _reusable(manifest, identity, store.collection_name)
    vectors: dict[str, tuple[str, list[float]]] = {}
    ids: set[str] = set()
    try:
        _, active = store.open_active()
    except IndexingError:
        logger.warning("index.previous_unopenable", extra={"component": _COMPONENT})
        return _Previous(manifest, False, {}, frozenset())
    try:
        for node, vector in active.entries():
            ids.add(node.node_id)
            if reusable:
                vectors[node.node_id] = (embed_string_sha256(node), vector)
    finally:
        active.close()
    return _Previous(manifest, False, vectors, frozenset(ids))


def _check_documents_outside_store(
    repo_root: Path, store: IndexStore, paths: Sequence[str]
) -> None:
    persist = store.persist_dir.resolve()
    try:
        inside = persist.relative_to(repo_root).as_posix()
    except ValueError:
        return  # the store is not inside the repository
    offending = [p for p in paths if p == inside or p.startswith(inside + "/")]
    if offending:
        raise IndexBuildError(
            f"{len(offending)} loaded documents are inside the vector store {persist}; the "
            "index must never index itself. Move VECTOR_STORE_DIR or ignore it."
        )


def _embed_pending(
    embed_model: BaseEmbedding, pending: list[BaseNode], dimension: int | None
) -> int | None:
    """Embed ``pending`` in place (``node.embedding``); returns the dimension seen."""
    batch_size = max(1, int(embed_model.embed_batch_size))
    started = time.perf_counter()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in batch]
        try:
            vectors = embed_model.get_text_embedding_batch(texts)
        except EmbeddingInputTooLongError as error:
            node = batch[error.index or 0]
            raise EmbeddingInputTooLongError(
                f"{node.metadata.get('file_path')}:{node.metadata.get('start_line')}-"
                f"{node.metadata.get('end_line')} ({len(texts[error.index or 0])} characters "
                "with its visible metadata) does not fit the embedding model's context.",
                index=error.index,
                length=error.length,
            ) from error
        expected = dimension if dimension is not None else (len(vectors[0]) if vectors else 0)
        checked = validate_vectors(vectors, len(batch), expected, str(embed_model.model_name))
        dimension = expected
        for node, vector in zip(batch, checked, strict=True):
            node.embedding = vector
        logger.info(
            "index.embedding_progress",
            extra={
                "component": _COMPONENT,
                "done": min(start + batch_size, len(pending)),
                "total": len(pending),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
    return dimension


def _write(
    staging: StoredGeneration, nodes: Sequence[BaseNode], write_batch_size: int | None
) -> None:
    step = min(staging.max_batch_size(), write_batch_size or staging.max_batch_size())
    for start in range(0, len(nodes), step):
        staging.add(nodes[start : start + step])


def _verify(staging: StoredGeneration, nodes: Sequence[BaseNode], dimension: int | None) -> None:
    if staging.metric() != METRIC:
        raise IndexBuildError(f"staging generation uses {staging.metric()!r}, not {METRIC!r}")
    if staging.search_breadth() < staging.count():
        raise IndexBuildError("staging's HNSW search would be approximate (ef_search < vectors)")
    expected = {node.node_id: node.hash for node in nodes}
    seen: dict[str, str] = {}
    for node, vector in staging.entries():
        seen[node.node_id] = node.hash
        if dimension is not None and len(vector) != dimension:
            raise IndexBuildError("staging holds a vector of the wrong dimension")
        validate_vectors([vector], 1, len(vector), "stored vector")
    if staging.count() != len(nodes) or seen.keys() != expected.keys():
        raise IndexBuildError(
            f"staging holds {staging.count()} vectors for {len(nodes)} nodes, or other IDs"
        )
    mismatched = sum(1 for node_id, digest in expected.items() if seen[node_id] != digest)
    if mismatched:
        raise IndexBuildError(f"{mismatched} nodes changed in the vector-store round trip")


def build_index_with_report(
    repo_path: Path,
    vector_store: IndexStore | None = None,
    embed_model: BaseEmbedding | None = None,
    *,
    rules_file: Path | None = None,
    ingestion_rules: IngestionRules | None = None,
    chunking_rules: ChunkingRules | None = None,
    write_batch_size: int | None = None,
) -> BuildResult:
    """Index ``repo_path`` and explain the run; see the module docstring."""
    started = time.perf_counter()
    store = get_vector_store() if vector_store is None else vector_store
    if not isinstance(store, IndexStore):
        raise IndexBuildError(
            "vector_store must be the location handle returned by get_vector_store() "
            f"(an IndexStore), not {type(store).__name__}."
        )
    embedder = get_embedding_model() if embed_model is None else embed_model
    if rules_file is None:
        from src import config

        rules_file = config.INGEST_IGNORE_FILE
    chunking_rules = chunking_rules if chunking_rules is not None else ChunkingRules()

    with store.lock():
        # 1. leftovers of ours, only when the active manifest is readable
        previous_manifest, unreadable = _read_previous_manifest(store)
        if not unreadable:
            keep = frozenset({previous_manifest.generation}) if previous_manifest else frozenset()
            store.discard_leftovers(keep)

        # 2. load and chunk
        loaded = load_repository_with_report(
            repo_path, rules=ingestion_rules, rules_file=rules_file
        )
        if not loaded.documents:
            raise NoDocumentsError(
                f"No documents were loaded from {str(repo_path)!r}; the index was not touched."
            )
        repo_root = Path(repo_path).resolve()
        _check_documents_outside_store(repo_root, store, [d.doc_id for d in loaded.documents])
        chunked = chunk_documents_with_report(loaded.documents, rules=chunking_rules)
        nodes: list[BaseNode] = list(chunked.nodes)
        repeated = sum(1 for count in Counter(n.node_id for n in nodes).values() if count > 1)
        if repeated:
            raise IndexBuildError(f"{repeated} node IDs are repeated; Chroma would drop them.")

        # 3. identity, digests
        repo = repo_identity(repo_root)
        node_digest = nodes_digest(nodes)
        shas = [embed_string_sha256(node) for node in nodes]
        embed_digest = embed_strings_digest(nodes, shas)

        # 4. the embedder, verified before anything is created
        identity = _identity(embedder)
        ollama_version = _server_version(embedder)

        # 5. previous index, reuse
        previous = _read_previous(store, previous_manifest, unreadable, identity)
        copies: list[BaseNode] = []
        pending: list[BaseNode] = []
        for node, sha in zip(nodes, shas, strict=True):
            copy = node.model_copy()
            hit = previous.vectors.get(node.node_id)
            if hit is not None and hit[0] == sha:
                copy.embedding = list(hit[1])
            else:
                copy.embedding = None
                pending.append(copy)
            copies.append(copy)

        # 6-9. staging, embedding, writing, verification, activation
        staging = store.create_staging()
        final_identity = identity
        activated = False
        try:
            dimension = _embed_pending(embedder, pending, identity.dimension)
            if dimension is None and copies and copies[0].embedding is not None:
                dimension = len(copies[0].embedding)
            _write(staging, copies, write_batch_size)
            staging.make_search_exhaustive()
            _verify(staging, nodes, dimension)
            if identity.verified and _identity(embedder).digest != identity.digest:
                raise IndexBuildError(
                    "The embedding model changed during indexing (its digest moved); nothing "
                    "was activated. Run `make index` again."
                )
            final_identity = (
                identity
                if identity.dimension is not None
                else EmbedderIdentity(identity.model, None, None, dimension)
            )
            manifest = IndexManifest(
                generation=staging.name,
                collection=store.collection_name,
                created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                repo=repo,
                documents=len(loaded.documents),
                nodes=len(nodes),
                node_digest=node_digest,
                embed_digest=embed_digest,
                embedder=final_identity,
                recipe=_recipe_dict(embedder) if identity.verified else None,
                ollama_version=ollama_version,
                metric=METRIC,
                max_chunk_chars=chunking_rules.max_chunk_chars,
                rules_file_sha256=_file_sha256(rules_file),
                libraries=_libraries(),
            )
            store.activate(staging, manifest)
            activated = True
        finally:
            if not activated:
                _discard_quietly(store, staging)

        # 10. the previous generation (and any other leftover of ours)
        previous_discarded = True
        leftovers = 0
        try:
            leftovers = len(store.discard_leftovers(frozenset({staging.name})))
        except (OSError, VectorStoreError):
            previous_discarded = False
            logger.warning("index.discard_failed", extra={"component": _COMPONENT})

    reused = len(copies) - len(pending)
    report = IndexReport(
        documents=len(loaded.documents),
        nodes=len(nodes),
        reused=reused,
        embedded=len(pending),
        removed=len(previous.ids - {n.node_id for n in nodes}),
        node_digest=node_digest,
        embed_digest=embed_digest,
        repo=repo,
        generation=staging.name,
        previous_generation=previous.manifest.generation if previous.manifest else None,
        embedder=final_identity,
        leftovers_removed=leftovers,
        previous_discarded=previous_discarded,
        duration_ms=(time.perf_counter() - started) * 1000,
        size_bytes=staging.size_bytes(),
    )
    logger.info(
        "index.done",
        extra={
            "component": _COMPONENT,
            "documents": report.documents,
            "nodes": report.nodes,
            "reused": report.reused,
            "embedded": report.embedded,
            "removed": report.removed,
            "duration_ms": round(report.duration_ms, 1),
        },
    )
    index = VectorStoreIndex.from_vector_store(staging.vector_store(), embed_model=embedder)
    return BuildResult(index, report, manifest)


def _recipe_dict(embed_model: BaseEmbedding) -> dict[str, object] | None:
    recipe = getattr(embed_model, "recipe", None)
    to_dict = getattr(recipe, "to_dict", None)
    if not callable(to_dict):
        return None
    result = to_dict()
    return dict(result) if isinstance(result, dict) else None


def _discard_quietly(store: IndexStore, staging: StoredGeneration) -> None:
    try:
        store.discard(staging)
    except (OSError, VectorStoreError):
        logger.warning("index.staging_discard_failed", extra={"component": _COMPONENT})


def build_index(
    repo_path: Path,
    vector_store: IndexStore | None = None,
    embed_model: BaseEmbedding | None = None,
    *,
    rules_file: Path | None = None,
    ingestion_rules: IngestionRules | None = None,
    chunking_rules: ChunkingRules | None = None,
    write_batch_size: int | None = None,
) -> VectorStoreIndex:
    """Index ``repo_path`` (Phase 3 contract); see the module docstring."""
    return build_index_with_report(
        repo_path,
        vector_store,
        embed_model,
        rules_file=rules_file,
        ingestion_rules=ingestion_rules,
        chunking_rules=chunking_rules,
        write_batch_size=write_batch_size,
    ).index


# --- CLI -------------------------------------------------------------------------------------


class ExitCode(IntEnum):
    OK = 0
    UNEXPECTED = 1
    CONFIGURATION = 10
    OLLAMA_UNAVAILABLE = 11
    MODEL_NOT_FOUND = 12
    VERIFICATION_FAILED = 13
    EXTERNAL_NETWORK = 14
    INGESTION = 20
    NO_DOCUMENTS = 21
    INDEX_LOCKED = 30
    INDEX_BUILD = 31
    VECTOR_STORE = 32
    INDEX_NOT_FOUND = 33
    INDEX_INCOMPATIBLE = 34  # Step 5
    INDEX_STALE = 35  # Step 5


_EXIT_CODE_BY_ERROR: Final[tuple[tuple[type[LocalStackError], ExitCode], ...]] = (
    (ExternalNetworkAccessError, ExitCode.EXTERNAL_NETWORK),
    (ConfigurationError, ExitCode.CONFIGURATION),
    (OllamaUnavailableError, ExitCode.OLLAMA_UNAVAILABLE),
    (ModelNotFoundError, ExitCode.MODEL_NOT_FOUND),
    (VerificationError, ExitCode.VERIFICATION_FAILED),
    (NoDocumentsError, ExitCode.NO_DOCUMENTS),
    (IngestionError, ExitCode.INGESTION),
    (IndexLockedError, ExitCode.INDEX_LOCKED),
    (IndexBuildError, ExitCode.INDEX_BUILD),
    (IndexNotFoundError, ExitCode.INDEX_NOT_FOUND),
    (IncompatibleIndexError, ExitCode.INDEX_INCOMPATIBLE),
    (StaleIndexError, ExitCode.INDEX_STALE),
    (VectorStoreError, ExitCode.VECTOR_STORE),
)


def exit_code_for(error: LocalStackError) -> ExitCode:
    for error_type, code in _EXIT_CODE_BY_ERROR:
        if isinstance(error, error_type):
            return code
    return ExitCode.UNEXPECTED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.indexing.build_index",
        description="Index a repository into the local vector store (no search; Step 6).",
    )
    parser.add_argument("--repo", type=Path, help="repository root (default: REPO_PATH)")
    parser.add_argument(
        "--rules-file", type=Path, help="ingestion rules (default: INGEST_IGNORE_FILE)"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _fail(out: TextIO, code: ExitCode, error: Exception) -> int:
    logger.error(
        "index.failed",
        extra={"component": _COMPONENT, "exit_code": int(code), "error_type": type(error).__name__},
    )
    out.write(f"[FAIL] indexing: {error}\n")
    out.write(f"RESULT: FAIL (exit {code.value} {code.name})\n")
    return int(code)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging()
    out = stdout if stdout is not None else sys.stdout
    try:
        config = importlib.import_module("src.config")
        repo: Path = args.repo if args.repo is not None else config.REPO_PATH
        rules_file: Path = (
            args.rules_file if args.rules_file is not None else config.INGEST_IGNORE_FILE
        )
        allowed: frozenset[str] = config.LOCAL_ALLOWED_HOSTS
    except ConfigurationError as error:
        return _fail(out, ExitCode.CONFIGURATION, error)
    logger.debug("index.started", extra={"component": _COMPONENT, "repo": str(repo)})
    with NetworkGuard(allowed_hosts=allowed) as guard:
        try:
            result = build_index_with_report(repo, rules_file=rules_file)
        except LocalStackError as error:
            return _fail(out, exit_code_for(error), error)
    if guard.violations:  # something swallowed the guard's exception
        first = guard.violations[0]
        blocked = ExternalNetworkAccessError(first.operation, first.destination)
        return _fail(out, ExitCode.EXTERNAL_NETWORK, blocked)
    if args.json:
        out.write(json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    else:
        out.write(result.report.render_text())
        out.write(f"RESULT: OK (exit {ExitCode.OK.value} OK)\n")
    return int(ExitCode.OK)


if __name__ == "__main__":
    raise SystemExit(main())


# Step 5: the loading API, re-exported where Phase 6 imports it (defined in src.indexing.loading,
# which imports this module's repo_identity at call time, so there is no import cycle).
from src.indexing.loading import load_index as load_index  # noqa: E402
from src.indexing.loading import load_index_with_report as load_index_with_report  # noqa: E402
