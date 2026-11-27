"""Phase 3, Step 5: reload the persisted index without re-indexing, and refuse a bad one.

"Persisted" is not "compatible" and not "up to date" (decision P5). Loading:

1. opens the active generation through the store handle, which never creates anything
   (``IndexNotFoundError`` when there is no index);
2. re-reads the manifest after opening: a build in another process may have activated a
   new generation and deleted this one in between (an already-open Chroma client keeps
   answering from the deleted files, measured), so a changed pointer means close and
   retry, a bounded number of times;
3. **compatibility, always** (:func:`compatibility_problems`, pure): the index and the live
   embedder must agree on model, model digest, recipe fingerprint and dimension; both must
   be verified identities; the metric must be cosine; the stored count must equal the
   manifest's. Any difference raises :class:`~src.errors.IncompatibleIndexError` listing
   all of them;
4. **drift, only when the repository's current ``nodes`` are given** (:func:`drift_problems`,
   pure): their Phase 2 digest must equal the manifest's, and a repository that had a
   HEAD must still be at that HEAD (Gate 1 Q-E: a new commit is drift even when the nodes
   are identical). Any difference raises :class:`~src.errors.StaleIndexError`. ``nodes=[]``
   is checked (and differs); ``nodes=None`` skips the check and logs it;
5. wraps the generation in a ``VectorStoreIndex`` with the explicit embedder: zero document
   embeddings, no write by this code (Chroma rewrites its own files on open, section 13).

Reading never takes the build lock. :class:`LoadResult` keeps the loaded generation and
offers :meth:`LoadResult.is_current` (Gate 1 Q-F): a cheap manifest read that tells a
long-lived reader (Phase 6) that another build has replaced what it serves; nothing is
reloaded automatically.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode

from src.errors import IncompatibleIndexError, StaleIndexError, VectorStoreError
from src.indexing.embeddings import EmbedderIdentity, IdentifiedEmbedding, get_embedding_model
from src.indexing.manifest import IndexManifest, RepoIdentity
from src.indexing.vector_store import METRIC, IndexStore, StoredGeneration, get_vector_store
from src.ingestion.digest import nodes_digest

logger = logging.getLogger(__name__)

_COMPONENT: Final = "indexing"
_REBUILD: Final = "Rebuild it with `make index` (unchanged nodes reuse their vectors)."
DEFAULT_ATTEMPTS: Final = 3


def _short(value: str | None) -> str:
    return "none" if value is None else value[:12]


# --- pure checks -------------------------------------------------------------------------------


def compatibility_problems(
    manifest: IndexManifest,
    embedder: EmbedderIdentity,
    stored_count: int,
    stored_metric: str,
    *,
    search_breadth: int,
) -> list[str]:
    """Every reason the index cannot be queried with ``embedder`` (empty: compatible)."""
    problems: list[str] = []
    built = manifest.embedder
    if not built.verified:
        problems.append("the index was built with an unverified embedder")
    if not embedder.verified:
        problems.append("the embedder cannot prove its model and recipe")
    if built.model != embedder.model:
        problems.append(f"model {built.model!r} in the index, {embedder.model!r} now")
    if built.digest != embedder.digest:
        problems.append(
            f"model digest {_short(built.digest)} in the index, {_short(embedder.digest)} now"
        )
    if built.recipe_fingerprint != embedder.recipe_fingerprint:
        problems.append(
            f"recipe fingerprint {_short(built.recipe_fingerprint)} in the index, "
            f"{_short(embedder.recipe_fingerprint)} now"
        )
    if built.dimension != embedder.dimension:
        problems.append(f"dimension {built.dimension} in the index, {embedder.dimension} now")
    if manifest.metric != METRIC or stored_metric != METRIC:
        problems.append(
            f"metric {manifest.metric!r} in the manifest and {stored_metric!r} in the store; "
            f"queries need {METRIC!r}"
        )
    if stored_count != manifest.nodes:
        problems.append(f"{stored_count} vectors stored for {manifest.nodes} nodes in the manifest")
    if search_breadth < stored_count:
        problems.append(
            f"the HNSW search is approximate and varies per process (ef_search {search_breadth} "
            f"< {stored_count} vectors; built before the 2026-11-27 fix)"
        )
    return problems


def drift_problems(
    manifest: IndexManifest, nodes: Sequence[BaseNode], current: RepoIdentity
) -> list[str]:
    """Every reason the index no longer matches the repository (empty: up to date)."""
    problems: list[str] = []
    if nodes_digest(nodes) != manifest.node_digest:
        problems.append(
            f"the repository's {len(nodes)} nodes no longer match the index's {manifest.nodes} "
            "(files changed, or the ingestion or chunking rules differ)"
        )
    recorded = manifest.repo
    if recorded.state == "git":
        if current.state != "git":
            problems.append(
                f"the repository had HEAD {_short(recorded.head)} and its HEAD can no longer "
                f"be read ({current.state})"
            )
        elif current.head != recorded.head:
            problems.append(f"HEAD moved from {_short(recorded.head)} to {_short(current.head)}")
    return problems


# --- the loaded index --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadResult:
    """A verified index and what was checked; it reads one generation until closed."""

    index: VectorStoreIndex
    manifest: IndexManifest
    drift_checked: bool
    duration_ms: float
    attempts: int
    _store: IndexStore = field(repr=False)
    _generation: StoredGeneration = field(repr=False)

    @property
    def generation(self) -> str:
        return self.manifest.generation

    def is_current(self) -> bool:
        """Whether the store still points at the loaded generation (one manifest read).

        False means another build activated a new generation: this index may be serving
        results from deleted files. Load again to serve the new one.
        """
        try:
            manifest = self._store.read_manifest()
        except VectorStoreError:
            return False
        return manifest is not None and manifest.generation == self.manifest.generation

    def close(self) -> None:
        """Release the generation's client (the index must not be used afterwards)."""
        self._generation.close()


def _verified_identity(embed_model: BaseEmbedding) -> EmbedderIdentity:
    if not isinstance(embed_model, IdentifiedEmbedding):
        raise IncompatibleIndexError(
            f"The embedder {type(embed_model).__name__} cannot prove its model and recipe, "
            "so an index cannot be loaded for it. Use get_embedding_model()."
        )
    return embed_model.identity()


def load_index_with_report(
    nodes: Sequence[BaseNode] | None = None,
    *,
    vector_store: IndexStore | None = None,
    embed_model: BaseEmbedding | None = None,
    repo_path: Path | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
) -> LoadResult:
    """Open, verify and wrap the active index; see the module docstring."""
    from src.indexing.build_index import repo_identity  # call time: avoids an import cycle

    started = time.perf_counter()
    store = get_vector_store() if vector_store is None else vector_store
    if not isinstance(store, IndexStore):
        raise VectorStoreError(
            "vector_store must be the location handle returned by get_vector_store() "
            f"(an IndexStore), not {type(store).__name__}."
        )
    embedder = get_embedding_model() if embed_model is None else embed_model
    identity = _verified_identity(embedder)

    for attempt in range(1, max(1, attempts) + 1):
        before = _active_name(store)
        try:
            manifest, generation = store.open_active()
        except VectorStoreError:
            # The pointer may have moved between reading it and opening its generation.
            if attempt < attempts and _active_name(store) != before:
                continue
            raise
        loaded = False
        try:
            if _pointer_moved(store, manifest.generation):
                logger.info("index.load_retry", extra={"component": _COMPONENT, "attempt": attempt})
                continue
            problems = compatibility_problems(
                manifest,
                identity,
                generation.count(),
                generation.metric(),
                search_breadth=generation.search_breadth(),
            )
            drift: list[str] = []
            if nodes is not None:
                current = repo_identity(Path(repo_path or manifest.repo.path))
                drift = drift_problems(manifest, nodes, current)
            if problems:
                raise IncompatibleIndexError(
                    "The index cannot be queried with this embedder:\n - "
                    + "\n - ".join(problems + [f"also stale: {d}" for d in drift])
                    + f"\n{_REBUILD}"
                )
            if drift:
                raise StaleIndexError(
                    "The index is out of date:\n - " + "\n - ".join(drift) + f"\n{_REBUILD}"
                )
            if nodes is None:
                logger.warning(
                    "index.drift_unchecked",
                    extra={"component": _COMPONENT, "generation": manifest.generation},
                )
            index = VectorStoreIndex.from_vector_store(
                generation.vector_store(), embed_model=embedder
            )
            loaded = True
        finally:
            if not loaded:
                generation.close()
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "index.loaded",
            extra={
                "component": _COMPONENT,
                "generation": manifest.generation,
                "nodes": manifest.nodes,
                "drift_checked": nodes is not None,
                "attempts": attempt,
                "duration_ms": round(duration_ms, 1),
            },
        )
        return LoadResult(
            index, manifest, nodes is not None, duration_ms, attempt, store, generation
        )
    raise VectorStoreError(
        f"The index at {store.persist_dir} kept changing while it was being loaded "
        f"({attempts} attempts); try again once the running `make index` finishes."
    )


def _active_name(store: IndexStore) -> str | None:
    """The generation the manifest names now (None if there is no index)."""
    manifest = store.read_manifest()
    return None if manifest is None else manifest.generation


def _pointer_moved(store: IndexStore, expected: str) -> bool:
    """Whether the manifest now names a generation other than ``expected``."""
    try:
        return _active_name(store) != expected
    except VectorStoreError:
        return True


def load_index(
    nodes: Sequence[BaseNode] | None = None,
    *,
    vector_store: IndexStore | None = None,
    embed_model: BaseEmbedding | None = None,
    repo_path: Path | None = None,
) -> VectorStoreIndex:
    """The verified active index (Phase 3 contract; Phase 6 calls it at start-up)."""
    return load_index_with_report(
        nodes, vector_store=vector_store, embed_model=embed_model, repo_path=repo_path
    ).index
