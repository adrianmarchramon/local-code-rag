"""Phase 3, Step 3: the persistent vector store, hardened, behind a location handle.

**One client factory** (:func:`open_chroma_client`) is the only place in ``src/`` that
creates a Chroma client. It pins every setting that decides where Chroma connects or which
module it imports, because ``chromadb.config.Settings`` reads a ``.env`` file from the
working directory and ``CHROMA_*`` environment variables: with ``CHROMA_API_IMPL`` and a
server host there, a "persistent" client silently becomes an HTTP client (measured, V1).
It then checks the *effective* settings and refuses anything else (fail-closed), and it
keeps the libraries that log node text at WARNING (V6).

**Collections**: writes create (``create_collection``, cosine space, no embedding
function); reads only open (``get_collection``, no embedding function). Never
``get_or_create_collection`` (a missing index would become an empty one, and an existing
one keeps its old metric), never ``ChromaVectorStore`` without a collection (it builds an
``HttpClient``), never ``from_params``, never ``query_texts`` (they would run Chroma's
default embedding function, which downloads a model).

**Exhaustive search** (a Step 2 defect fixed on 2026-11-27 with the owner's OK): a build
raises each generation's HNSW ``ef_search`` to at least its vector count
(:meth:`ChromaGeneration.make_search_exhaustive`, kept by Chroma), and loading refuses a
generation below that. With Chroma's default (100) the search is approximate. Below
``sync_threshold`` (1000) the graph is never written to disk and every process rebuilds
it, so the misses differed from run to run: measured on the corpus, and on 658 synthetic
vectors (1-5 of 30 queries wrong, a different result in every process).

**Layout** (Gate 2 Q-B: one Chroma directory per generation, so a discarded index leaves
no residue; ``delete_collection`` leaves deleted text in the sqlite file, V2)::

    <persist_dir>/                 0700
        manifest.json              active-generation pointer + manifest (src.indexing.manifest)
        .lock                      fcntl lock held while building
        gen-<utc>-<hex>/           one complete Chroma store per generation (0700)
            .local-code-rag-generation   marker: only directories with it are ever deleted
            chroma.sqlite3, <segment dirs>

:class:`IndexStore` is the Chroma-independent protocol the orchestrator drives (Gate 1
Q-A); :func:`get_vector_store` returns the Chroma implementation. Constructing a store and
reading its manifest never write to disk; ``PersistentClient`` would create missing paths
(V5), so nothing is opened for reading before checking that it exists.
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import logging
import os
import re
import secrets
import shutil
import stat
import warnings
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from src.errors import IndexLockedError, IndexNotFoundError, VectorStoreError
from src.indexing.manifest import MANIFEST_NAME, IndexManifest, read_manifest, write_manifest
from src.logging_config import contain_library_loggers

if TYPE_CHECKING:
    from chromadb.api.client import Client
    from chromadb.api.models.Collection import Collection
    from chromadb.config import Settings

logger = logging.getLogger(__name__)

_COMPONENT: Final = "vector_store"
METRIC: Final = "cosine"
SEARCH_EF_FLOOR: Final = 1024  # ef_search = max(this, vectors): an exhaustive HNSW search
LOCK_NAME: Final = ".lock"
MARKER_NAME: Final = ".local-code-rag-generation"
_MARKER_TEXT: Final = "local-code-rag index generation\n"
GENERATION_RE: Final = re.compile(r"gen-\d{8}T\d{12}Z-[0-9a-f]{8}")
# Chroma's collection-name rule (also enforced by src.config for COLLECTION_NAME).
COLLECTION_NAME_RE: Final = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]")
_RUST_API: Final = "chromadb.api.rust.RustBindingsAPI"
_READ_PAGE: Final = 1000
_PROJECT_ROOT: Final = Path(__file__).absolute().parents[2]


# --- the single client factory ---------------------------------------------------------------


def chroma_settings(path: Path) -> Settings:
    """Settings for an embedded, local, telemetry-free store at ``path`` (nothing read
    from ``.env`` or the environment for the fields that decide where it connects)."""
    from chromadb.config import Settings

    return Settings(
        _env_file=None,  # type: ignore[call-arg]  # pydantic-settings init option (V1)
        chroma_api_impl=_RUST_API,
        is_persistent=True,
        persist_directory=str(path),
        anonymized_telemetry=False,
        allow_reset=False,
        chroma_server_host=None,
        chroma_server_http_port=None,
        chroma_server_headers=None,
        chroma_server_ssl_enabled=False,
        chroma_otel_collection_endpoint="",
        chroma_otel_collection_headers={},
        chroma_otel_granularity=None,
        chroma_client_auth_provider=None,
        chroma_client_auth_credentials=None,
    )


def _effective_problems(settings: Settings, path: Path) -> list[str]:
    expected: dict[str, object] = {
        "chroma_api_impl": _RUST_API,
        "is_persistent": True,
        "persist_directory": str(path),
        "anonymized_telemetry": False,
        "allow_reset": False,
        "chroma_server_host": None,
        "chroma_otel_collection_endpoint": "",
        "chroma_otel_granularity": None,
    }
    return [key for key, value in expected.items() if getattr(settings, key) != value]


def open_chroma_client(path: Path) -> Client:
    """The only way this project opens Chroma: an existing directory, local, verified."""
    info = _lstat(path)
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise VectorStoreError(
            f"{path} is not an existing directory; Chroma would create it, so it is never "
            "opened for reading."
        )
    import chromadb

    contain_library_loggers()
    from chromadb.api.client import Client

    created = chromadb.PersistentClient(path=str(path), settings=chroma_settings(path))
    if not isinstance(created, Client):  # the embedded client class, which has close()
        raise VectorStoreError(f"Chroma returned an unexpected client type for {path}.")
    client = created
    problems = _effective_problems(client.get_settings(), path)
    if problems:
        client.close()
        raise VectorStoreError(
            f"The Chroma client for {path} is not the expected local one (effective settings "
            f"differ: {sorted(problems)}); refusing to use it."
        )
    return client


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


# --- protocols (Chroma-independent) ----------------------------------------------------------


class StoredGeneration(Protocol):
    """One complete, self-contained set of vectors."""

    @property
    def name(self) -> str: ...

    def vector_store(self) -> BasePydanticVectorStore: ...

    def count(self) -> int: ...

    def metric(self) -> str: ...

    def search_breadth(self) -> int: ...

    def make_search_exhaustive(self) -> int: ...

    def max_batch_size(self) -> int: ...

    def add(self, nodes: Sequence[BaseNode]) -> None: ...

    def entries(self) -> Iterator[tuple[BaseNode, list[float]]]: ...

    def size_bytes(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class IndexStore(Protocol):
    """Where an index lives, and its lifecycle: the orchestrator's only view of storage."""

    @property
    def persist_dir(self) -> Path: ...

    @property
    def collection_name(self) -> str: ...

    def read_manifest(self) -> IndexManifest | None: ...

    def open_active(self) -> tuple[IndexManifest, StoredGeneration]: ...

    def create_staging(self) -> StoredGeneration: ...

    def activate(self, generation: StoredGeneration, manifest: IndexManifest) -> None: ...

    def discard(self, generation: StoredGeneration) -> None: ...

    def discard_leftovers(self, keep: frozenset[str]) -> list[str]: ...

    def lock(self) -> contextlib.AbstractContextManager[None]: ...


# --- Chroma implementation -------------------------------------------------------------------


class ChromaGeneration:
    """A generation directory with its own embedded Chroma client."""

    def __init__(self, name: str, path: Path, client: Client, collection: Collection):
        self._name = name
        self._path = path
        self._client: Client | None = client
        self._collection: Collection | None = collection

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    def _live(self) -> Collection:
        if self._collection is None:
            raise VectorStoreError(f"generation {self._name} is closed")
        return self._collection

    def vector_store(self) -> BasePydanticVectorStore:
        from llama_index.vector_stores.chroma import ChromaVectorStore

        return ChromaVectorStore(chroma_collection=self._live())

    def count(self) -> int:
        return self._live().count()

    def metric(self) -> str:
        hnsw = self._live().configuration_json.get("hnsw") or {}
        space = hnsw.get("space") if isinstance(hnsw, dict) else None
        return str(space)

    def search_breadth(self) -> int:
        """HNSW ``ef_search``; the search is exhaustive (exact) only when it is >= count()."""
        hnsw = self._live().configuration_json.get("hnsw") or {}
        value = hnsw.get("ef_search") if isinstance(hnsw, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def make_search_exhaustive(self) -> int:
        """Raise ``ef_search`` to max(SEARCH_EF_FLOOR, count()), which Chroma persists."""
        from chromadb.errors import ChromaError

        breadth = max(SEARCH_EF_FLOOR, self.count())
        try:
            with warnings.catch_warnings():
                # Chroma re-reads embedding_function=None as a "legacy" config here and warns;
                # it keeps no embedding function (read in 1.5.9, measured: none built).
                warnings.filterwarnings(
                    "ignore", "legacy embedding function config", DeprecationWarning
                )
                self._live().modify(configuration={"hnsw": {"ef_search": breadth}})
        except (ChromaError, ValueError) as exc:
            raise VectorStoreError(
                f"generation {self._name} refused ef_search {breadth} ({type(exc).__name__})"
            ) from exc
        if self.search_breadth() != breadth:
            raise VectorStoreError(f"generation {self._name} did not keep ef_search {breadth}")
        return breadth

    def max_batch_size(self) -> int:
        if self._client is None:
            raise VectorStoreError(f"generation {self._name} is closed")
        return self._client.get_max_batch_size()

    def add(self, nodes: Sequence[BaseNode]) -> None:
        """Write nodes that carry their embedding, in batches Chroma accepts (V4: 5461)."""
        store = self.vector_store()
        step = self.max_batch_size()
        for start in range(0, len(nodes), step):
            store.add(list(nodes[start : start + step]))

    def entries(self) -> Iterator[tuple[BaseNode, list[float]]]:
        """Every stored node, reconstructed exactly, with its vector."""
        collection = self._live()
        offset = 0
        while True:
            page = collection.get(
                include=["embeddings", "metadatas", "documents"], limit=_READ_PAGE, offset=offset
            )
            ids = page["ids"]
            if not ids:
                return
            metadatas = page["metadatas"] or []
            documents = page["documents"] or []
            embeddings = page["embeddings"]
            if embeddings is None or len(metadatas) != len(ids) or len(documents) != len(ids):
                raise VectorStoreError(f"generation {self._name} returned an incomplete page")
            for index, node_id in enumerate(ids):
                try:
                    node = metadata_dict_to_node(dict(metadatas[index]), text=documents[index])
                except (ValueError, KeyError, TypeError) as exc:
                    raise VectorStoreError(
                        f"generation {self._name} holds a node that cannot be reconstructed"
                    ) from exc
                if node.node_id != node_id:
                    raise VectorStoreError(f"generation {self._name} holds inconsistent IDs")
                yield node, [float(value) for value in embeddings[index]]
            offset += len(ids)

    def size_bytes(self) -> int:
        return sum(p.lstat().st_size for p in self._path.rglob("*") if p.is_file())

    def close(self) -> None:
        """Release the client and every reference to it (open file handles included)."""
        client = self._client
        self._collection = None
        self._client = None
        if client is not None:
            client.close()
        gc.collect()


class ChromaIndexStore:
    """:class:`IndexStore` over one directory of Chroma generations."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        if not COLLECTION_NAME_RE.fullmatch(collection_name):
            raise VectorStoreError(
                f"collection name {collection_name!r} is invalid: use 3-512 characters from "
                "[a-zA-Z0-9._-], starting and ending with a letter or digit."
            )
        root = Path(persist_dir)
        self._root = root if root.is_absolute() else _PROJECT_ROOT / root
        self._collection = collection_name

    @property
    def persist_dir(self) -> Path:
        return self._root

    @property
    def collection_name(self) -> str:
        return self._collection

    # --- reading: never writes -----------------------------------------------------------

    def read_manifest(self) -> IndexManifest | None:
        info = _lstat(self._root)
        if info is None:
            return None
        if not stat.S_ISDIR(info.st_mode):
            raise VectorStoreError(f"{self._root} is not a directory (symlinks are not followed)")
        return read_manifest(self._root)

    def open_active(self) -> tuple[IndexManifest, ChromaGeneration]:
        manifest = self.read_manifest()
        if manifest is None:
            raise IndexNotFoundError(
                f"No index at {self._root}. Build it with `make index` "
                "(python -m src.indexing.build_index)."
            )
        if manifest.collection != self._collection:
            raise VectorStoreError(
                f"The index at {self._root} holds collection {manifest.collection!r}, not "
                f"{self._collection!r}."
            )
        path = self._generation_path(manifest.generation)
        if not self._is_ours(path):
            raise VectorStoreError(
                f"The index manifest at {self._root} points to generation "
                f"{manifest.generation!r}, which is missing or not a generation of this "
                "project. It is never recreated silently: run `make index`."
            )
        return manifest, self._open(manifest.generation, path, create=False)

    # --- writing: only while building, under the lock ------------------------------------

    def create_staging(self) -> ChromaGeneration:
        self._ensure_root()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"gen-{stamp}-{secrets.token_hex(4)}"
        path = self._generation_path(name)
        path.mkdir(mode=0o700)
        marker = path / MARKER_NAME
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, _MARKER_TEXT.encode("ascii"))
        finally:
            os.close(fd)
        return self._open(name, path, create=True)

    def activate(self, generation: StoredGeneration, manifest: IndexManifest) -> None:
        if manifest.generation != generation.name or manifest.collection != self._collection:
            raise VectorStoreError("the manifest does not describe the generation to activate")
        if not self._is_ours(self._generation_path(generation.name)):
            raise VectorStoreError(f"generation {generation.name!r} is not a staging directory")
        write_manifest(self._root, manifest)

    def discard(self, generation: StoredGeneration) -> None:
        generation.close()
        self._delete(generation.name)

    def discard_leftovers(self, keep: frozenset[str]) -> list[str]:
        """Delete this project's generations not in ``keep``; anything unknown stays."""
        if _lstat(self._root) is None:
            return []
        removed = []
        for entry in sorted(self._root.iterdir()):
            if entry.name not in keep and self._is_ours(entry):
                self._delete(entry.name)
                removed.append(entry.name)
        return removed

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self._ensure_root()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(self._root / LOCK_NAME, flags, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise IndexLockedError(
                    f"Another indexing run holds the lock on {self._root}; wait for it to finish."
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # --- internals -----------------------------------------------------------------------

    def _generation_path(self, name: str) -> Path:
        if not GENERATION_RE.fullmatch(name):
            raise VectorStoreError(f"{name!r} is not a generation name")
        return self._root / name

    def _is_ours(self, path: Path) -> bool:
        """A real directory (not a symlink) named like a generation, with our marker."""
        if not GENERATION_RE.fullmatch(path.name):
            return False
        info = _lstat(path)
        if info is None or not stat.S_ISDIR(info.st_mode):
            return False
        marker = _lstat(path / MARKER_NAME)
        if marker is None or not stat.S_ISREG(marker.st_mode):
            return False
        try:
            return (path / MARKER_NAME).read_text(encoding="ascii") == _MARKER_TEXT
        except (OSError, UnicodeDecodeError):
            return False

    def _delete(self, name: str) -> None:
        path = self._generation_path(name)
        if not self._is_ours(path):
            raise VectorStoreError(f"refusing to delete {name!r}: not a generation of this project")
        shutil.rmtree(path)
        logger.debug(
            "vector_store.generation_deleted", extra={"component": _COMPONENT, "generation": name}
        )

    def _ensure_root(self) -> None:
        info = _lstat(self._root)
        if info is None:
            self._root.mkdir(mode=0o700, parents=True)
            os.chmod(self._root, 0o700)  # mkdir's mode is filtered by the umask
            return
        if not stat.S_ISDIR(info.st_mode):
            raise VectorStoreError(f"{self._root} is not a directory (symlinks are not followed)")
        if stat.S_IMODE(info.st_mode) & 0o077:
            os.chmod(self._root, 0o700)
            logger.warning(
                "vector_store.permissions_fixed",
                extra={"component": _COMPONENT, "old_mode": oct(stat.S_IMODE(info.st_mode))},
            )

    def _open(self, name: str, path: Path, *, create: bool) -> ChromaGeneration:
        from chromadb.errors import ChromaError

        client = open_chroma_client(path)
        opened = False
        try:
            if create:
                collection = client.create_collection(
                    self._collection,
                    embedding_function=None,
                    configuration={"hnsw": {"space": "cosine"}},
                )
            else:
                collection = client.get_collection(self._collection, embedding_function=None)
            opened = True
        except (ChromaError, ValueError) as exc:
            raise VectorStoreError(
                f"generation {name!r} has no usable collection {self._collection!r} "
                f"({type(exc).__name__}); run `make index`."
            ) from exc
        finally:
            if not opened:
                client.close()
        generation = ChromaGeneration(name, path, client, collection)
        if generation.metric() != METRIC:
            generation.close()
            raise VectorStoreError(f"generation {name!r} does not use the {METRIC} metric")
        return generation


def get_vector_store(
    persist_dir: Path | None = None, collection: str | None = None
) -> ChromaIndexStore:
    """The index location (Phase 3 contract): ``VECTOR_STORE_DIR`` and ``COLLECTION_NAME``
    by default. Builds a handle only; nothing is read, created or opened here."""
    if persist_dir is None or collection is None:
        from src import config

        persist_dir = config.VECTOR_STORE_DIR if persist_dir is None else persist_dir
        collection = config.COLLECTION_NAME if collection is None else collection
    return ChromaIndexStore(persist_dir, collection)


__all__ = [
    "COLLECTION_NAME_RE",
    "GENERATION_RE",
    "LOCK_NAME",
    "MANIFEST_NAME",
    "MARKER_NAME",
    "METRIC",
    "SEARCH_EF_FLOOR",
    "ChromaGeneration",
    "ChromaIndexStore",
    "IndexStore",
    "StoredGeneration",
    "chroma_settings",
    "get_vector_store",
    "open_chroma_client",
]
