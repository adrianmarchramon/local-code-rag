"""The index manifest: active-generation pointer and everything a reader must verify.

One versioned JSON file, ``manifest.json`` at the store root, is both the pointer to the
active generation and its manifest (decisions P4, P5, P7). It is written last, atomically
(temporary file in the same directory, ``fsync``, ``os.replace``, ``fsync`` of the
directory), so a reader sees the previous complete index or the new one, never a mix.

It records counts, digests, versions, paths and settings only: never code, symbols or
vectors. Its ``.json`` extension is not one the loader ingests, so an index inside an
indexed repository is never indexed itself.

Step 5 (``load_index``) compares it with the live model and recipe (incompatibility) and,
given the repository's nodes, with their digest and the repository identity (drift).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from src.errors import VectorStoreError
from src.indexing.embeddings import EmbedderIdentity

FORMAT: Final = "local-code-rag/index-manifest"
SCHEMA_VERSION: Final = 1
MANIFEST_NAME: Final = "manifest.json"

RepoState = Literal["git", "not_a_git_repo", "head_unresolvable"]
_REPO_STATES: Final = frozenset({"git", "not_a_git_repo", "head_unresolvable"})
_TOP_KEYS: Final = (
    "format",
    "schema_version",
    "generation",
    "collection",
    "created_utc",
    "repo",
    "documents",
    "nodes",
    "node_digest",
    "embed_digest",
    "embedder",
    "recipe",
    "ollama_version",
    "metric",
    "max_chunk_chars",
    "rules_file_sha256",
    "libraries",
)


@dataclass(frozen=True, slots=True)
class RepoIdentity:
    """Which repository an index was built from (decision P7)."""

    path: str  # resolved, absolute
    state: RepoState
    head: str | None  # the commit SHA when state == "git"
    reason: str | None  # why HEAD could not be resolved, when state == "head_unresolvable"

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "state": self.state, "head": self.head, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """What is in the active generation and how it was produced."""

    generation: str
    collection: str
    created_utc: str
    repo: RepoIdentity
    documents: int
    nodes: int
    node_digest: str  # Phase 2 corpus digest of the nodes (src.ingestion.digest)
    embed_digest: str  # sha256 over "<node id>:<sha256 of its EMBED string>" lines
    embedder: EmbedderIdentity
    recipe: Mapping[str, object] | None  # EmbeddingRecipe.to_dict(), None if unverified
    ollama_version: str | None
    metric: str
    max_chunk_chars: int
    rules_file_sha256: str | None
    libraries: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "generation": self.generation,
            "collection": self.collection,
            "created_utc": self.created_utc,
            "repo": self.repo.to_dict(),
            "documents": self.documents,
            "nodes": self.nodes,
            "node_digest": self.node_digest,
            "embed_digest": self.embed_digest,
            "embedder": self.embedder.to_dict(),
            "recipe": dict(self.recipe) if self.recipe is not None else None,
            "ollama_version": self.ollama_version,
            "metric": self.metric,
            "max_chunk_chars": self.max_chunk_chars,
            "rules_file_sha256": self.rules_file_sha256,
            "libraries": dict(self.libraries),
        }


def format_manifest(manifest: IndexManifest) -> str:
    return json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n"


def _fail(path: Path, problem: str) -> VectorStoreError:
    return VectorStoreError(
        f"The index manifest {path} is unusable ({problem}). It is never repaired in place: "
        "run `make index` to build and activate a new index."
    )


def _str(data: Mapping[str, object], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise _fail(path, f"{key} must be a string")
    return value


def _opt_str(data: Mapping[str, object], key: str, path: Path) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise _fail(path, f"{key} must be a string or null")
    return value


def _int(data: Mapping[str, object], key: str, path: Path) -> int:
    value = data.get(key)
    if type(value) is not int or value < 0:
        raise _fail(path, f"{key} must be a non-negative integer")
    return value


def _dict(data: Mapping[str, object], key: str, path: Path) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise _fail(path, f"{key} must be an object")
    return value


def parse_manifest(text: str, path: Path) -> IndexManifest:
    """Strictly parse a manifest; :class:`VectorStoreError` on any deviation."""
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _fail(path, "not valid JSON") from exc
    if not isinstance(data, dict):
        raise _fail(path, "not a JSON object")
    if data.get("format") != FORMAT:
        raise _fail(path, "unknown format")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise _fail(
            path,
            f"schema version {data.get('schema_version')!r}, this code reads {SCHEMA_VERSION}; "
            "an index written by another version is never overwritten silently",
        )
    if tuple(data) != _TOP_KEYS:
        raise _fail(path, "unexpected keys")
    repo = _dict(data, "repo", path)
    state = repo.get("state")
    if state not in _REPO_STATES or tuple(repo) != ("path", "state", "head", "reason"):
        raise _fail(path, "invalid repo identity")
    embedder = _dict(data, "embedder", path)
    dimension = embedder.get("dimension")
    if dimension is not None and (type(dimension) is not int or dimension <= 0):
        raise _fail(path, "invalid embedder dimension")
    recipe = data.get("recipe")
    if recipe is not None and not isinstance(recipe, dict):
        raise _fail(path, "recipe must be an object or null")
    libraries = _dict(data, "libraries", path)
    if not all(isinstance(v, str) for v in libraries.values()):
        raise _fail(path, "library versions must be strings")
    return IndexManifest(
        generation=_str(data, "generation", path),
        collection=_str(data, "collection", path),
        created_utc=_str(data, "created_utc", path),
        repo=RepoIdentity(
            path=_str(repo, "path", path),
            state=state,  # type: ignore[arg-type]  # checked against _REPO_STATES above
            head=_opt_str(repo, "head", path),
            reason=_opt_str(repo, "reason", path),
        ),
        documents=_int(data, "documents", path),
        nodes=_int(data, "nodes", path),
        node_digest=_str(data, "node_digest", path),
        embed_digest=_str(data, "embed_digest", path),
        embedder=EmbedderIdentity(
            model=_str(embedder, "model", path),
            digest=_opt_str(embedder, "digest", path),
            recipe_fingerprint=_opt_str(embedder, "recipe_fingerprint", path),
            dimension=dimension,  # checked above: None or a positive int
        ),
        recipe=recipe,
        ollama_version=_opt_str(data, "ollama_version", path),
        metric=_str(data, "metric", path),
        max_chunk_chars=_int(data, "max_chunk_chars", path),
        rules_file_sha256=_opt_str(data, "rules_file_sha256", path),
        libraries={str(k): str(v) for k, v in libraries.items()},
    )


def read_manifest(root: Path) -> IndexManifest | None:
    """The manifest at ``root``, or None if there is none. Reads only; never writes."""
    path = root / MANIFEST_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail(path, f"unreadable: {type(exc).__name__}") from exc
    return parse_manifest(text, path)


def write_manifest(root: Path, manifest: IndexManifest) -> None:
    """Atomically replace ``root/manifest.json`` (mode 0600), durable on return."""
    path = root / MANIFEST_NAME
    temporary = root / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    data = format_manifest(manifest).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
