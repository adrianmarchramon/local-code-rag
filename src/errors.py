"""Typed error hierarchy for the local stack: configuration, Ollama access, network policy.

Messages are meant to be actionable: what failed, the likely cause, and the command that
fixes it. Code that translates a lower-level failure chains it with ``raise ... from``.
"""

from __future__ import annotations


class LocalStackError(Exception):
    """Base class for every error raised by this project's local-stack code."""


class ConfigurationError(LocalStackError):
    """A setting is malformed or violates the local-only policy (raised on import)."""


class OllamaUnavailableError(LocalStackError):
    """The Ollama server cannot be reached at the configured address."""


class ModelNotFoundError(LocalStackError):
    """A configured model is not present in the local Ollama model store."""

    def __init__(self, model: str, fix_command: str | None = None) -> None:
        self.model = model
        self.fix_command = fix_command or f"ollama pull {model}"
        super().__init__(
            f"Model {model!r} is not in the local Ollama store. Fix: run `{self.fix_command}`."
        )


class ExternalNetworkAccessError(LocalStackError):
    """A connection or name lookup to a non-loopback destination was blocked."""

    def __init__(self, operation: str, destination: str) -> None:
        self.operation = operation
        self.destination = destination
        super().__init__(
            f"Blocked {operation} to non-loopback destination {destination!r}: this project "
            "must not reach external networks at runtime. If it is a trusted local service "
            "(e.g. the `ollama` container in Docker Compose), add it to LOCAL_ALLOWED_HOSTS."
        )


class VerificationError(LocalStackError):
    """A local inference check ran but produced an invalid result."""


class IngestionError(LocalStackError):
    """Ingestion cannot proceed: invalid filtering rules or an unusable repository."""


class IgnoreRulesError(IngestionError):
    """An ignore source (built-in patterns, corpus rules file, or a corpus ``.gitignore``)
    cannot be read or compiled.

    Fatal on purpose (fail-closed): indexing without the project's ignore rules could
    index exactly what the project meant to hide.
    """


class RepositoryRootError(IngestionError):
    """The repository to ingest does not exist or is not a directory (fatal)."""


class NoDocumentsError(IngestionError):
    """The repository root is valid but nothing in it passed the filters (decision D8).

    Raised by ``load_repository`` so that an empty index is never built silently; the
    function that also returns the load report returns the empty result instead.
    """


class ChunkingError(IngestionError):
    """Chunking cannot proceed, or produced a span that breaks its own contract (a bug)."""


class InvalidChunkingRulesError(ChunkingError):
    """A chunking parameter is invalid, such as a non-positive size cap."""


class ParserUnavailableError(ChunkingError):
    """A grammar registered for a language could not be loaded.

    Grammars ship compiled inside their wheels (decision P1) and are never downloaded, so
    this means the virtual environment does not match ``uv.lock``: run ``make install``.
    """


class InvalidDocumentError(ChunkingError):
    """A document handed to the chunker breaks the loader's contract.

    Node IDs are derived from ``doc_id``, so a document without the loader's six metadata
    keys, with ``doc_id != file_path`` (for example a ``Document`` created without ``id_``,
    which gets a random UUID) or a repeated ``doc_id`` would produce nodes that change from
    run to run. Build documents with ``src.ingestion.loader.load_repository``.
    """


# --- Phase 3: embeddings and the vector index ------------------------------------------------


class EmbeddingRecipeError(ConfigurationError):
    """``EMBEDDING_MODEL`` has no embedding recipe (prefixes, dimension, context).

    Embedding without the model's task prefixes would silently put documents and queries
    in different parts of the vector space, so an unknown model is a configuration error.
    """


class UnverifiedModelError(VerificationError):
    """The local embedding model is not the one its recipe was verified for.

    Raised when the model has no local weights, its digest is not in the recipe's list of
    verified digests, or ``ollama show`` reports another context length or dimension.
    """


class InvalidEmbeddingError(VerificationError):
    """The embedder returned the wrong number of vectors, or an invalid vector."""


class EmbeddingInputTooLongError(VerificationError):
    """An input exceeds the embedding model's context and ``truncate`` is off.

    ``index`` is the input's position in its batch and ``length`` its length in
    characters, when known; the text itself is never part of the error.
    """

    def __init__(self, message: str, *, index: int | None = None, length: int | None = None):
        self.index = index
        self.length = length
        super().__init__(message)


class IndexingError(LocalStackError):
    """Base class for vector-index errors (Phase 3)."""


class VectorStoreError(IndexingError):
    """The vector store cannot be opened safely, or holds something unexpected."""


class IndexNotFoundError(IndexingError):
    """There is no active index at the configured location (run ``make index``)."""


class IndexLockedError(IndexingError):
    """Another indexing run holds the index lock."""


class IndexBuildError(IndexingError):
    """Building the index failed; the previous index, if any, is still the active one."""
