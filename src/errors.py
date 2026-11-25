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
