"""Phase 3, Step 2: the local embedding model, its recipe and the LlamaIndex adapter.

One immutable :class:`EmbeddingRecipe` per model decides how text becomes a vector
(decision P2), and the same recipe embeds at index time and at query time:

* task prefixes: nomic-embed-text v1.5 was trained with ``search_document: `` for indexed
  text and ``search_query: `` for questions; Ollama adds none (its template is
  ``{{ .Prompt }}``), so the adapter adds them, exactly once, and never strips or
  normalizes the text;
* ``truncate=False``: Ollama 0.34.0 otherwise cuts an input at the model's 2048-token
  context without any error; here an input that does not fit raises
  :class:`~src.errors.EmbeddingInputTooLongError` instead;
* dimension and context, checked against ``ollama show``; batch size (operational only).

The recipe's fingerprint covers only what changes the vectors (Gate 1 Q-E: batching does
not, measured max |diff| = 0), and the recipe lists the model digests it was verified
for (Q-F): an unknown digest behind the same name fails instead of being embedded with
possibly wrong prefixes. A model without a recipe is a configuration error.

:class:`OllamaEmbeddingAdapter` is a LlamaIndex ``BaseEmbedding`` over the project's
hardened Ollama client (decision P1: explicit host, finite timeout, no redirects, proxy
variables ignored). It validates every vector (count, dimension, finite, not all zero),
retries only transient failures (timeout, connection), isolates an over-long input in a
batch by its position and length, never by its text, and can prove which model it is
(:meth:`OllamaEmbeddingAdapter.identity`, no automatic ``pull``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import PrivateAttr

from src.errors import (
    EmbeddingInputTooLongError,
    EmbeddingRecipeError,
    InvalidEmbeddingError,
    ModelNotFoundError,
    OllamaUnavailableError,
    UnverifiedModelError,
)
from src.model_names import is_cloud_model, normalize_model_name
from src.ollama_client import EmbeddingResult, LocalModel, ModelInfo

logger = logging.getLogger(__name__)

_COMPONENT: Final = "embeddings"
_SHA256_HEX_CHARS: Final = 64

# nomic-embed-text:latest = v1.5 (137M, F16, 274,302,450 B), verified on this host on
# 2026-11-26/27: dimension 768, context 2048, unit-norm, template {{ .Prompt }}.
NOMIC_EMBED_TEXT_V15_DIGEST: Final = (
    "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
)


@dataclass(frozen=True, slots=True)
class EmbeddingRecipe:
    """How one model turns text into vectors; identical at index and query time."""

    model: str  # normalized name, e.g. "nomic-embed-text:latest"
    document_prefix: str
    query_prefix: str
    truncate: bool
    dimension: int
    context_length: int
    batch_size: int
    verified_digests: frozenset[str]

    def __post_init__(self) -> None:
        if self.model != normalize_model_name(self.model):
            raise EmbeddingRecipeError(
                f"Recipe model {self.model!r} must be normalized "
                f"({normalize_model_name(self.model)!r})."
            )
        for name in ("dimension", "context_length", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise EmbeddingRecipeError(f"Recipe {name} must be a positive integer: {value!r}")
        if not self.verified_digests or any(
            len(d) != _SHA256_HEX_CHARS or d.strip("0123456789abcdef")
            for d in self.verified_digests
        ):
            raise EmbeddingRecipeError(
                f"Recipe for {self.model!r} needs at least one verified sha256 model digest."
            )

    def vector_fields(self) -> dict[str, object]:
        """The fields that change the vectors: what the fingerprint covers (Q-E)."""
        return {
            "model": self.model,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
            "truncate": self.truncate,
            "dimension": self.dimension,
            "context_length": self.context_length,
        }

    def fingerprint(self) -> str:
        """sha256 of the canonical JSON of :meth:`vector_fields`."""
        canonical = json.dumps(
            self.vector_fields(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form for the index manifest."""
        return {
            **self.vector_fields(),
            "batch_size": self.batch_size,
            "verified_digests": sorted(self.verified_digests),
        }


RECIPES: Final[Mapping[str, EmbeddingRecipe]] = MappingProxyType(
    {
        "nomic-embed-text:latest": EmbeddingRecipe(
            model="nomic-embed-text:latest",
            document_prefix="search_document: ",
            query_prefix="search_query: ",
            truncate=False,
            dimension=768,
            context_length=2048,
            batch_size=10,
            verified_digests=frozenset({NOMIC_EMBED_TEXT_V15_DIGEST}),
        ),
    }
)


def recipe_for(model: str, recipes: Mapping[str, EmbeddingRecipe] = RECIPES) -> EmbeddingRecipe:
    """The recipe of ``model`` (``:latest`` implied); :class:`EmbeddingRecipeError` if none."""
    key = normalize_model_name(model)
    try:
        return recipes[key]
    except KeyError as exc:
        raise EmbeddingRecipeError(
            f"EMBEDDING_MODEL={model!r} has no embedding recipe (known: {sorted(recipes)}). "
            "Embedding without the model's task prefixes would silently misplace documents "
            "and queries. Use a known model, or verify the new one (prefixes, dimension, "
            "context, digest) and add its recipe to src/indexing/embeddings.py."
        ) from exc


@dataclass(frozen=True, slots=True)
class EmbedderIdentity:
    """What produced a set of vectors; ``digest`` and fingerprint are None if unverified."""

    model: str
    digest: str | None
    recipe_fingerprint: str | None
    dimension: int | None

    @property
    def verified(self) -> bool:
        return self.digest is not None and self.recipe_fingerprint is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "digest": self.digest,
            "recipe_fingerprint": self.recipe_fingerprint,
            "dimension": self.dimension,
            "verified": self.verified,
        }


@runtime_checkable
class IdentifiedEmbedding(Protocol):
    """An embedder that can prove which model and recipe it embeds with."""

    def identity(self) -> EmbedderIdentity: ...


@runtime_checkable
class ServedEmbedding(Protocol):
    """An embedder that can name the version of the server computing its vectors."""

    def server_version(self) -> str: ...


class EmbeddingClient(Protocol):
    """The part of the Ollama client the adapter needs (narrower than ``OllamaClient``)."""

    def version(self) -> str: ...

    def list_models(self) -> list[LocalModel]: ...

    def show(self, model: str) -> ModelInfo: ...

    def embed(
        self, model: str, inputs: Sequence[str], *, truncate: bool | None = None
    ) -> EmbeddingResult: ...


def validate_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int, dimension: int, model: str
) -> list[list[float]]:
    """``vectors`` as lists, or :class:`InvalidEmbeddingError` (positions only, no text)."""
    if len(vectors) != expected_count:
        raise InvalidEmbeddingError(
            f"{model!r} returned {len(vectors)} vectors for {expected_count} inputs."
        )
    checked: list[list[float]] = []
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        if len(values) != dimension:
            raise InvalidEmbeddingError(
                f"{model!r} returned a vector of dimension {len(values)} for input {index}; "
                f"the recipe expects {dimension}."
            )
        if not all(math.isfinite(value) for value in values):
            raise InvalidEmbeddingError(f"{model!r} returned NaN/inf values for input {index}.")
        if not any(values):
            raise InvalidEmbeddingError(f"{model!r} returned an all-zero vector for input {index}.")
        checked.append(values)
    return checked


class OllamaEmbeddingAdapter(BaseEmbedding):
    """LlamaIndex embedder over the hardened Ollama client, applying one recipe."""

    _client: EmbeddingClient = PrivateAttr()
    _recipe: EmbeddingRecipe = PrivateAttr()
    _retries: int = PrivateAttr()
    _backoff_s: float = PrivateAttr()
    _sleep: Callable[[float], None] = PrivateAttr()

    def __init__(
        self,
        client: EmbeddingClient,
        recipe: EmbeddingRecipe,
        *,
        retries: int = 2,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=recipe.model, embed_batch_size=recipe.batch_size, **kwargs)
        self._client = client
        self._recipe = recipe
        self._retries = max(0, retries)
        self._backoff_s = backoff_s
        self._sleep = sleep

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbeddingAdapter"

    @property
    def recipe(self) -> EmbeddingRecipe:
        return self._recipe

    # --- LlamaIndex hooks: the prefix is added here, exactly once --------------------------

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed([self._recipe.query_prefix + query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([self._recipe.document_prefix + text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed([self._recipe.document_prefix + text for text in texts])

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._get_text_embeddings(texts)

    # --- identity (Step 2: the model behind the name, verified, never pulled) ---------------

    def identity(self) -> EmbedderIdentity:
        """Verify the local model against the recipe; raise if it is not the verified one."""
        recipe = self._recipe
        listed = {normalize_model_name(m.name): m for m in self._client.list_models()}
        found = listed.get(recipe.model)
        if found is None:
            raise ModelNotFoundError(recipe.model, "make pull-models")
        if found.size_bytes <= 0 or is_cloud_model(found.name):
            raise UnverifiedModelError(
                f"{recipe.model!r} is listed by Ollama but has no local weights "
                f"(size {found.size_bytes} B): a remote/cloud model would send code to "
                "external servers. Configure a local model."
            )
        if found.digest not in recipe.verified_digests:
            raise UnverifiedModelError(
                f"{recipe.model!r} has digest {found.digest[:12]}, which its recipe was not "
                "verified for (a new model may be behind the same name). Check its prefixes, "
                "dimension and context, then add the digest to the recipe in "
                "src/indexing/embeddings.py."
            )
        info = self._client.show(recipe.model)
        if (info.context_length, info.embedding_length) != (
            recipe.context_length,
            recipe.dimension,
        ):
            raise UnverifiedModelError(
                f"{recipe.model!r} reports context {info.context_length} and dimension "
                f"{info.embedding_length}; its recipe expects {recipe.context_length} and "
                f"{recipe.dimension}."
            )
        return EmbedderIdentity(recipe.model, found.digest, recipe.fingerprint(), recipe.dimension)

    def server_version(self) -> str:
        """The Ollama server's version, for the index manifest."""
        return self._client.version()

    # --- internals ---------------------------------------------------------------------------

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            result = self._call(inputs)
        except EmbeddingInputTooLongError as exc:
            raise self._isolate_too_long(inputs) from exc
        return validate_vectors(
            result.vectors, len(inputs), self._recipe.dimension, self._recipe.model
        )

    def _call(self, inputs: list[str]) -> EmbeddingResult:
        attempt = 0
        while True:
            try:
                return self._client.embed(
                    self._recipe.model, inputs, truncate=self._recipe.truncate
                )
            except OllamaUnavailableError:
                if attempt >= self._retries:
                    raise
                attempt += 1
                logger.warning(
                    "embedding.retry",
                    extra={"component": _COMPONENT, "attempt": attempt, "batch": len(inputs)},
                )
                self._sleep(self._backoff_s * attempt)

    def _isolate_too_long(self, inputs: list[str]) -> EmbeddingInputTooLongError:
        """Find the first input that does not fit, by embedding them one by one."""
        position = 0
        if len(inputs) > 1:
            for index, text in enumerate(inputs):
                try:
                    self._call([text])
                except EmbeddingInputTooLongError:
                    position = index
                    break
        length = len(inputs[position])
        return EmbeddingInputTooLongError(
            f"Input {position} of a batch of {len(inputs)} ({length} characters, prefix "
            f"included) exceeds the {self._recipe.context_length}-token context of "
            f"{self._recipe.model!r}; truncation is off on purpose. Lower max_chunk_chars or "
            "shorten the visible metadata.",
            index=position,
            length=length,
        )


def get_embedding_model(
    *, client: EmbeddingClient | None = None, recipe: EmbeddingRecipe | None = None
) -> OllamaEmbeddingAdapter:
    """The project's embedder, from configuration; builds no connection and does no I/O.

    ``client`` and ``recipe`` are injectable (tests); by default they come from
    ``OLLAMA_BASE_URL``, ``REQUEST_TIMEOUT`` and ``EMBEDDING_MODEL`` in :mod:`src.config`,
    imported here so that an invalid environment surfaces as ``ConfigurationError`` at
    call time.
    """
    if client is None or recipe is None:
        from src import config  # lazy on purpose, see the docstring
        from src.ollama_client import OllamaHttpClient

        if recipe is None:
            recipe = recipe_for(config.EMBEDDING_MODEL)
        if client is None:
            client = OllamaHttpClient(config.OLLAMA_BASE_URL, config.REQUEST_TIMEOUT)
    return OllamaEmbeddingAdapter(client, recipe)
