"""Deterministic fake embedder for tests (Phase 3, Step 1): no model, no network.

Each text becomes a small, L2-normalized bag-of-tokens vector: every lower-cased
alphanumeric token adds one to the coordinate chosen by its sha256 (``hashlib``, never
``hash()``, which changes between processes). Texts that share words are close, so
similarity rankings mean something; identical texts give identical vectors.

It records every call (batch sizes and texts embedded), can be told to raise any
exception on a given call (fault injection, ``KeyboardInterrupt`` included), and has an
identity like the real adapter's (:class:`src.indexing.embeddings.EmbedderIdentity`), so
the orchestrator can verify and reuse its vectors.

Not a test module (pytest does not collect it).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Final

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import PrivateAttr

from src.indexing.embeddings import EmbedderIdentity

_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")
FAKE_MODEL: Final = "fake-embedding:latest"


class FakeEmbedding(BaseEmbedding):
    """Token-hashing embedder with call accounting and fault injection."""

    dimension: int = 64
    digest: str = "f" * 64

    _batches: list[int] = PrivateAttr(default_factory=list)
    _failures: dict[int, BaseException] = PrivateAttr(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        return "FakeEmbedding"

    # --- accounting and faults ---------------------------------------------------------------

    @property
    def batches(self) -> list[int]:
        """Sizes of the text batches embedded so far (queries count as batches of 1)."""
        return list(self._batches)

    @property
    def texts_embedded(self) -> int:
        return sum(self._batches)

    def fail_on_call(self, call_number: int, error: BaseException) -> None:
        """Raise ``error`` on the ``call_number``-th call (1-based), before embedding."""
        self._failures[call_number] = error

    def identity(self) -> EmbedderIdentity:
        fingerprint = hashlib.sha256(f"fake:{self.dimension}".encode()).hexdigest()
        return EmbedderIdentity(FAKE_MODEL, self.digest, fingerprint, self.dimension)

    # --- the vectors -------------------------------------------------------------------------

    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            slot = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % self.dimension
            values[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:  # no token at all: a fixed unit vector, never the zero vector
            return [1.0] + [0.0] * (self.dimension - 1)
        return [v / norm for v in values]

    def _embed(self, texts: list[str]) -> list[Embedding]:
        self._batches.append(len(texts))
        failure = self._failures.pop(len(self._batches), None)
        if failure is not None:
            raise failure
        return [self.vector(text) for text in texts]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)
