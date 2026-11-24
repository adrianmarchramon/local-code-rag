"""Ollama access behind a small Protocol, plus the adapter over the ``ollama`` package.

Checks and tests depend on :class:`OllamaClient` and the plain dataclasses below, never on
``ollama`` types, so they can run against in-memory fakes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Protocol

import httpx

import ollama
from src.errors import ModelNotFoundError, OllamaUnavailableError, VerificationError

_MAX_SERVER_MESSAGE_CHARS: Final = 300


@dataclass(frozen=True, slots=True)
class LocalModel:
    """One entry of the local model store (``ollama list``)."""

    name: str
    digest: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What ``ollama show`` reports about a model (unreported fields are None)."""

    architecture: str | None
    context_length: int | None
    embedding_length: int | None
    num_ctx: int | None
    parent_model: str | None
    capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Generation:
    """A non-streamed completion with Ollama's own counters (durations in ns)."""

    text: str
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    eval_count: int | None
    eval_duration_ns: int | None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vectors for a batch of inputs (durations in ns)."""

    vectors: list[list[float]]
    total_duration_ns: int | None
    load_duration_ns: int | None


@dataclass(frozen=True, slots=True)
class RunningModel:
    """A model currently loaded by the server (``ollama ps``)."""

    name: str
    context_length: int | None
    size_bytes: int


class OllamaClient(Protocol):
    """The subset of the Ollama API this project relies on."""

    def list_models(self) -> list[LocalModel]: ...

    def show(self, model: str) -> ModelInfo: ...

    def generate(
        self,
        model: str,
        prompt: str,
        options: Mapping[str, Any],
        keep_alive: float | str | None = None,
    ) -> Generation: ...

    def embed(self, model: str, inputs: Sequence[str]) -> EmbeddingResult: ...


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _num_ctx_from_parameters(parameters: str | None) -> int | None:
    for line in (parameters or "").splitlines():
        key, _, value = line.strip().partition(" ")
        if key == "num_ctx":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


class OllamaHttpClient:
    """:class:`OllamaClient` over ``ollama.Client``, configured for local-only use.

    * explicit host from configuration (``OLLAMA_HOST`` is never consulted);
    * finite timeout (the package default is none);
    * redirects not followed, so a local server cannot bounce a request elsewhere;
    * ``trust_env=False``, so proxy variables can never route local traffic via a proxy.

    ``transport`` exists for tests (``httpx.MockTransport``); production leaves it None.
    """

    def __init__(
        self, base_url: str, timeout: float, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        options: dict[str, Any] = {
            "timeout": timeout,
            "follow_redirects": False,
            "trust_env": False,
        }
        if transport is not None:
            options["transport"] = transport
        self._client = ollama.Client(host=base_url, **options)
        self._base_url = base_url
        self._timeout = timeout

    def list_models(self) -> list[LocalModel]:
        with self._translate_errors():
            response = self._client.list()
        return [
            LocalModel(name=m.model or "", digest=m.digest or "", size_bytes=int(m.size or 0))
            for m in response.models
        ]

    def show(self, model: str) -> ModelInfo:
        with self._translate_errors(model):
            response = self._client.show(model)
        info: Mapping[str, Any] = response.modelinfo or {}
        architecture = info.get("general.architecture")
        prefix = architecture if isinstance(architecture, str) else ""
        parent = response.details.parent_model if response.details else None
        return ModelInfo(
            architecture=prefix or None,
            context_length=_as_int(info.get(f"{prefix}.context_length")),
            embedding_length=_as_int(info.get(f"{prefix}.embedding_length")),
            num_ctx=_num_ctx_from_parameters(response.parameters),
            parent_model=parent or None,
            capabilities=tuple(response.capabilities or ()),
        )

    def generate(
        self,
        model: str,
        prompt: str,
        options: Mapping[str, Any],
        keep_alive: float | str | None = None,
    ) -> Generation:
        with self._translate_errors(model):
            response = self._client.generate(
                model=model,
                prompt=prompt,
                options=dict(options),
                keep_alive=keep_alive,
                stream=False,
            )
        return Generation(
            text=response.response or "",
            total_duration_ns=response.total_duration,
            load_duration_ns=response.load_duration,
            prompt_eval_count=response.prompt_eval_count,
            eval_count=response.eval_count,
            eval_duration_ns=response.eval_duration,
        )

    def embed(self, model: str, inputs: Sequence[str]) -> EmbeddingResult:
        with self._translate_errors(model):
            response = self._client.embed(model=model, input=list(inputs))
        return EmbeddingResult(
            vectors=[[float(x) for x in vector] for vector in response.embeddings],
            total_duration_ns=response.total_duration,
            load_duration_ns=response.load_duration,
        )

    def running_models(self) -> list[RunningModel]:
        """Models loaded right now, with the context length the server allocated."""
        with self._translate_errors():
            response = self._client.ps()
        return [
            RunningModel(
                name=m.name or m.model or "",
                context_length=m.context_length,
                size_bytes=int(m.size or 0),
            )
            for m in response.models
        ]

    @contextmanager
    def _translate_errors(self, model: str | None = None) -> Iterator[None]:
        """Map ``ollama``/``httpx`` failures to this project's typed errors."""
        try:
            yield
        except ollama.ResponseError as exc:
            if exc.status_code == 404 and model is not None:
                raise ModelNotFoundError(model) from exc
            message = str(exc.error)[:_MAX_SERVER_MESSAGE_CHARS]
            raise VerificationError(
                f"Ollama at {self._base_url} returned HTTP {exc.status_code}: {message}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailableError(
                f"Ollama at {self._base_url} did not answer within {self._timeout:g} s. A cold "
                "model load on CPU can be slow: retry, or raise REQUEST_TIMEOUT."
            ) from exc
        except ConnectionError as exc:  # what ollama raises for httpx.ConnectError
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self._base_url}. Start it (`sudo systemctl "
                "start ollama` or `ollama serve`) and check OLLAMA_BASE_URL."
            ) from exc
        except httpx.TransportError as exc:
            raise OllamaUnavailableError(
                f"Connection to Ollama at {self._base_url} failed ({type(exc).__name__})."
            ) from exc
