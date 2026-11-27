"""Phase 3, Steps 1-2: the embedding recipe, the hardened adapter and the fake embedder.

Hermetic: the adapter talks to ``httpx.MockTransport``, to an in-memory client, or (proxy
test) to a fake Ollama on a loopback socket. No model and no external network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import http.server
import json
import math
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
from fake_embedding import FAKE_MODEL, FakeEmbedding
from llama_index.core import MockEmbedding

from src.errors import (
    ConfigurationError,
    EmbeddingInputTooLongError,
    EmbeddingRecipeError,
    InvalidEmbeddingError,
    ModelNotFoundError,
    OllamaUnavailableError,
    UnverifiedModelError,
    VerificationError,
)
from src.indexing.embeddings import (
    NOMIC_EMBED_TEXT_V15_DIGEST,
    RECIPES,
    EmbeddingRecipe,
    IdentifiedEmbedding,
    OllamaEmbeddingAdapter,
    get_embedding_model,
    recipe_for,
    validate_vectors,
)
from src.ollama_client import EmbeddingResult, LocalModel, ModelInfo, OllamaHttpClient

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://localhost:11434"
EXTERNAL_PROXY = "http://203.0.113.10:3128"  # TEST-NET-3: never routed
NOMIC = RECIPES["nomic-embed-text:latest"]
# Small recipe for tests that build vectors by hand (dimension 3).
TINY = dataclasses.replace(NOMIC, dimension=3, batch_size=10)
# Pinned: changing the vector fields of the nomic recipe must be a deliberate decision.
NOMIC_FINGERPRINT = "546cb922dd10be43bb41dfb6ef3818579b7f0f3e12cb8498609dd87cd453ca81"

Handler = Callable[[httpx.Request], httpx.Response]


def _unit(index: int, dimension: int = 3) -> list[float]:
    return [1.0 if i == index % dimension else 0.0 for i in range(dimension)]


@dataclasses.dataclass
class FakeClient:
    """In-memory EmbeddingClient: records inputs, answers with unit vectors."""

    dimension: int = 3
    models: list[LocalModel] = dataclasses.field(
        default_factory=lambda: [
            LocalModel("nomic-embed-text:latest", NOMIC_EMBED_TEXT_V15_DIGEST, 274_302_450)
        ]
    )
    info: ModelInfo = dataclasses.field(
        default_factory=lambda: ModelInfo("nomic-bert", 2048, 768, 8192, None, ("embedding",))
    )
    vectors: Callable[[Sequence[str]], list[list[float]]] | None = None
    failures: list[Exception] = dataclasses.field(default_factory=list)
    too_long_marker: str | None = None
    calls: list[tuple[list[str], bool | None]] = dataclasses.field(default_factory=list)

    def version(self) -> str:
        return "0.34.0"

    def list_models(self) -> list[LocalModel]:
        return list(self.models)

    def show(self, model: str) -> ModelInfo:
        return self.info

    def embed(
        self, model: str, inputs: Sequence[str], *, truncate: bool | None = None
    ) -> EmbeddingResult:
        self.calls.append((list(inputs), truncate))
        if self.failures:
            raise self.failures.pop(0)
        if self.too_long_marker is not None and any(self.too_long_marker in i for i in inputs):
            raise EmbeddingInputTooLongError("too long")
        vectors = (
            self.vectors(inputs)
            if self.vectors
            else [_unit(i, self.dimension) for i in range(len(inputs))]
        )
        return EmbeddingResult(vectors, None, None, prompt_eval_count=len(inputs))


def _adapter(client: FakeClient, recipe: EmbeddingRecipe = TINY) -> OllamaEmbeddingAdapter:
    return OllamaEmbeddingAdapter(client, recipe, sleep=lambda _: None)


def _http_adapter(handler: Handler, recipe: EmbeddingRecipe = TINY) -> OllamaEmbeddingAdapter:
    client = OllamaHttpClient(BASE_URL, 5.0, transport=httpx.MockTransport(handler))
    return OllamaEmbeddingAdapter(client, recipe, sleep=lambda _: None)


# --- the recipe ------------------------------------------------------------------------------


def test_the_nomic_recipe_matches_the_measured_model() -> None:
    assert NOMIC.document_prefix == "search_document: "
    assert NOMIC.query_prefix == "search_query: "
    assert NOMIC.truncate is False
    assert (NOMIC.dimension, NOMIC.context_length, NOMIC.batch_size) == (768, 2048, 10)
    assert NOMIC.verified_digests == {NOMIC_EMBED_TEXT_V15_DIGEST}


def test_the_fingerprint_is_pinned_and_covers_only_vector_fields() -> None:
    assert NOMIC.fingerprint() == NOMIC_FINGERPRINT
    assert dataclasses.replace(NOMIC, batch_size=32).fingerprint() == NOMIC_FINGERPRINT
    for change in (
        {"document_prefix": ""},
        {"query_prefix": "query: "},
        {"truncate": True},
        {"dimension": 512},
        {"context_length": 8192},
        {"model": "other:latest"},
    ):
        assert dataclasses.replace(NOMIC, **change).fingerprint() != NOMIC_FINGERPRINT


def test_the_manifest_form_records_batch_size_and_digests() -> None:
    data = NOMIC.to_dict()

    assert data["batch_size"] == 10
    assert data["verified_digests"] == [NOMIC_EMBED_TEXT_V15_DIGEST]
    assert json.loads(json.dumps(data)) == data


@pytest.mark.parametrize(
    "change",
    [
        {"model": "nomic-embed-text"},
        {"dimension": 0},
        {"batch_size": -1},
        {"context_length": 1.5},
        {"verified_digests": frozenset()},
        {"verified_digests": frozenset({"abc"})},
    ],
    ids=["unnormalized", "dimension", "batch", "context", "no_digest", "bad_digest"],
)
def test_an_invalid_recipe_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(EmbeddingRecipeError):
        dataclasses.replace(NOMIC, **change)  # type: ignore[arg-type]


def test_recipes_are_found_by_normalized_name() -> None:
    assert recipe_for("nomic-embed-text") is NOMIC
    assert recipe_for("nomic-embed-text:latest") is NOMIC


def test_a_model_without_a_recipe_is_a_configuration_error() -> None:
    with pytest.raises(EmbeddingRecipeError, match="no embedding recipe") as raised:
        recipe_for("mxbai-embed-large")
    assert isinstance(raised.value, ConfigurationError)


# --- the adapter: prefixes, truncate, batching, no text normalization ------------------------


def test_documents_and_queries_get_their_prefix_exactly_once_with_truncate_off() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(
            200, json={"embeddings": [_unit(i) for i in range(3)][: len(body["input"])]}
        )

    adapter = _http_adapter(handler)
    adapter.get_text_embedding("def f(): pass")
    adapter.get_query_embedding("how is f used?")
    adapter.get_text_embedding_batch(["a", "b"])

    assert [b["input"] for b in bodies] == [
        ["search_document: def f(): pass"],
        ["search_query: how is f used?"],
        ["search_document: a", "search_document: b"],
    ]
    assert all(b["truncate"] is False and b["model"] == "nomic-embed-text:latest" for b in bodies)


def test_leading_whitespace_and_newlines_reach_the_model_unchanged() -> None:
    client = FakeClient()
    text = "\n\n    def indented():\n\treturn 1\n\n"

    _adapter(client).get_text_embedding(text)
    _adapter(client).get_query_embedding("  spaced query\n")

    assert client.calls[0][0] == ["search_document: " + text]
    assert client.calls[1][0] == ["search_query:   spaced query\n"]


def test_batches_follow_the_recipe_batch_size() -> None:
    client = FakeClient()

    vectors = _adapter(client).get_text_embedding_batch([f"t{i}" for i in range(25)])

    assert [len(inputs) for inputs, _ in client.calls] == [10, 10, 5]
    assert len(vectors) == 25


def test_async_variants_apply_the_same_recipe() -> None:
    client = FakeClient()
    adapter = _adapter(client)

    asyncio.run(adapter.aget_query_embedding("q"))
    asyncio.run(adapter.aget_text_embedding("d"))
    asyncio.run(adapter.aget_text_embedding_batch(["x", "y"]))

    assert [inputs for inputs, _ in client.calls] == [
        ["search_query: q"],
        ["search_document: d"],
        ["search_document: x", "search_document: y"],
    ]


# --- vector validation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vectors", "match"),
    [
        (lambda inputs: [], "returned 0 vectors for 1 inputs"),
        (lambda inputs: [[1.0, 0.0]], "dimension 2"),
        (lambda inputs: [[math.nan, 0.0, 1.0]], "NaN/inf"),
        (lambda inputs: [[0.0, 0.0, 0.0]], "all-zero"),
    ],
    ids=["count", "dimension", "nan", "zero"],
)
def test_invalid_vectors_are_rejected(
    vectors: Callable[[Sequence[str]], list[list[float]]], match: str
) -> None:
    with pytest.raises(InvalidEmbeddingError, match=match):
        _adapter(FakeClient(vectors=vectors)).get_text_embedding("x")


def test_validate_vectors_returns_plain_float_lists() -> None:
    assert validate_vectors([(1, 0, 0)], 1, 3, "m") == [[1.0, 0.0, 0.0]]


# --- over-long input: isolated by position and length, never by text -------------------------


def test_an_over_long_input_in_a_batch_is_isolated_without_its_text() -> None:
    marker = "SECRET_CODE_TEXT_93a1"
    client = FakeClient(too_long_marker=marker)
    texts = ["a", "b", f"{marker} " * 50, "c"]

    with pytest.raises(EmbeddingInputTooLongError) as raised:
        _adapter(client).get_text_embedding_batch(texts)

    error = raised.value
    assert (error.index, error.length) == (2, len("search_document: ") + len(texts[2]))
    assert marker not in str(error)
    assert isinstance(error, VerificationError)


def test_the_real_client_answer_is_isolated_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        if any(len(i) > 100 for i in inputs):
            return httpx.Response(
                400, json={"error": "the input length exceeds the context length"}
            )
        return httpx.Response(200, json={"embeddings": [_unit(0)] * len(inputs)})

    with pytest.raises(EmbeddingInputTooLongError) as raised:
        _http_adapter(handler).get_text_embedding_batch(["short", "x" * 500])

    assert raised.value.index == 1


# --- failures and retries --------------------------------------------------------------------


def test_transient_failures_are_retried_a_bounded_number_of_times() -> None:
    sleeps: list[float] = []
    client = FakeClient(failures=[OllamaUnavailableError("down")] * 3)
    adapter = OllamaEmbeddingAdapter(client, TINY, retries=2, sleep=sleeps.append)

    with pytest.raises(OllamaUnavailableError):
        adapter.get_text_embedding("x")

    assert len(client.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_a_transient_failure_followed_by_success_recovers() -> None:
    client = FakeClient(failures=[OllamaUnavailableError("blip")])

    assert len(_adapter(client).get_text_embedding("x")) == 3
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "error",
    [VerificationError("bad"), ModelNotFoundError("m"), EmbeddingInputTooLongError("long")],
    ids=["verification", "missing_model", "too_long"],
)
def test_non_transient_failures_are_not_retried(error: Exception) -> None:
    client = FakeClient(failures=[error, error])

    with pytest.raises(type(error)):
        _adapter(client).get_text_embedding("x")

    assert len(client.calls) <= 2  # 1 call, plus 1 isolation call for an over-long input


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (lambda r: httpx.Response(404, json={"error": "model not found"}), ModelNotFoundError),
        (lambda r: httpx.Response(400, json={"error": "invalid input"}), VerificationError),
    ],
    ids=["404", "400"],
)
def test_http_errors_map_to_typed_errors(answer: Handler, error: type[Exception]) -> None:
    with pytest.raises(error):
        _http_adapter(answer).get_text_embedding("x")


@pytest.mark.parametrize(
    "transport_error",
    [httpx.ReadTimeout, httpx.ConnectError],
    ids=["timeout", "connection"],
)
def test_timeouts_and_refused_connections_are_retried_then_unavailable(
    transport_error: type[httpx.TransportError],
) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise transport_error("no", request=request)

    with pytest.raises(OllamaUnavailableError):
        _http_adapter(handler).get_text_embedding("x")
    assert len(attempts) == 3


# --- identity: the local model is the verified one, never pulled -----------------------------


def test_the_identity_of_the_verified_model() -> None:
    identity = _adapter(FakeClient(), NOMIC).identity()

    assert identity.model == "nomic-embed-text:latest"
    assert identity.digest == NOMIC_EMBED_TEXT_V15_DIGEST
    assert identity.recipe_fingerprint == NOMIC_FINGERPRINT
    assert (identity.dimension, identity.verified) == (768, True)


@pytest.mark.parametrize(
    ("client", "error", "match"),
    [
        (FakeClient(models=[]), ModelNotFoundError, "make pull-models"),
        (
            FakeClient(
                models=[LocalModel("nomic-embed-text:latest", NOMIC_EMBED_TEXT_V15_DIGEST, 0)]
            ),
            UnverifiedModelError,
            "no local weights",
        ),
        (
            FakeClient(models=[LocalModel("nomic-embed-text:latest", "c" * 64, 274_302_450)]),
            UnverifiedModelError,
            "not verified for",
        ),
        (
            FakeClient(info=ModelInfo("nomic-bert", 8192, 768, None, None, ())),
            UnverifiedModelError,
            "context 8192",
        ),
        (
            FakeClient(info=ModelInfo("nomic-bert", 2048, 1024, None, None, ())),
            UnverifiedModelError,
            "dimension 1024",
        ),
    ],
    ids=["missing", "no_weights", "unknown_digest", "context", "dimension"],
)
def test_an_unverified_model_is_refused(
    client: FakeClient, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        _adapter(client, NOMIC).identity()


def test_embedders_that_can_prove_their_identity_are_recognized() -> None:
    assert isinstance(_adapter(FakeClient()), IdentifiedEmbedding)
    assert isinstance(FakeEmbedding(), IdentifiedEmbedding)
    assert not isinstance(MockEmbedding(embed_dim=3), IdentifiedEmbedding)


# --- construction does no I/O ----------------------------------------------------------------


def test_get_embedding_model_builds_from_configuration_without_any_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("constructing the embedder must not open a connection")

    monkeypatch.setattr(socket.socket, "connect", no_connect)

    adapter = get_embedding_model()

    assert adapter.recipe is NOMIC
    assert (adapter.model_name, adapter.embed_batch_size) == ("nomic-embed-text:latest", 10)


def test_get_embedding_model_accepts_injected_parts() -> None:
    client = FakeClient()

    adapter = get_embedding_model(client=client, recipe=TINY)
    adapter.get_query_embedding("q")

    assert client.calls == [(["search_query: q"], False)]


# --- proxies are ignored: a fake Ollama on loopback answers, the TEST-NET proxy never ---------


class _FakeOllama(http.server.BaseHTTPRequestHandler):
    hits: ClassVar[list[str]] = []

    def do_POST(self) -> None:  # http.server's method name
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        type(self).hits.append(self.path)
        payload = json.dumps({"embeddings": [_unit(0)] * len(body["input"])}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # the stdlib's signature
        return


@pytest.fixture
def fake_ollama() -> Iterator[str]:
    _FakeOllama.hits = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_variables_are_ignored_against_a_fake_loopback_ollama(
    fake_ollama: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, EXTERNAL_PROXY)
    monkeypatch.setenv("OLLAMA_HOST", "203.0.113.10:11434")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    adapter = get_embedding_model(client=OllamaHttpClient(fake_ollama, 5.0), recipe=TINY)

    assert len(adapter.get_query_embedding("q")) == 3
    assert _FakeOllama.hits == ["/api/embed"]  # a proxied request would hit the guard instead


# --- Step 1: the deterministic fake embedder -------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_the_fake_embedder_is_normalized_and_meaningful() -> None:
    fake = FakeEmbedding()
    token = fake.vector("def verify_token(token): check signature and expiry of the token")
    question = fake.vector("how is the token signature checked")
    unrelated = fake.vector("parse csv rows into a dataframe")

    assert math.isclose(math.sqrt(_cosine(token, token)), 1.0)
    assert _cosine(question, token) > _cosine(question, unrelated)
    assert fake.vector("") == [1.0] + [0.0] * 63  # never the zero vector


def test_the_fake_embedder_counts_calls_and_injects_faults() -> None:
    fake = FakeEmbedding()
    fake.fail_on_call(2, KeyboardInterrupt())

    fake.get_text_embedding_batch(["a", "b", "c"])
    with pytest.raises(KeyboardInterrupt):
        fake.get_query_embedding("q")

    assert fake.batches == [3, 1]
    assert fake.texts_embedded == 4
    assert fake.identity().model == FAKE_MODEL


def test_the_fake_embedder_is_identical_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    code = (
        "import json, sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/tests'];"
        "from fake_embedding import FakeEmbedding;"
        "print(json.dumps(FakeEmbedding().vector('validate the access token')))"
    )
    outputs = []
    for seed in ("1", "2"):
        result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
            [sys.executable, "-I", "-c", code, str(REPO_ROOT)],
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": seed, "HOME": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == FakeEmbedding().vector("validate the access token")


def test_the_adapter_reports_the_server_version() -> None:
    assert _adapter(FakeClient()).server_version() == "0.34.0"
