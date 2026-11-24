"""Tests for the ollama adapter, driven by httpx.MockTransport (no real Ollama).

The proxy test is the exception: it opens a real loopback socket to a closed port, which
the suite-wide network guard allows.
"""

import json
import socket
from collections.abc import Callable

import httpx
import pytest

from src.errors import ModelNotFoundError, OllamaUnavailableError, VerificationError
from src.ollama_client import (
    EmbeddingResult,
    Generation,
    LocalModel,
    ModelInfo,
    OllamaHttpClient,
    RunningModel,
)

BASE_URL = "http://localhost:11434"
EXTERNAL_PROXY = "http://203.0.113.10:3128"  # TEST-NET-3: never routed

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, timeout: float = 5.0) -> OllamaHttpClient:
    return OllamaHttpClient(BASE_URL, timeout, transport=httpx.MockTransport(handler))


def test_list_models_maps_name_digest_and_size() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        models = [{"model": "qwen2.5-coder:7b", "digest": "dae161e27b0e", "size": 4683087561}]
        return httpx.Response(200, json={"models": models})

    assert _client(handler).list_models() == [
        LocalModel(name="qwen2.5-coder:7b", digest="dae161e27b0e", size_bytes=4683087561)
    ]
    assert [(url.host, url.port, url.path) for url in seen] == [("localhost", 11434, "/api/tags")]


def test_show_extracts_architecture_lengths_and_num_ctx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "nomic-embed-text"
        return httpx.Response(
            200,
            json={
                "model_info": {
                    "general.architecture": "nomic-bert",
                    "nomic-bert.context_length": 2048,
                    "nomic-bert.embedding_length": 768,
                },
                "parameters": 'num_ctx                        8192\nstop    "<|im_end|>"',
                "details": {"parent_model": "", "family": "nomic-bert"},
                "capabilities": ["embedding"],
            },
        )

    assert _client(handler).show("nomic-embed-text") == ModelInfo(
        architecture="nomic-bert",
        context_length=2048,
        embedding_length=768,
        num_ctx=8192,
        parent_model=None,
        capabilities=("embedding",),
    )


def test_generate_sends_options_without_streaming_and_maps_counters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/generate"
        assert body["model"] == "m"
        assert body["prompt"] == "p"
        assert body["options"] == {"temperature": 0, "seed": 42, "num_predict": 8}
        assert body["stream"] is False
        assert body["keep_alive"] == 0
        return httpx.Response(
            200,
            json={
                "model": "m",
                "response": "def f(): pass",
                "done": True,
                "total_duration": 5_000,
                "load_duration": 1_000,
                "prompt_eval_count": 10,
                "eval_count": 7,
                "eval_duration": 2_000,
            },
        )

    generation = _client(handler).generate(
        "m", "p", {"temperature": 0, "seed": 42, "num_predict": 8}, keep_alive=0
    )

    assert generation == Generation(
        text="def f(): pass",
        total_duration_ns=5_000,
        load_duration_ns=1_000,
        prompt_eval_count=10,
        eval_count=7,
        eval_duration_ns=2_000,
    )


def test_embed_maps_vectors_and_durations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"] == ["a", "b"]
        return httpx.Response(
            200,
            json={
                "model": "e",
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "total_duration": 9,
                "load_duration": 3,
            },
        )

    assert _client(handler).embed("e", ["a", "b"]) == EmbeddingResult(
        vectors=[[0.1, 0.2], [0.3, 0.4]], total_duration_ns=9, load_duration_ns=3
    )


def test_running_models_reports_the_allocated_context_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        loaded = {"name": "m-rag:latest", "size": 5_600_000_000, "context_length": 16384}
        return httpx.Response(200, json={"models": [loaded]})

    assert _client(handler).running_models() == [
        RunningModel(name="m-rag:latest", context_length=16384, size_bytes=5_600_000_000)
    ]


def test_http_404_becomes_model_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'ghost' not found"})

    with pytest.raises(ModelNotFoundError, match="ollama pull ghost") as exc_info:
        _client(handler).show("ghost")

    assert exc_info.value.model == "ghost"


def test_other_http_errors_become_verification_errors_with_the_server_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model requires more system memory"})

    with pytest.raises(VerificationError, match=r"HTTP 500.*more system memory"):
        _client(handler).generate("m", "p", {})


def test_timeout_becomes_ollama_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(OllamaUnavailableError, match=r"within 5 s"):
        _client(handler, timeout=5.0).generate("m", "p", {})


def test_connection_failure_becomes_ollama_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OllamaUnavailableError, match="Cannot connect"):
        _client(handler).list_models()


def test_redirects_are_not_followed() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(307, headers={"Location": "http://203.0.113.10/api/tags"})

    with pytest.raises(VerificationError, match="HTTP 307"):
        _client(handler).list_models()

    assert len(seen) == 1


def test_proxy_and_ollama_host_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, EXTERNAL_PROXY)
    monkeypatch.setenv("all_proxy", EXTERNAL_PROXY)
    monkeypatch.setenv("OLLAMA_HOST", "203.0.113.10:11434")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    with socket.socket() as placeholder:
        placeholder.bind(("127.0.0.1", 0))  # bound but not listening: connections are refused
        port = placeholder.getsockname()[1]
        client = OllamaHttpClient(f"http://127.0.0.1:{port}", timeout=5.0)

        # A proxied or OLLAMA_HOST-redirected request would hit the network guard instead.
        with pytest.raises(OllamaUnavailableError, match="Cannot connect"):
            client.list_models()
