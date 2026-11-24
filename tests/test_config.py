"""Tests for src.config: defaults, environment overrides, and fail-closed validation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from src.config import Settings, load_settings
from src.errors import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_match_the_agreed_phase0_decisions() -> None:
    assert load_settings({}) == Settings(
        ollama_base_url="http://localhost:11434",
        code_model="qwen2.5-coder-7b-rag",
        embedding_model="nomic-embed-text",
        request_timeout=180.0,
        llm_context_window=16384,
        local_allowed_hosts=frozenset(),
    )


def test_environment_overrides_every_contract_value() -> None:
    settings = load_settings(
        {
            "OLLAMA_BASE_URL": "http://127.0.0.1:11500",
            "CODE_MODEL": "qwen2.5-coder:7b",
            "EMBEDDING_MODEL": "nomic-embed-text:v1.5",
            "REQUEST_TIMEOUT": "42.5",
            "LLM_CONTEXT_WINDOW": "8192",
        }
    )

    assert settings.ollama_base_url == "http://127.0.0.1:11500"
    assert settings.code_model == "qwen2.5-coder:7b"
    assert settings.embedding_model == "nomic-embed-text:v1.5"
    assert settings.request_timeout == 42.5
    assert settings.llm_context_window == 8192


def test_contract_constants_exist_and_mirror_the_loaded_settings() -> None:
    assert isinstance(config.OLLAMA_BASE_URL, str)
    assert isinstance(config.CODE_MODEL, str)
    assert isinstance(config.EMBEDDING_MODEL, str)
    assert isinstance(config.REQUEST_TIMEOUT, float)
    assert isinstance(config.LLM_CONTEXT_WINDOW, int)
    assert config.SETTINGS.ollama_base_url == config.OLLAMA_BASE_URL
    assert config.SETTINGS.code_model == config.CODE_MODEL
    assert config.SETTINGS.embedding_model == config.EMBEDDING_MODEL
    assert config.SETTINGS.request_timeout == config.REQUEST_TIMEOUT
    assert config.SETTINGS.llm_context_window == config.LLM_CONTEXT_WINDOW
    assert config.SETTINGS.local_allowed_hosts == config.LOCAL_ALLOWED_HOSTS


def test_derived_model_is_distinct_from_its_base() -> None:
    # `make models` creates CODE_DERIVED_MODEL; it must never overwrite the pulled base.
    assert len({config.CODE_DERIVED_MODEL, config.CODE_BASE_MODEL}) == 2
    assert config.DEFAULT_LLM_CONTEXT_WINDOW == config.DERIVED_MODEL_NUM_CTX


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://LOCALHOST:11434/",
        "http://127.0.0.1:11434",
        "http://127.0.0.2:11434",
        "http://[::1]:11434",
        "https://localhost",
    ],
)
def test_loopback_base_urls_are_accepted(url: str) -> None:
    assert load_settings({"OLLAMA_BASE_URL": url}).ollama_base_url == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "http://192.168.1.10:11434",
        "http://10.0.0.5:11434",
        "http://0.0.0.0:11434",
        "http://[2001:db8::1]:11434",
        "http://localhost.example.com:11434",
        "http://ollama:11434",
    ],
)
def test_non_loopback_base_urls_are_rejected(url: str) -> None:
    with pytest.raises(ConfigurationError, match="LOCAL_ALLOWED_HOSTS"):
        load_settings({"OLLAMA_BASE_URL": url})


def test_allowlisted_host_is_accepted() -> None:
    settings = load_settings(
        {"OLLAMA_BASE_URL": "http://ollama:11434", "LOCAL_ALLOWED_HOSTS": " Ollama , qdrant "}
    )

    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.local_allowed_hosts == frozenset({"ollama", "qdrant"})


def test_allowlist_does_not_open_other_hosts() -> None:
    with pytest.raises(ConfigurationError, match=r"example\.com"):
        load_settings(
            {"OLLAMA_BASE_URL": "http://example.com:11434", "LOCAL_ALLOWED_HOSTS": "ollama"}
        )


@pytest.mark.parametrize(
    "url",
    [
        "localhost:11434",
        "ftp://localhost:11434",
        "http://localhost:99999",
        "http://localhost:11434/api",
        "http://localhost:11434?x=1",
        "http://localhost:11434#frag",
        "http://",
        "http://[::1",
    ],
)
def test_malformed_base_urls_are_rejected(url: str) -> None:
    with pytest.raises(ConfigurationError, match="OLLAMA_BASE_URL"):
        load_settings({"OLLAMA_BASE_URL": url})


def test_credentials_in_base_url_are_rejected_without_echoing_them() -> None:
    with pytest.raises(ConfigurationError, match="credentials") as exc_info:
        load_settings({"OLLAMA_BASE_URL": "http://user:s3cret@localhost:11434"})

    assert "s3cret" not in str(exc_info.value)


@pytest.mark.parametrize("hosts", ["ollama/x", "http://ollama", "a b", "ollama:11434", "*"])
def test_invalid_allowlist_entries_are_rejected(hosts: str) -> None:
    with pytest.raises(ConfigurationError, match="LOCAL_ALLOWED_HOSTS"):
        load_settings({"LOCAL_ALLOWED_HOSTS": hosts})


@pytest.mark.parametrize("variable", ["CODE_MODEL", "EMBEDDING_MODEL"])
@pytest.mark.parametrize(
    "model",
    [
        "qwen3-coder:480b-cloud",
        "gpt-oss:20b-cloud",
        "kimi-k2:cloud",
        "some-model-cloud",
        "QWEN3-CODER:480B-CLOUD",
    ],
)
def test_cloud_models_are_rejected(variable: str, model: str) -> None:
    with pytest.raises(ConfigurationError, match=rf"{variable}.*cloud"):
        load_settings({variable: model})


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("REQUEST_TIMEOUT", "abc"),
        ("REQUEST_TIMEOUT", "0"),
        ("REQUEST_TIMEOUT", "-1"),
        ("REQUEST_TIMEOUT", "nan"),
        ("REQUEST_TIMEOUT", "inf"),
        ("LLM_CONTEXT_WINDOW", "abc"),
        ("LLM_CONTEXT_WINDOW", "0"),
        ("LLM_CONTEXT_WINDOW", "-4096"),
        ("LLM_CONTEXT_WINDOW", "16384.0"),
        ("CODE_MODEL", "qwen 2.5"),
        ("CODE_MODEL", ""),
        ("OLLAMA_BASE_URL", "   "),
    ],
)
def test_invalid_values_are_rejected_naming_the_variable(variable: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=variable):
        load_settings({variable: value})


def test_invalid_environment_fails_closed_at_import_time() -> None:
    env = {**os.environ, "OLLAMA_BASE_URL": "http://example.com:11434"}

    result = subprocess.run(
        [sys.executable, "-c", "import src.config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "ConfigurationError" in result.stderr
    assert "LOCAL_ALLOWED_HOSTS" in result.stderr
