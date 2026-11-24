"""Tests for src.model_names: cloud-tag detection and name normalization (pure, no I/O)."""

import pytest

from src.model_names import is_cloud_model, normalize_model_name


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
def test_cloud_tags_are_detected(model: str) -> None:
    assert is_cloud_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "qwen2.5-coder:7b",
        "qwen2.5-coder-7b-rag",
        "nomic-embed-text",
        "cloudy:7b",
        "hf.co/org/repo:Q4_K_M",
        "localhost:5000/team/model",
    ],
)
def test_local_model_names_are_not_mistaken_for_cloud(model: str) -> None:
    assert not is_cloud_model(model)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nomic-embed-text", "nomic-embed-text:latest"),
        ("nomic-embed-text:latest", "nomic-embed-text:latest"),
        ("qwen2.5-coder:7b", "qwen2.5-coder:7b"),
        ("qwen2.5-coder-7b-rag", "qwen2.5-coder-7b-rag:latest"),
        ("hf.co/org/repo", "hf.co/org/repo:latest"),
        ("hf.co/org/repo:Q4_K_M", "hf.co/org/repo:Q4_K_M"),
        ("localhost:5000/team/model", "localhost:5000/team/model:latest"),
    ],
)
def test_normalize_adds_the_implicit_latest_tag(name: str, expected: str) -> None:
    assert normalize_model_name(name) == expected
