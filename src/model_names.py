"""Pure helpers for Ollama model names (no I/O, no configuration).

Kept apart from ``src.config`` so that code which must import cleanly even when the
configuration is invalid (e.g. the health check) can still use them.
"""

from __future__ import annotations


def is_cloud_model(model: str) -> bool:
    """Whether ``model`` is an Ollama cloud tag (``name:cloud``, ``name:<x>-cloud``...).

    Cloud models run on Ollama's servers, so prompts would leave the machine.
    """
    ref = model.strip().lower()
    name, _, tag = ref.rpartition(":")
    if not name or "/" in tag:  # no tag, or the colon belongs to a registry host:port
        name, tag = ref, ""
    return tag == "cloud" or tag.endswith("-cloud") or name.endswith("-cloud")


def normalize_model_name(model: str) -> str:
    """Add the implicit ``:latest`` tag, as ``ollama list`` shows untagged models."""
    last_segment = model.rsplit("/", 1)[-1]
    return model if ":" in last_segment else f"{model}:latest"
