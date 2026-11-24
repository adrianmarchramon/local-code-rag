"""Centralized, validated configuration: the contract later phases import from.

Phases 3-5 do ``from src.config import CODE_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL,
REQUEST_TIMEOUT, LLM_CONTEXT_WINDOW``: these module-level constants keep exactly these
names. Each one can be overridden by an environment variable of the same name
(Phase 8 sets ``OLLAMA_BASE_URL=http://ollama:11434`` in Docker Compose).

Values are validated fail-closed at import time, without network access or file reads:

* ``OLLAMA_BASE_URL`` must point to a loopback host (``localhost``, ``127.0.0.0/8``,
  ``::1``), unless its host is listed in ``LOCAL_ALLOWED_HOSTS`` (comma-separated, empty
  by default; meant for the ``ollama`` service in Docker Compose).
* ``CODE_MODEL`` and ``EMBEDDING_MODEL`` cannot be Ollama cloud tags (``:cloud`` or
  ``-cloud``): cloud models process prompts on external servers.

An invalid value raises :class:`src.errors.ConfigurationError` with an actionable message.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from src.errors import ConfigurationError
from src.model_names import is_cloud_model
from src.network_guard import is_loopback_host

# --- Model provisioning: single source of truth (fixed, not overridable by env) ---------
# ollama/Modelfile and the Makefile repeat these literals; unit tests fail if they drift.
# The derived name is separate from CODE_MODEL on purpose: overriding CODE_MODEL must
# never make `make models` overwrite the pulled base model.
CODE_BASE_MODEL: Final = "qwen2.5-coder:7b"  # pulled from the registry: `make pull-models`
CODE_DERIVED_MODEL: Final = "qwen2.5-coder-7b-rag"  # created locally: `make models`
DERIVED_MODEL_NUM_CTX: Final = 16_384  # `PARAMETER num_ctx` in ollama/Modelfile

# --- Defaults of the overridable settings ------------------------------------------------
DEFAULT_OLLAMA_BASE_URL: Final = "http://localhost:11434"
DEFAULT_CODE_MODEL: Final = CODE_DERIVED_MODEL
DEFAULT_EMBEDDING_MODEL: Final = "nomic-embed-text"
DEFAULT_REQUEST_TIMEOUT: Final = 180.0  # seconds; local models are slow on CPU
DEFAULT_LLM_CONTEXT_WINDOW: Final = DERIVED_MODEL_NUM_CTX

_HOST_LABEL: Final = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOSTNAME_RE: Final = re.compile(rf"{_HOST_LABEL}(?:\.{_HOST_LABEL})*")
_USERINFO_RE: Final = re.compile(r"//[^/@]*@")


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings."""

    ollama_base_url: str
    code_model: str
    embedding_model: str
    request_timeout: float
    llm_context_window: int
    local_allowed_hosts: frozenset[str]


def load_settings(env: Mapping[str, str]) -> Settings:
    """Build validated settings from an environment mapping (pure: no I/O)."""
    allowed_hosts = _parse_allowed_hosts(env.get("LOCAL_ALLOWED_HOSTS", ""))
    return Settings(
        ollama_base_url=_parse_base_url(
            _read(env, "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL), allowed_hosts
        ),
        code_model=_parse_model(env, "CODE_MODEL", DEFAULT_CODE_MODEL),
        embedding_model=_parse_model(env, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        request_timeout=_parse_positive_float(env, "REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
        llm_context_window=_parse_positive_int(
            env, "LLM_CONTEXT_WINDOW", DEFAULT_LLM_CONTEXT_WINDOW
        ),
        local_allowed_hosts=allowed_hosts,
    )


def _read(env: Mapping[str, str], name: str, default: object) -> str:
    raw = env.get(name)
    if raw is None:
        return str(default)
    value = raw.strip()
    if not value:
        raise ConfigurationError(
            f"{name} is set but empty. Unset it to use the default ({default!r}) or give it "
            "a value."
        )
    return value


def _redact_userinfo(url: str) -> str:
    return _USERINFO_RE.sub("//***@", url)


def _parse_base_url(raw: str, allowed_hosts: frozenset[str]) -> str:
    name = "OLLAMA_BASE_URL"
    shown = _redact_userinfo(raw)
    expected = (
        f"Expected an http(s) URL with a host and optional port, e.g. {DEFAULT_OLLAMA_BASE_URL!r}."
    )
    try:
        parts = urlsplit(raw)
        has_credentials = parts.username is not None or parts.password is not None
    except ValueError as exc:
        raise ConfigurationError(f"{name}={shown!r} is not a valid URL. {expected}") from exc
    if has_credentials:
        raise ConfigurationError(
            f"{name} must not embed credentials (user:password@host); a local Ollama needs none."
        )
    try:
        _ = parts.port  # reading .port validates it: ValueError if non-numeric or out of range
    except ValueError as exc:
        raise ConfigurationError(f"{name}={shown!r} has an invalid port. {expected}") from exc
    if parts.scheme not in {"http", "https"}:
        raise ConfigurationError(f"{name}={shown!r} must use the http or https scheme.")
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise ConfigurationError(f"{name}={shown!r} must not have a path, query, or fragment.")
    host = parts.hostname
    if not host:
        raise ConfigurationError(f"{name}={shown!r} has no host. {expected}")
    if not (is_loopback_host(host) or host in allowed_hosts):
        raise ConfigurationError(
            f"{name}={shown!r} points to non-loopback host {host!r}; this project only talks "
            f"to a local Ollama. Use {DEFAULT_OLLAMA_BASE_URL!r}, or, for a trusted local "
            "service such as the Docker Compose `ollama` container, add the host to "
            "LOCAL_ALLOWED_HOSTS (e.g. LOCAL_ALLOWED_HOSTS=ollama)."
        )
    return raw.rstrip("/")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _parse_allowed_hosts(raw: str) -> frozenset[str]:
    hosts = frozenset(item.strip().lower() for item in raw.split(",") if item.strip())
    for host in sorted(hosts):
        if not (_HOSTNAME_RE.fullmatch(host) or _is_ip_literal(host)):
            raise ConfigurationError(
                f"LOCAL_ALLOWED_HOSTS contains {host!r}, which is not a bare hostname or IP "
                "address. Use a comma-separated list without scheme, port, or path, e.g. "
                "LOCAL_ALLOWED_HOSTS=ollama."
            )
    return hosts


def _parse_model(env: Mapping[str, str], name: str, default: str) -> str:
    value = _read(env, name, default)
    if any(char.isspace() for char in value):
        raise ConfigurationError(f"{name}={value!r} must not contain whitespace.")
    if is_cloud_model(value):
        raise ConfigurationError(
            f"{name}={value!r} is an Ollama cloud model: it would send prompts to external "
            f"servers. Use a local model instead (default: {default!r})."
        )
    return value


def _parse_positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _read(env, name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name}={raw!r} is not a number (seconds).") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name}={raw!r} must be a finite number greater than 0.")
    return value


def _parse_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _read(env, name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name}={raw!r} is not an integer (tokens).") from exc
    if value <= 0:
        raise ConfigurationError(f"{name}={raw!r} must be an integer greater than 0.")
    return value


# Fail-closed validation at import time (the only import-time side effect in this package).
SETTINGS: Final[Settings] = load_settings(os.environ)

OLLAMA_BASE_URL: Final[str] = SETTINGS.ollama_base_url
CODE_MODEL: Final[str] = SETTINGS.code_model
EMBEDDING_MODEL: Final[str] = SETTINGS.embedding_model
REQUEST_TIMEOUT: Final[float] = SETTINGS.request_timeout
LLM_CONTEXT_WINDOW: Final[int] = SETTINGS.llm_context_window
LOCAL_ALLOWED_HOSTS: Final[frozenset[str]] = SETTINGS.local_allowed_hosts
