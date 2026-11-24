"""Structured logging: one JSON object per line on stderr, standard library only.

Each line has ``timestamp`` (ISO-8601, UTC), ``level``, ``logger`` and ``event`` (the log
message, written as a short dotted event name), plus the context passed with ``extra=``,
e.g. ``component``, ``duration_ms`` or ``model``.

Policy: log metadata only (lengths, dimensions, durations, counts, model names). Never log
full prompts, indexed code, model output, vectors, or secrets. As a safety net, context
keys that usually carry such payloads are replaced by a short ``[redacted ...]`` marker.

Library modules only call ``logging.getLogger(__name__)``; entry points (CLIs) call
:func:`configure_logging` once.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final, TextIO

JSON_HANDLER_NAME: Final = "local-code-rag-json"

_STANDARD_RECORD_KEYS: Final = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", (), None))
) | {"message", "asctime"}

REDACTED_KEYS: Final = frozenset(
    {
        "prompt",
        "prompts",
        "system",
        "messages",
        "input",
        "inputs",
        "response",
        "completion",
        "content",
        "text",
        "code",
        "source",
        "snippet",
        "chunk",
        "embedding",
        "embeddings",
        "vector",
        "vectors",
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
    }
)

_NOISY_LIBRARY_LOGGERS: Final = ("httpx", "httpcore")


def _redact(value: object) -> str:
    if isinstance(value, str | bytes | list | tuple | dict):
        return f"[redacted {type(value).__name__} len={len(value)}]"
    return f"[redacted {type(value).__name__}]"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line with redacted payload-like context."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key in _STANDARD_RECORD_KEYS or key.startswith("_") or key in payload:
                continue
            payload[key] = _redact(value) if key.lower() in REDACTED_KEYS else value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc_message"] = str(record.exc_info[1])
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: int | str = logging.INFO, stream: TextIO | None = None) -> None:
    """Install the JSON handler on the root logger (idempotent; stderr by default)."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if h.get_name() == JSON_HANDLER_NAME]:
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(JSON_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    for name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
