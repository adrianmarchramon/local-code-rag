"""Tests for src.logging_config: one JSON line per record, metadata only."""

import io
import json
import logging
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest

from src.logging_config import JSON_HANDLER_NAME, configure_logging

logger = logging.getLogger("tests.logging_config")


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    stream = io.StringIO()
    configure_logging(level="DEBUG", stream=stream)
    yield stream
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_each_record_is_one_json_line_with_the_required_fields(log_stream: io.StringIO) -> None:
    logger.info("check.done", extra={"component": "health_check", "duration_ms": 12.5})

    [record] = _records(log_stream)

    assert record["level"] == "INFO"
    assert record["logger"] == "tests.logging_config"
    assert record["event"] == "check.done"
    assert record["component"] == "health_check"
    assert record["duration_ms"] == 12.5
    timestamp = datetime.fromisoformat(str(record["timestamp"]))
    assert timestamp.utcoffset() == timedelta(0)


def test_payload_like_context_is_redacted(log_stream: io.StringIO) -> None:
    logger.info(
        "generation.done",
        extra={"prompt": "def secret_function(): ...", "embedding": [0.25, 0.5], "model": "m"},
    )

    output = log_stream.getvalue()
    [record] = _records(log_stream)

    assert "secret_function" not in output
    assert "0.25" not in output
    assert str(record["prompt"]).startswith("[redacted")
    assert str(record["embedding"]).startswith("[redacted")
    assert record["model"] == "m"


def test_exceptions_are_logged_as_type_and_message_without_traceback(
    log_stream: io.StringIO,
) -> None:
    try:
        raise ValueError("bad value")
    except ValueError:
        logger.exception("step.failed")

    [record] = _records(log_stream)

    assert record["exc_type"] == "ValueError"
    assert record["exc_message"] == "bad value"
    assert "Traceback" not in log_stream.getvalue()


def test_configure_logging_is_idempotent(log_stream: io.StringIO) -> None:
    configure_logging(level="DEBUG", stream=log_stream)

    ours = [h for h in logging.getLogger().handlers if h.get_name() == JSON_HANDLER_NAME]
    assert len(ours) == 1


def test_logs_go_to_stderr_by_default(log_stream: io.StringIO) -> None:
    configure_logging()

    [handler] = [h for h in logging.getLogger().handlers if h.get_name() == JSON_HANDLER_NAME]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr
