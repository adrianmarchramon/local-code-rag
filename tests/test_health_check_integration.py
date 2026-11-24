"""Integration: the real local Ollama passes every health check.

Excluded from `make test`; run with `make test-integration` (needs Ollama running and the
configured models present). It still runs under the suite-wide network guard.
"""

import io
import logging
from collections.abc import Iterator

import pytest

from src.health_check import ExitCode, main

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_verify_local_passes_against_the_real_local_ollama() -> None:
    out = io.StringIO()

    exit_code = main(stdout=out)

    assert exit_code == ExitCode.OK, out.getvalue()
