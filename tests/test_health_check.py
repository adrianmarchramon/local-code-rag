"""Unit tests for src.health_check with an in-memory fake client (no real Ollama)."""

import contextlib
import io
import logging
import math
import socket
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from src import config
from src.errors import (
    ExternalNetworkAccessError,
    LocalStackError,
    OllamaUnavailableError,
)
from src.health_check import (
    CheckContext,
    CheckResult,
    ExitCode,
    HealthReport,
    Status,
    default_checks,
    main,
    run_checks,
)
from src.ollama_client import EmbeddingResult, Generation, LocalModel, ModelInfo

CODE_MODEL = "qwen2.5-coder-7b-rag"
EMBEDDING_MODEL = "nomic-embed-text"
EXTERNAL_IP = "203.0.113.10"  # TEST-NET-3: never routed
GENERATED_CODE = "def reverse_string(s: str) -> str:\n    return s[::-1]\n"


@dataclass
class FakeOllamaClient:
    """Configurable stand-in for OllamaClient."""

    models: list[LocalModel] = field(
        default_factory=lambda: [
            LocalModel(f"{CODE_MODEL}:latest", "a" * 64, 4_683_087_561),
            LocalModel(f"{EMBEDDING_MODEL}:latest", "b" * 64, 274_302_450),
        ]
    )
    text: str = GENERATED_CODE
    vectors: list[list[float]] = field(default_factory=lambda: [[0.6, 0.8, 0.0], [0.0, 0.6, 0.8]])
    embedding_length: int | None = 3
    failures: dict[str, LocalStackError] = field(default_factory=dict)
    before_call: Callable[[str], None] | None = None
    calls: list[str] = field(default_factory=list)

    def _enter(self, method: str) -> None:
        self.calls.append(method)
        if self.before_call is not None:
            self.before_call(method)
        if method in self.failures:
            raise self.failures[method]

    def list_models(self) -> list[LocalModel]:
        self._enter("list_models")
        return list(self.models)

    def show(self, model: str) -> ModelInfo:
        self._enter("show")
        return ModelInfo(
            architecture="fake",
            context_length=2048,
            embedding_length=self.embedding_length,
            num_ctx=None,
            parent_model=None,
            capabilities=("embedding",),
        )

    def generate(
        self,
        model: str,
        prompt: str,
        options: Mapping[str, Any],
        keep_alive: float | str | None = None,
    ) -> Generation:
        self._enter("generate")
        return Generation(
            text=self.text,
            total_duration_ns=3_000_000_000,
            load_duration_ns=1_000_000_000,
            prompt_eval_count=30,
            eval_count=20,
            eval_duration_ns=2_000_000_000,
        )

    def embed(self, model: str, inputs: Sequence[str]) -> EmbeddingResult:
        self._enter("embed")
        return EmbeddingResult(
            vectors=[list(v) for v in self.vectors], total_duration_ns=5, load_duration_ns=1
        )


def _context(client: FakeOllamaClient) -> CheckContext:
    return CheckContext(
        client=client,
        base_url="http://localhost:11434",
        code_model=CODE_MODEL,
        embedding_model=EMBEDDING_MODEL,
        fix_commands={CODE_MODEL: "make models", EMBEDDING_MODEL: "make pull-models"},
    )


def _run(client: FakeOllamaClient) -> HealthReport:
    return run_checks(default_checks(), _context(client))


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_exit_code_table_is_stable() -> None:
    assert {code.name: code.value for code in ExitCode} == {
        "OK": 0,
        "UNEXPECTED": 1,
        "CONFIGURATION": 10,
        "OLLAMA_UNAVAILABLE": 11,
        "MODEL_NOT_FOUND": 12,
        "VERIFICATION_FAILED": 13,
        "EXTERNAL_NETWORK": 14,
    }


def test_happy_path_passes_every_check() -> None:
    report = _run(FakeOllamaClient())

    assert report.exit_code == ExitCode.OK
    assert [(r.name, r.status) for r in report.results] == [
        ("service_reachable", Status.PASS),
        ("models_present", Status.PASS),
        ("generation", Status.PASS),
        ("embedding", Status.PASS),
        ("network_guard", Status.PASS),
    ]


def test_generation_metrics_come_from_ollama_counters() -> None:
    generation = _run(FakeOllamaClient()).results[2]

    assert generation.metrics["generated_tokens"] == 20
    assert generation.metrics["tokens_per_s"] == 10.0
    assert generation.metrics["load_ms"] == 1000.0
    assert generation.preview is not None
    assert "s[::-1]" in generation.preview


def test_embedding_reports_the_observed_dimension() -> None:
    embedding = _run(FakeOllamaClient()).results[3]

    assert embedding.metrics["dimension"] == 3
    assert embedding.metrics["reported_dimension"] == 3


def test_service_down_exits_11_and_stops() -> None:
    client = FakeOllamaClient(failures={"list_models": OllamaUnavailableError("down")})

    report = _run(client)

    assert report.exit_code == ExitCode.OLLAMA_UNAVAILABLE
    assert report.failed_check == "service_reachable"
    assert client.calls == ["list_models"]


def test_timeout_during_generation_exits_11() -> None:
    client = FakeOllamaClient(failures={"generate": OllamaUnavailableError("timed out")})

    assert _run(client).exit_code == ExitCode.OLLAMA_UNAVAILABLE


@pytest.mark.parametrize(
    ("listed", "missing", "fix"),
    [
        ([f"{EMBEDDING_MODEL}:latest"], CODE_MODEL, "make models"),
        ([f"{CODE_MODEL}:latest"], EMBEDDING_MODEL, "make pull-models"),
    ],
)
def test_missing_model_exits_12_with_the_fix_command(
    listed: list[str], missing: str, fix: str
) -> None:
    client = FakeOllamaClient(models=[LocalModel(name, "c" * 64, 1) for name in listed])

    report = _run(client)

    assert report.exit_code == ExitCode.MODEL_NOT_FOUND
    assert missing in str(report.error)
    assert fix in str(report.error)


def test_listed_model_without_local_weights_is_rejected() -> None:
    client = FakeOllamaClient(
        models=[
            LocalModel(f"{CODE_MODEL}:latest", "", 0),  # how a remote/cloud model looks
            LocalModel(f"{EMBEDDING_MODEL}:latest", "b" * 64, 274_302_450),
        ]
    )

    report = _run(client)

    assert report.exit_code == ExitCode.VERIFICATION_FAILED
    assert "no local weights" in str(report.error)


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_generation_exits_13(text: str) -> None:
    assert _run(FakeOllamaClient(text=text)).exit_code == ExitCode.VERIFICATION_FAILED


def test_generation_without_def_is_only_a_warning() -> None:
    report = _run(FakeOllamaClient(text="Sure! Here is how you reverse a string."))

    assert report.exit_code == ExitCode.OK
    assert report.results[2].status == Status.WARN


@pytest.mark.parametrize(
    "vectors",
    [
        [[0.6, math.nan, 0.0], [0.0, 0.6, 0.8]],
        [[0.6, math.inf, 0.0], [0.0, 0.6, 0.8]],
        [[0.6, 0.8, 0.0], [0.6, 0.8]],
        [[0.0, 0.0, 0.0], [0.0, 0.6, 0.8]],
        [[0.6, 0.8, 0.0]],
        [[], []],
    ],
    ids=["nan", "inf", "inconsistent-dimension", "zero-norm", "wrong-count", "empty"],
)
def test_invalid_vectors_exit_13(vectors: list[list[float]]) -> None:
    report = _run(FakeOllamaClient(vectors=vectors))

    assert report.exit_code == ExitCode.VERIFICATION_FAILED
    assert report.failed_check == "embedding"


def test_dimension_mismatch_with_ollama_show_is_a_warning() -> None:
    report = _run(FakeOllamaClient(embedding_length=768))

    assert report.exit_code == ExitCode.OK
    assert report.results[3].status == Status.WARN
    assert "768" in report.results[3].detail


def _connect_externally(method: str) -> None:
    if method == "generate":
        with socket.socket() as sock:
            sock.connect((EXTERNAL_IP, 80))


def _connect_externally_and_swallow(method: str) -> None:
    with contextlib.suppress(ExternalNetworkAccessError):
        _connect_externally(method)


@pytest.mark.parametrize(
    "hook", [_connect_externally, _connect_externally_and_swallow], ids=["raised", "swallowed"]
)
def test_external_network_attempt_exits_14(hook: Callable[[str], None]) -> None:
    report = _run(FakeOllamaClient(before_call=hook))

    assert report.exit_code == ExitCode.EXTERNAL_NETWORK
    assert report.failed_check == "generation"
    assert [v.destination for v in report.violations] == [f"{EXTERNAL_IP}:80"]


def test_new_checks_run_without_changing_the_orchestrator() -> None:
    class AlwaysPasses:
        name = "custom"

        def run(self, context: CheckContext) -> CheckResult:
            return CheckResult(self.name, Status.PASS, "ok")

    report = run_checks((*default_checks(), AlwaysPasses()), _context(FakeOllamaClient()))

    assert [r.name for r in report.results][-2:] == ["custom", "network_guard"]


def test_main_prints_the_report_and_logs_metadata_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeOllamaClient(
        models=[
            LocalModel(config.CODE_MODEL, "a" * 64, 1),
            LocalModel(config.EMBEDDING_MODEL, "b" * 64, 1),
        ]
    )
    out = io.StringIO()

    exit_code = main(stdout=out, client_factory=lambda settings: fake)

    assert exit_code == ExitCode.OK
    assert "RESULT: OK" in out.getvalue()
    assert "s[::-1]" in out.getvalue()  # the fixed test prompt's answer may be previewed
    logs = capsys.readouterr().err
    assert '"event": "health_check.finished"' in logs
    assert "s[::-1]" not in logs  # ...but never logged


def test_main_maps_invalid_configuration_to_exit_10_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://{EXTERNAL_IP}:11434")
    monkeypatch.delitem(sys.modules, "src.config")
    out = io.StringIO()

    def must_not_be_called(settings: object) -> FakeOllamaClient:
        raise AssertionError("no client may be created with an invalid configuration")

    exit_code = main(stdout=out, client_factory=must_not_be_called)

    assert exit_code == ExitCode.CONFIGURATION
    assert "LOCAL_ALLOWED_HOSTS" in out.getvalue()
    assert "Traceback" not in capsys.readouterr().err
