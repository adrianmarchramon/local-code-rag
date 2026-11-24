"""Local inference health check: ``make verify-local`` (``python -m src.health_check``).

Runs these checks in order, stopping at the first failure, under a network guard:

1. ``service_reachable``: the Ollama server answers at ``OLLAMA_BASE_URL``.
2. ``models_present``: ``CODE_MODEL`` and ``EMBEDDING_MODEL`` are in the local store with
   weights on disk (remote/cloud models have none).
3. ``generation``: the code model answers a fixed prompt deterministically (temperature
   0, fixed seed, bounded ``num_predict``); latency, cold-load time and tokens/s come from
   Ollama's own counters.
4. ``embedding``: two distinct snippets yield vectors of one non-zero dimension, finite
   values and non-zero norm; the dimension is compared with ``ollama show``.
5. ``network_guard``: nothing tried to reach a non-loopback destination.

The report goes to stdout and may preview the model's answer to the fixed test prompt.
JSON logs go to stderr and carry metadata only.

Exit codes:
    0   all checks passed (warnings allowed)
    1   unexpected error (a bug; Python prints the traceback)
    10  configuration error (fix the environment variable named in the message)
    11  Ollama unavailable (not running, wrong address, or timed out)
    12  model missing (``make pull-models`` / ``make models``)
    13  verification failed (empty answer, invalid vectors, model not local...)
    14  external network access attempted (blocked by the network guard)
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Final, Protocol, TextIO

from src.errors import (
    ConfigurationError,
    ExternalNetworkAccessError,
    LocalStackError,
    ModelNotFoundError,
    OllamaUnavailableError,
    VerificationError,
)
from src.logging_config import configure_logging
from src.model_names import is_cloud_model, normalize_model_name
from src.network_guard import NetworkGuard, NetworkViolation
from src.ollama_client import OllamaClient, OllamaHttpClient

if TYPE_CHECKING:
    from src.config import Settings

# Fixed name: under `python -m src.health_check`, __name__ would be "__main__".
logger = logging.getLogger("src.health_check")

_COMPONENT: Final = "health_check"
_PREVIEW_CHARS: Final = 160


class ExitCode(IntEnum):
    """Process exit codes of the health check (documented in the module docstring)."""

    OK = 0
    UNEXPECTED = 1
    CONFIGURATION = 10
    OLLAMA_UNAVAILABLE = 11
    MODEL_NOT_FOUND = 12
    VERIFICATION_FAILED = 13
    EXTERNAL_NETWORK = 14


_EXIT_CODE_BY_ERROR: Final[tuple[tuple[type[LocalStackError], ExitCode], ...]] = (
    (ExternalNetworkAccessError, ExitCode.EXTERNAL_NETWORK),
    (ConfigurationError, ExitCode.CONFIGURATION),
    (OllamaUnavailableError, ExitCode.OLLAMA_UNAVAILABLE),
    (ModelNotFoundError, ExitCode.MODEL_NOT_FOUND),
    (VerificationError, ExitCode.VERIFICATION_FAILED),
)


def exit_code_for(error: LocalStackError) -> ExitCode:
    """Map a typed error to its exit code."""
    for error_type, code in _EXIT_CODE_BY_ERROR:
        if isinstance(error, error_type):
            return code
    return ExitCode.UNEXPECTED


class Status(StrEnum):
    """Outcome of one check."""

    PASS = "PASS"  # noqa: S105 - a check status, not a password
    WARN = "WARN"
    FAIL = "FAIL"


Metric = int | float | str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Typed result of one check. ``preview`` is printed in the report, never logged."""

    name: str
    status: Status
    detail: str
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    preview: str | None = None


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything a check needs; built once by :func:`main` (or by tests)."""

    client: OllamaClient
    base_url: str
    code_model: str
    embedding_model: str
    fix_commands: Mapping[str, str] = field(default_factory=dict)


class Check(Protocol):
    """A single-responsibility check: returns a result or raises a LocalStackError."""

    @property
    def name(self) -> str: ...

    def run(self, context: CheckContext) -> CheckResult: ...


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Outcome of a run: passed results, the first error (if any), blocked attempts."""

    results: tuple[CheckResult, ...]
    error: LocalStackError | None = None
    failed_check: str | None = None
    violations: tuple[NetworkViolation, ...] = ()

    @property
    def exit_code(self) -> ExitCode:
        if self.violations:
            return ExitCode.EXTERNAL_NETWORK
        return ExitCode.OK if self.error is None else exit_code_for(self.error)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _ns_to_ms(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else round(nanoseconds / 1e6, 1)


def _rate(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / 1e9), 2)


def _metrics(**values: Metric | None) -> dict[str, Metric]:
    return {key: value for key, value in values.items() if value is not None}


def _preview(text: str) -> str:
    flat = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    return flat if len(flat) <= _PREVIEW_CHARS else f"{flat[:_PREVIEW_CHARS]}..."


class ServiceReachableCheck:
    """The Ollama server answers at the configured (loopback) address."""

    name = "service_reachable"

    def run(self, context: CheckContext) -> CheckResult:
        started = time.perf_counter()
        models = context.client.list_models()
        return CheckResult(
            self.name,
            Status.PASS,
            f"Ollama answered at {context.base_url} ({len(models)} local models)",
            _metrics(latency_ms=_elapsed_ms(started), local_models=len(models)),
        )


class ModelsPresentCheck:
    """CODE_MODEL and EMBEDDING_MODEL are present locally, with weights on disk."""

    name = "models_present"

    def run(self, context: CheckContext) -> CheckResult:
        listed = {normalize_model_name(m.name): m for m in context.client.list_models()}
        fixes = {normalize_model_name(k): v for k, v in context.fix_commands.items()}
        metrics: dict[str, Metric] = {}
        found_names: list[str] = []
        for role, model in (("code", context.code_model), ("embedding", context.embedding_model)):
            key = normalize_model_name(model)
            found = listed.get(key)
            if found is None:
                raise ModelNotFoundError(model, fixes.get(key))
            if found.size_bytes <= 0 or is_cloud_model(found.name):
                raise VerificationError(
                    f"{model!r} is listed by Ollama but has no local weights "
                    f"(size {found.size_bytes} B): it is a remote/cloud model and would send "
                    "prompts to external servers. Configure a local model instead."
                )
            metrics[f"{role}_model_bytes"] = found.size_bytes
            found_names.append(f"{model} ({found.size_bytes / 1e9:.2f} GB, {found.digest[:12]})")
        return CheckResult(self.name, Status.PASS, "local: " + ", ".join(found_names), metrics)


class GenerationCheck:
    """The code model answers a fixed prompt deterministically."""

    name = "generation"
    prompt: Final = "Write a Python function that reverses a string."

    def __init__(self, *, seed: int = 42, num_predict: int = 128) -> None:
        self._options: dict[str, int] = {"temperature": 0, "seed": seed, "num_predict": num_predict}

    def run(self, context: CheckContext) -> CheckResult:
        model = context.code_model
        started = time.perf_counter()
        generation = context.client.generate(model, self.prompt, self._options)
        wall_ms = _elapsed_ms(started)
        text = generation.text.strip()
        if not text:
            raise VerificationError(
                f"{model!r} returned an empty answer to the test prompt. Inspect it with "
                f"`ollama run {model}` and check the server log (`journalctl -u ollama`)."
            )
        metrics = _metrics(
            wall_ms=wall_ms,
            total_ms=_ns_to_ms(generation.total_duration_ns),
            load_ms=_ns_to_ms(generation.load_duration_ns),
            prompt_tokens=generation.prompt_eval_count,
            generated_tokens=generation.eval_count,
            tokens_per_s=_rate(generation.eval_count, generation.eval_duration_ns),
        )
        if "def" in text:
            detail = f"{model} answered ({len(text)} chars)"
            return CheckResult(self.name, Status.PASS, detail, metrics, _preview(text))
        detail = f"{model} answered, but with no `def`: not the expected Python function"
        return CheckResult(self.name, Status.WARN, detail, metrics, _preview(text))


def _validate_vectors(model: str, vectors: Sequence[Sequence[float]], expected: int) -> int:
    if len(vectors) != expected:
        raise VerificationError(f"{model!r} returned {len(vectors)} vectors for {expected} inputs.")
    dimensions = sorted({len(vector) for vector in vectors})
    if len(dimensions) != 1:
        raise VerificationError(f"{model!r} returned vectors of different sizes {dimensions}.")
    dimension = dimensions[0]
    if dimension == 0:
        raise VerificationError(f"{model!r} returned empty vectors.")
    for index, vector in enumerate(vectors):
        if not all(math.isfinite(value) for value in vector):
            raise VerificationError(f"{model!r} returned NaN/inf values (vector {index}).")
        if math.fsum(value * value for value in vector) == 0:
            raise VerificationError(f"{model!r} returned a zero vector (input {index}).")
    return dimension


class EmbeddingCheck:
    """The embedding model returns sane, consistent vectors for distinct inputs."""

    name = "embedding"
    inputs: Final = ("def greet(): return 'hello'", "def add(a, b): return a + b")

    def run(self, context: CheckContext) -> CheckResult:
        model = context.embedding_model
        started = time.perf_counter()
        result = context.client.embed(model, self.inputs)
        wall_ms = _elapsed_ms(started)
        dimension = _validate_vectors(model, result.vectors, expected=len(self.inputs))
        reported = context.client.show(model).embedding_length
        warnings: list[str] = []
        if reported is not None and reported != dimension:
            warnings.append(f"`ollama show` reports dimension {reported}, observed {dimension}")
        if result.vectors[0] == result.vectors[1]:
            warnings.append("two different inputs produced identical vectors")
        detail = "; ".join([f"{len(result.vectors)} vectors, dimension {dimension}", *warnings])
        return CheckResult(
            self.name,
            Status.WARN if warnings else Status.PASS,
            detail,
            _metrics(
                dimension=dimension,
                reported_dimension=reported,
                wall_ms=wall_ms,
                load_ms=_ns_to_ms(result.load_duration_ns),
            ),
        )


def default_checks() -> tuple[Check, ...]:
    """The checks run by ``verify-local``, in order."""
    return (ServiceReachableCheck(), ModelsPresentCheck(), GenerationCheck(), EmbeddingCheck())


def _log_check(name: str, status: str, started: float, metrics: Mapping[str, Metric]) -> None:
    logger.info(
        "health_check.check",
        extra={
            "component": _COMPONENT,
            "check": name,
            "status": status,
            "duration_ms": _elapsed_ms(started),
            "metrics": dict(metrics),
        },
    )


def run_checks(
    checks: Iterable[Check], context: CheckContext, *, allowed_hosts: Iterable[str] = ()
) -> HealthReport:
    """Run ``checks`` in order under a network guard; stop at the first failure."""
    results: list[CheckResult] = []
    with NetworkGuard(allowed_hosts=allowed_hosts) as guard:
        for check in checks:
            started = time.perf_counter()
            try:
                result = check.run(context)
            except LocalStackError as error:
                _log_check(check.name, Status.FAIL, started, {"error_type": type(error).__name__})
                return HealthReport(tuple(results), error, check.name, guard.violations)
            if guard.violations:  # blocked, but swallowed by the check or a library
                first = guard.violations[0]
                blocked = ExternalNetworkAccessError(first.operation, first.destination)
                _log_check(check.name, Status.FAIL, started, {"error_type": type(blocked).__name__})
                return HealthReport(tuple(results), blocked, check.name, guard.violations)
            _log_check(check.name, result.status, started, result.metrics)
            results.append(result)
    results.append(
        CheckResult(
            "network_guard",
            Status.PASS,
            "no connection or lookup to a non-loopback destination was attempted",
            {"blocked_attempts": 0},
        )
    )
    return HealthReport(tuple(results))


def render_report(report: HealthReport, context: CheckContext) -> str:
    """Human-readable report for stdout."""
    lines = [
        f"verify-local: Ollama at {context.base_url} | code model {context.code_model} | "
        f"embedding model {context.embedding_model}",
    ]
    for result in report.results:
        lines.append(f"[{result.status}] {result.name}: {result.detail}")
        if result.metrics:
            lines.append("       " + " ".join(f"{k}={v}" for k, v in result.metrics.items()))
        if result.preview:
            lines.append(f"       preview: {result.preview}")
    if report.error is not None:
        lines.append(f"[{Status.FAIL}] {report.failed_check}: {report.error}")
    for violation in report.violations:
        lines.append(
            f"       blocked {violation.operation} to {violation.destination} "
            f"at {violation.timestamp.isoformat()}"
        )
    code = report.exit_code
    verdict = "OK" if code == ExitCode.OK else "FAIL"
    lines.append(f"RESULT: {verdict} (exit {code.value} {code.name})")
    return "\n".join(lines) + "\n"


def _default_client(settings: Settings) -> OllamaClient:
    return OllamaHttpClient(settings.ollama_base_url, settings.request_timeout)


def main(
    *,
    stdout: TextIO | None = None,
    client_factory: Callable[[Settings], OllamaClient] | None = None,
) -> int:
    """CLI entry point; returns the process exit code."""
    configure_logging()
    out = stdout if stdout is not None else sys.stdout
    try:
        # Imported here, not at module level, so an invalid configuration becomes exit
        # code 10 with an actionable message instead of an import-time traceback.
        config = importlib.import_module("src.config")
    except ConfigurationError as error:
        logger.error(
            "health_check.configuration_invalid",
            extra={"component": _COMPONENT, "error": str(error)},
        )
        code = ExitCode.CONFIGURATION
        out.write(f"[{Status.FAIL}] configuration: {error}\n")
        out.write(f"RESULT: FAIL (exit {code.value} {code.name})\n")
        return int(code)

    settings: Settings = config.SETTINGS
    context = CheckContext(
        client=(client_factory or _default_client)(settings),
        base_url=settings.ollama_base_url,
        code_model=settings.code_model,
        embedding_model=settings.embedding_model,
        fix_commands={
            config.CODE_DERIVED_MODEL: "make models",
            config.CODE_BASE_MODEL: "make pull-models",
            config.DEFAULT_EMBEDDING_MODEL: "make pull-models",
        },
    )
    report = run_checks(default_checks(), context, allowed_hosts=settings.local_allowed_hosts)
    out.write(render_report(report, context))
    logger.info(
        "health_check.finished",
        extra={
            "component": _COMPONENT,
            "exit_code": int(report.exit_code),
            "failed_check": report.failed_check,
            "blocked_attempts": len(report.violations),
        },
    )
    return int(report.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
