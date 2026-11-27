"""Phase 3, Step 6: retrieval evaluation against the owner's golden queries.

Golden queries (``eval/retrieval_golden.json``, git-ignored: Gate 1 Q-G) are written by
the owner before any result is seen::

    {"version": 1, "queries": [{"id", "intent", "lang", "kind", "query", "expected_files"}]}

``lang`` is ``es`` or ``en``; ES/EN variants of one question share an ``intent``. ``kind``
is ``concept`` or ``identifier`` (an exact name is asked for). :func:`parse_golden` is
strict: unknown or missing keys, duplicate IDs, an empty or repeated ``expected_files``, a
file the index does not hold, or a query or intent whose whole text is a template marker
(``REPLACE_WITH_...``, ``<...>``, or starting with TODO, FIXME, TBD or XXX) are errors that
name the query ID and the field, never the query text.

Metrics are computed over the ranking of chunks: a query's rank is the position of the
first chunk whose ``file_path`` is one of its expected files (None if none within the
depth). hit@k is the share of queries ranked within k; MRR@20 is the mean of 1/rank (0
beyond 20). They are broken down by language, by ``kind`` and by code vs docs (docs when
every expected file is Markdown). With 8-15 queries they are indicative: no threshold.

The evaluator talks to a :class:`Retriever` protocol, so Phase 4 can plug its hybrid
retriever and reranker into the same metrics. :class:`VectorRetriever` is Phase 3's.

The baseline (``eval/retrieval_baseline.json``, versioned) records metrics, rank per ID,
the queries' sha256, the index identity, the corpus HEAD and versions: never query text.
It is never overwritten without ``--overwrite-baseline`` (an owner decision).

CLI: ``python -m src.evaluation.retrieval [--golden PATH] [--repo PATH] [--rules-file PATH]
[--json] [--write-baseline PATH [--overwrite-baseline]] [--review-dir DIR]``
(``make eval-retrieval EVAL_FLAGS=...``). ``--review-dir`` writes the full queries and the
retrieved code to a private directory (0700/0600) outside both repositories, for the
owner's review in their own terminal; it is refused inside either repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import stat
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, Protocol, TextIO

from llama_index.core import VectorStoreIndex

from src.errors import (
    ConfigurationError,
    ExternalNetworkAccessError,
    GoldenQueriesError,
    InvalidQueryError,
    LocalStackError,
)
from src.indexing.manifest import IndexManifest
from src.logging_config import configure_logging
from src.network_guard import NetworkGuard

logger = logging.getLogger("src.evaluation.retrieval")

_COMPONENT: Final = "evaluation"
GOLDEN_VERSION: Final = 1
DEPTHS: Final = (1, 3, 5, 10, 20)
MRR_DEPTH: Final = 20
DEPTH: Final = 20  # Phase 4's RETRIEVE_TOP_K
LANGS: Final = frozenset({"es", "en"})
KINDS: Final = frozenset({"concept", "identifier"})
BASELINE_FORMAT: Final = "local-code-rag/retrieval-baseline"
USAGE_ERROR: Final = 2
_QUERY_KEYS: Final = frozenset({"id", "intent", "lang", "kind", "query", "expected_files"})
_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# A whole-text template marker is never a real question: four slipped through Gate 2.
_PLACEHOLDER_RE: Final = re.compile(
    r"\s*(?:REPLACE_WITH_\w*|<[^<>\n]+>|(?:TODO|FIXME|TBD|XXX)\b.*)\s*", re.DOTALL
)
_PROJECT_ROOT: Final = Path(__file__).absolute().parents[2]
DEFAULT_GOLDEN: Final = _PROJECT_ROOT / "eval" / "retrieval_golden.json"
LIBRARIES: Final = ("chromadb", "llama-index-core", "llama-index-vector-stores-chroma", "ollama")


# --- golden queries ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    id: str
    intent: str
    lang: str
    kind: str
    query: str
    expected_files: tuple[str, ...]


def _where(position: int, raw: object) -> str:
    if isinstance(raw, dict) and isinstance(raw.get("id"), str) and _ID_RE.fullmatch(raw["id"]):
        return f"query {raw['id']!r}"
    return f"query #{position + 1}"


def parse_golden(
    data: bytes, *, known_files: Collection[str] | None = None
) -> tuple[GoldenQuery, ...]:
    """Strictly parse the golden file; :class:`GoldenQueriesError` names ID and field only."""
    try:
        document: object = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoldenQueriesError(
            f"the golden file is not UTF-8 JSON ({type(exc).__name__})"
        ) from exc
    if not isinstance(document, dict) or set(document) != {"version", "queries"}:
        raise GoldenQueriesError('the golden file must be an object with keys "version", "queries"')
    if document["version"] != GOLDEN_VERSION:
        raise GoldenQueriesError(f"version must be {GOLDEN_VERSION}")
    raw_queries = document["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise GoldenQueriesError("queries must be a non-empty list")
    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_queries):
        where = _where(position, raw)
        if not isinstance(raw, dict):
            raise GoldenQueriesError(f"{where}: must be an object")
        unknown = sorted(set(raw) - _QUERY_KEYS)
        missing = sorted(_QUERY_KEYS - set(raw))
        if unknown or missing:
            raise GoldenQueriesError(f"{where}: unknown keys {unknown}, missing keys {missing}")
        query_id = raw["id"]
        if not isinstance(query_id, str) or not _ID_RE.fullmatch(query_id):
            raise GoldenQueriesError(f"{where}: id must match {_ID_RE.pattern}")
        if query_id in seen:
            raise GoldenQueriesError(f"{where}: id is repeated")
        seen.add(query_id)
        for key in ("intent", "query"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise GoldenQueriesError(f"{where}: {key} must be a non-empty string")
            if _PLACEHOLDER_RE.fullmatch(raw[key]):
                raise GoldenQueriesError(f"{where}: {key} looks like a template placeholder")
        if raw["lang"] not in LANGS:
            raise GoldenQueriesError(f"{where}: lang must be one of {sorted(LANGS)}")
        if raw["kind"] not in KINDS:
            raise GoldenQueriesError(f"{where}: kind must be one of {sorted(KINDS)}")
        files = raw["expected_files"]
        if (
            not isinstance(files, list)
            or not files
            or not all(isinstance(f, str) and f for f in files)
            or len(set(files)) != len(files)
        ):
            raise GoldenQueriesError(
                f"{where}: expected_files must be a non-empty list of distinct paths"
            )
        if known_files is not None:
            absent = [index for index, f in enumerate(files) if f not in known_files]
            if absent:
                raise GoldenQueriesError(
                    f"{where}: expected_files{absent} are not files in the index"
                )
        queries.append(
            GoldenQuery(
                query_id, raw["intent"], raw["lang"], raw["kind"], raw["query"], tuple(files)
            )
        )
    return tuple(queries)


def golden_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- metrics (pure) ----------------------------------------------------------------------------


def first_relevant_rank(ranked_files: Sequence[str], expected: Collection[str]) -> int | None:
    """1-based position of the first chunk from an expected file, or None."""
    for position, file_path in enumerate(ranked_files, start=1):
        if file_path in expected:
            return position
    return None


def hit_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None], depth: int = MRR_DEPTH) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= depth) / len(ranks) if ranks else 0.0


def summarize(ranks: Sequence[int | None]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"n": len(ranks)}
    for k in DEPTHS:
        summary[f"hit@{k}"] = round(hit_at(ranks, k), 4)
    summary[f"mrr@{MRR_DEPTH}"] = round(mrr(ranks), 4)
    return summary


# --- retrievers -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedChunk:
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None
    node_id: str
    cosine: float | None
    text: str


class Retriever(Protocol):
    """What the evaluator needs: a name and a ranking of chunks for a query."""

    @property
    def name(self) -> str: ...

    def rank(self, query: str, depth: int) -> Sequence[RankedChunk]: ...


class VectorRetriever:
    """Phase 3's similarity search over a loaded, verified index."""

    def __init__(self, index: VectorStoreIndex) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return "vector-cosine"

    def rank(self, query: str, depth: int) -> Sequence[RankedChunk]:
        from src.indexing.search import search

        return [
            RankedChunk(
                h.file_path, h.start_line, h.end_line, h.symbol, h.node_id, h.cosine, h.text
            )
            for h in search(self._index, query, top_k=depth)
        ]


# --- evaluation --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    query: GoldenQuery
    rank: int | None
    chunks: tuple[RankedChunk, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    retriever: str
    depth: int
    outcomes: tuple[QueryOutcome, ...]
    metrics: dict[str, float | int]
    breakdown: dict[str, dict[str, dict[str, float | int]]]

    def ranks(self) -> dict[str, int | None]:
        return {o.query.id: o.rank for o in self.outcomes}


def _group(file_languages: Mapping[str, str], query: GoldenQuery) -> str:
    languages = {file_languages.get(f) for f in query.expected_files}
    return "docs" if languages == {"markdown"} else "code"


def evaluate(
    queries: Sequence[GoldenQuery],
    retriever: Retriever,
    file_languages: Mapping[str, str],
    *,
    depth: int = DEPTH,
) -> EvaluationResult:
    """Rank every query and compute metrics, overall and broken down."""
    outcomes = []
    for query in queries:
        chunks = tuple(retriever.rank(query.query, depth))
        rank = first_relevant_rank([c.file_path for c in chunks], query.expected_files)
        outcomes.append(QueryOutcome(query, rank, chunks))
    breakdown: dict[str, dict[str, dict[str, float | int]]] = {}
    for dimension, key in (
        ("lang", lambda q: q.lang),
        ("kind", lambda q: q.kind),
        ("target", lambda q: _group(file_languages, q)),
    ):
        groups: dict[str, list[int | None]] = {}
        for outcome in outcomes:
            groups.setdefault(key(outcome.query), []).append(outcome.rank)
        breakdown[dimension] = {name: summarize(r) for name, r in sorted(groups.items())}
    result = EvaluationResult(
        retriever.name, depth, tuple(outcomes), summarize([o.rank for o in outcomes]), breakdown
    )
    logger.info(
        "evaluation.done",
        extra={
            "component": _COMPONENT,
            "queries": len(outcomes),
            **{k: v for k, v in result.metrics.items() if k != "n"},
        },
    )
    return result


def _libraries() -> dict[str, str]:
    found = {}
    for package in LIBRARIES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not installed"
    return found


def baseline_record(
    result: EvaluationResult, queries_sha256: str, manifest: IndexManifest
) -> dict[str, object]:
    """The versioned baseline: metrics and ranks by ID, index identity; no query text."""
    return {
        "format": BASELINE_FORMAT,
        "version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "retriever": result.retriever,
        "depth": result.depth,
        "queries": {"sha256": queries_sha256, "count": len(result.outcomes)},
        "index": {
            "generation": manifest.generation,
            "nodes": manifest.nodes,
            "node_digest": manifest.node_digest,
            "embed_digest": manifest.embed_digest,
            "model": manifest.embedder.model,
            "model_digest": manifest.embedder.digest,
            "recipe_fingerprint": manifest.embedder.recipe_fingerprint,
        },
        "corpus_head": manifest.repo.head,
        "versions": _libraries(),
        "metrics": result.metrics,
        "breakdown": result.breakdown,
        "ranks": result.ranks(),
    }


# --- review dump (the owner runs it) -----------------------------------------------------------


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def write_review(
    result: EvaluationResult, directory: Path, forbidden_roots: Sequence[Path]
) -> Path:
    """Write queries and retrieved code to ``directory`` (0700, file 0600), outside repos."""
    target = directory.resolve()
    for root in forbidden_roots:
        if _inside(target, root.resolve()):
            raise InvalidQueryError(
                "--review-dir must be outside this repository and the indexed one; use mktemp -d."
            )
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = target.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise InvalidQueryError("--review-dir is not a directory")
    os.chmod(target, 0o700)
    from src.indexing.search import sanitize

    lines: list[str] = []
    for outcome in result.outcomes:
        q = outcome.query
        lines.append(f"=== {q.id} [{q.lang}, {q.kind}, intent {q.intent}] rank {outcome.rank}")
        lines.append(f"query: {sanitize(q.query)}")
        lines.append(f"expected: {', '.join(q.expected_files)}")
        for position, chunk in enumerate(outcome.chunks[:5], start=1):
            mark = "*" if chunk.file_path in q.expected_files else " "
            cosine = "-" if chunk.cosine is None else f"{chunk.cosine:.4f}"
            lines.append(
                f"--- {mark}{position} {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
                f"{chunk.symbol or '-'} cos {cosine}"
            )
            lines.append(sanitize(chunk.text))
        lines.append("")
    path = target / "review.txt"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview("\n".join(lines).encode("utf-8"))
        while view:
            view = view[os.write(fd, view) :]
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


# --- CLI ---------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.retrieval",
        description="Evaluate retrieval on the golden queries (ranks by ID; no query text).",
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="golden queries file")
    parser.add_argument("--repo", type=Path, help="repository root (default: REPO_PATH)")
    parser.add_argument(
        "--rules-file", type=Path, help="ingestion rules (default: INGEST_IGNORE_FILE)"
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--write-baseline", type=Path, help="write the baseline to this file")
    parser.add_argument("--overwrite-baseline", action="store_true", help="owner decision only")
    parser.add_argument("--review-dir", type=Path, help="owner only: dump queries and code here")
    return parser


def _fail(out: TextIO, code: int, name: str, error: Exception | str) -> int:
    logger.error(
        "evaluation.failed",
        extra={"component": _COMPONENT, "exit_code": code, "error_type": type(error).__name__},
    )
    out.write(f"[FAIL] evaluation: {error}\n")
    out.write(f"RESULT: FAIL (exit {code} {name})\n")
    return code


def render(result: EvaluationResult) -> str:
    m = result.metrics
    lines = [
        f"Retriever {result.retriever}, depth {result.depth}, {m['n']} queries",
        "  " + "  ".join(f"{k} {v}" for k, v in m.items() if k != "n"),
    ]
    for dimension, groups in result.breakdown.items():
        for name, summary in groups.items():
            lines.append(
                f"  {dimension}={name} (n={summary['n']}): hit@1 {summary['hit@1']}  "
                f"hit@5 {summary['hit@5']}  mrr@20 {summary['mrr@20']}"
            )
    lines.append(
        "  ranks: "
        + ", ".join(f"{k}={v if v is not None else '-'}" for k, v in result.ranks().items())
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging()
    out = stdout if stdout is not None else sys.stdout
    from src.indexing.build_index import ExitCode, exit_code_for, load_index_with_report
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository

    try:
        config = importlib.import_module("src.config")
        repo: Path = args.repo if args.repo is not None else config.REPO_PATH
        rules_file: Path = (
            args.rules_file if args.rules_file is not None else config.INGEST_IGNORE_FILE
        )
        allowed: frozenset[str] = config.LOCAL_ALLOWED_HOSTS
    except ConfigurationError as error:
        return _fail(out, ExitCode.CONFIGURATION, ExitCode.CONFIGURATION.name, error)
    if (
        args.write_baseline is not None
        and args.write_baseline.exists()
        and not args.overwrite_baseline
    ):
        return _fail(
            out,
            USAGE_ERROR,
            "USAGE",
            "the baseline exists; overwriting it is an owner decision (--overwrite-baseline)",
        )
    try:
        data = args.golden.read_bytes()
    except OSError as error:
        return _fail(
            out, USAGE_ERROR, "USAGE", f"cannot read the golden file ({type(error).__name__})"
        )
    with NetworkGuard(allowed_hosts=allowed) as guard:
        try:
            nodes = chunk_documents(load_repository(repo, rules_file=rules_file))
            languages = {str(n.metadata["file_path"]): str(n.metadata["language"]) for n in nodes}
            queries = parse_golden(data, known_files=languages.keys())
            loaded = load_index_with_report(nodes, repo_path=repo)
            try:
                result = evaluate(queries, VectorRetriever(loaded.index), languages)
            finally:
                loaded.close()
            if args.review_dir is not None:
                write_review(result, args.review_dir, [_PROJECT_ROOT, repo])
        except (GoldenQueriesError, InvalidQueryError) as error:
            return _fail(out, USAGE_ERROR, "USAGE", error)
        except LocalStackError as error:
            code = exit_code_for(error)
            return _fail(out, int(code), code.name, error)
    if guard.violations:
        first = guard.violations[0]
        blocked = ExternalNetworkAccessError(first.operation, first.destination)
        return _fail(out, ExitCode.EXTERNAL_NETWORK, ExitCode.EXTERNAL_NETWORK.name, blocked)
    record = baseline_record(result, golden_sha256(data), loaded.manifest)
    if args.write_baseline is not None:
        args.write_baseline.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.json:
        out.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    else:
        out.write(render(result))
        out.write(f"queries sha256 {record['queries']['sha256']}\nRESULT: OK\n")  # type: ignore[index]
    return int(ExitCode.OK)


if __name__ == "__main__":
    raise SystemExit(main())
