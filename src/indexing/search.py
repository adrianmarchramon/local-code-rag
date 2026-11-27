"""Phase 3, Step 6: similarity search over a loaded, verified index, and its CLI.

:func:`search` is pure over an already-loaded ``VectorStoreIndex``: it never loads, builds
or writes. Results are immutable :class:`SearchHit` records with the Phase 6 ``Source``
fields (``file_path``, ``symbol``, ``start_line``, ``end_line``, ``score``) plus the rank,
the language and the cosine similarity. LlamaIndex's Chroma score is ``exp(cosine - 1)``
(decision P2), so ``cosine = 1 + ln(score)``; a missing score gives ``None``.

The query is embedded by the index's embedder with the recipe's *query* prefix (the
documents were embedded with the *document* prefix); an empty query is refused (it would
embed the prefix alone), and a query over the model's context raises
``EmbeddingInputTooLongError``: it is never truncated.

CLI: ``python -m src.indexing.search QUERY|- [--top-k 5] [--language L] [--repo PATH]
[--rules-file PATH] [--json] [--show]`` (``make search SEARCH_FLAGS=...``). It chunks the
repository without embedding, loads the index with those nodes (so a stale index is
refused) and prints rank, path, lines, symbol, cosine and score. Code is printed only with
``--show``, with control characters and ANSI sequences escaped. The query text is never
logged; it may contain code or secrets. Exit codes: as ``src.indexing.build_index``, plus
2 for a usage error (empty query, invalid top-k or language).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TextIO

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from src.errors import (
    ConfigurationError,
    ExternalNetworkAccessError,
    InvalidQueryError,
    LocalStackError,
)
from src.logging_config import configure_logging
from src.network_guard import NetworkGuard

logger = logging.getLogger("src.indexing.search")

_COMPONENT: Final = "search"
DEFAULT_TOP_K: Final = 5
USAGE_ERROR: Final = 2


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked chunk: where it is, how similar it is; its text only for ``--show``."""

    rank: int
    file_path: str
    symbol: str | None
    start_line: int
    end_line: int
    language: str
    score: float | None
    cosine: float | None
    node_id: str
    text: str

    def to_dict(self, *, with_text: bool = False) -> dict[str, object]:
        data = asdict(self)
        if not with_text:
            del data["text"]
        return data


def cosine_from_score(score: float | None) -> float | None:
    """``1 + ln(score)`` for LlamaIndex's Chroma score ``exp(cos - 1)``; None if unknown."""
    if score is None or not math.isfinite(score) or score <= 0:
        return None
    return 1.0 + math.log(score)


def _known_languages() -> frozenset[str]:
    from src.ingestion.rules import IngestionRules

    return frozenset(IngestionRules().languages)


def validate_query(query: str, top_k: int, language: str | None) -> None:
    """Raise :class:`InvalidQueryError` (never quoting the query) for unusable input."""
    if not query.strip():
        raise InvalidQueryError("The query is empty; it would embed the prefix alone.")
    if type(top_k) is not int or top_k < 1:
        raise InvalidQueryError(f"top_k must be a positive integer, got {top_k!r}.")
    if language is not None and language not in _known_languages():
        raise InvalidQueryError(
            f"Unknown language {language!r}; use one of {sorted(_known_languages())}."
        )


def search(
    index: VectorStoreIndex, query: str, *, top_k: int = DEFAULT_TOP_K, language: str | None = None
) -> tuple[SearchHit, ...]:
    """The ``top_k`` most similar chunks to ``query`` (optionally one ``language`` only)."""
    validate_query(query, top_k, language)
    filters = None
    if language is not None:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)]
        )
    started = time.perf_counter()
    results = index.as_retriever(similarity_top_k=top_k, filters=filters).retrieve(query)
    hits = []
    for rank, result in enumerate(results, start=1):
        metadata = result.node.metadata
        symbol = metadata.get("symbol")
        hits.append(
            SearchHit(
                rank=rank,
                file_path=str(metadata["file_path"]),
                symbol=str(symbol) if symbol is not None else None,
                start_line=int(metadata["start_line"]),
                end_line=int(metadata["end_line"]),
                language=str(metadata["language"]),
                score=result.score,
                cosine=cosine_from_score(result.score),
                node_id=result.node.node_id,
                text=result.node.get_content(),
            )
        )
    logger.info(
        "search.done",
        extra={
            "component": _COMPONENT,
            "query_length": len(query),
            "top_k": top_k,
            "language": language,
            "hits": len(hits),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return tuple(hits)


# --- presentation ------------------------------------------------------------------------------

_KEEP: Final = {"\n", "\t"}


def sanitize(text: str) -> str:
    """Escape control characters (ANSI sequences included) except newline and tab."""
    out = []
    for char in text:
        code = ord(char)
        if char in _KEEP or not (code < 0x20 or 0x7F <= code < 0xA0):
            out.append(char)
        else:
            out.append(f"\\x{code:02x}")
    return "".join(out)


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render(hits: Sequence[SearchHit], *, show: bool) -> str:
    lines = []
    for hit in hits:
        lines.append(
            f"{hit.rank:>3}  cos {_format(hit.cosine)}  score {_format(hit.score)}  "
            f"{hit.file_path}:{hit.start_line}-{hit.end_line}  {hit.symbol or '-'}"
        )
        if show:
            lines.append(sanitize(hit.text))
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


# --- CLI ---------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.indexing.search",
        description="Search the verified local index (no re-indexing; code only with --show).",
    )
    parser.add_argument("query", help="the question, or - to read it from stdin")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="results (default 5)")
    parser.add_argument("--language", help="only chunks of this language label")
    parser.add_argument("--repo", type=Path, help="repository root (default: REPO_PATH)")
    parser.add_argument(
        "--rules-file", type=Path, help="ingestion rules (default: INGEST_IGNORE_FILE)"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the hits as JSON")
    output.add_argument("--show", action="store_true", help="also print each chunk's code")
    return parser


def _fail(out: TextIO, code: int, name: str, error: Exception | str) -> int:
    logger.error(
        "search.failed",
        extra={"component": _COMPONENT, "exit_code": code, "error_type": type(error).__name__},
    )
    out.write(f"[FAIL] search: {error}\n")
    out.write(f"RESULT: FAIL (exit {code} {name})\n")
    return code


def main(
    argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stdin: TextIO | None = None
) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging()
    out = stdout if stdout is not None else sys.stdout
    query: str = (
        (stdin if stdin is not None else sys.stdin).read() if args.query == "-" else args.query
    )
    try:
        validate_query(query, args.top_k, args.language)
    except InvalidQueryError as error:
        return _fail(out, USAGE_ERROR, "USAGE", error)
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
    with NetworkGuard(allowed_hosts=allowed) as guard:
        try:
            nodes = chunk_documents(load_repository(repo, rules_file=rules_file))
            loaded = load_index_with_report(nodes, repo_path=repo)
            try:
                hits = search(loaded.index, query, top_k=args.top_k, language=args.language)
            finally:
                loaded.close()
        except LocalStackError as error:
            code = exit_code_for(error)
            return _fail(out, int(code), code.name, error)
    if guard.violations:
        first = guard.violations[0]
        blocked = ExternalNetworkAccessError(first.operation, first.destination)
        return _fail(out, ExitCode.EXTERNAL_NETWORK, ExitCode.EXTERNAL_NETWORK.name, blocked)
    if args.json:
        out.write(json.dumps([hit.to_dict() for hit in hits], indent=2, ensure_ascii=False) + "\n")
    else:
        out.write(render(hits, show=args.show))
        out.write(f"RESULT: OK ({len(hits)} hits; generation {loaded.generation})\n")
    return int(ExitCode.OK)


if __name__ == "__main__":
    raise SystemExit(main())
