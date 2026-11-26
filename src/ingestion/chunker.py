"""Phase 2 chunker: Documents become deterministic LlamaIndex nodes (Steps 3-6).

API (the contract with Phases 3-6):

* :func:`chunk_documents` returns ``list[BaseNode]``: every document's nodes, documents
  in input order and nodes in text order;
* :func:`chunk_documents_with_report` also returns the spans and a :class:`ChunkReport`.

Pipeline: validate the documents (the loader's contract) -> dispatch each one by its
``language`` (:mod:`src.ingestion.strategies`) -> segment it -> build its nodes
(:mod:`src.ingestion.nodes`) -> report. Deterministic, no I/O, no network, no mutable
state: Phase 6 runs it on every start-up. A file whose strategy fails with a
``RecursionError`` or ``ValueError`` is segmented as generic text instead, counted and
logged (Gate 1 Q5); a missing grammar (:class:`~src.errors.ParserUnavailableError`) and
invalid input (:class:`~src.errors.InvalidDocumentError`) are fatal.

The parameters are an immutable, validated dataclass (decision log); the modules that do
the work receive them as arguments and no module in ``src/`` imports this one.

CLI: ``python -m src.ingestion.chunker [--repo PATH] [--rules-file PATH]
[--json | --list | --show FILE_PATH]`` (``make chunk``). It loads the repository with the
Phase 1 loader under :class:`NetworkGuard` and uses the loader's exit codes (2 for a
``--show`` path that is not a loaded document). ``--list`` prints symbols and ``--show``
prints file content: they are meant for the owner's own terminal.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode

from src.errors import (
    ConfigurationError,
    ExternalNetworkAccessError,
    IngestionError,
    InvalidChunkingRulesError,
    InvalidDocumentError,
)
from src.ingestion.chunk_report import ChunkReport, DocumentChunks, build_chunk_report
from src.ingestion.documents import METADATA_KEYS
from src.ingestion.loader import ExitCode, load_repository_with_report
from src.ingestion.nodes import build_nodes
from src.ingestion.spans import Segmentation, Span
from src.ingestion.strategies import (
    SYMBOL_STYLES,
    FallbackReason,
    Strategy,
    fallback_reason,
    segmenter_for,
)
from src.ingestion.text_splitter import split_text
from src.logging_config import configure_logging
from src.network_guard import NetworkGuard

# Fixed name: under `python -m src.ingestion.chunker`, __name__ would be "__main__".
logger = logging.getLogger("src.ingestion.chunker")

# Decision P2: 1500 characters of span text. The embedder would allow more (about 3,900
# characters even at 2 characters per token with the visible metadata), but the LLM's
# prompt evaluation on this host is the binding cost (docs/decisions.md, section 2.6).
DEFAULT_MAX_CHUNK_CHARS: Final = 1500


@dataclass(frozen=True, slots=True)
class ChunkingRules:
    """Every parameter chunking decides with; validated on creation."""

    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS

    def __post_init__(self) -> None:
        if type(self.max_chunk_chars) is not int or self.max_chunk_chars <= 0:
            raise InvalidChunkingRulesError(
                f"max_chunk_chars must be a positive integer (characters of span text), "
                f"got {self.max_chunk_chars!r}. The agreed value is {DEFAULT_MAX_CHUNK_CHARS}."
            )


USAGE_ERROR: Final = 2  # as argparse: --show names a path that is not a loaded document

_KEY_TYPES: Final = {
    "file_path": str,
    "language": str,
    "size_bytes": int,
    "num_lines": int,
    "content_hash": str,
    "directory": str,
}


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Nodes in document then text order, their spans (same order), and the report."""

    nodes: tuple[TextNode, ...]
    spans: tuple[Span, ...]
    report: ChunkReport


def _validate(documents: Sequence[Document]) -> None:
    seen: set[str] = set()
    for document in documents:
        if not isinstance(document, Document) or not isinstance(document.text, str):
            raise InvalidDocumentError(f"Expected a llama_index Document, got {type(document)!r}.")
        metadata = document.metadata
        for key in METADATA_KEYS:
            if key not in metadata or type(metadata[key]) is not _KEY_TYPES[key]:
                raise InvalidDocumentError(
                    f"Document {document.doc_id!r}: metadata key {key!r} is missing or not "
                    f"a {_KEY_TYPES[key].__name__}. Build documents with load_repository."
                )
        if document.doc_id != metadata["file_path"]:
            raise InvalidDocumentError(
                f"Document {document.doc_id!r} has doc_id != file_path "
                f"({metadata['file_path']!r}); node IDs would not be deterministic."
            )
        if document.doc_id in seen:
            raise InvalidDocumentError(f"Document {document.doc_id!r} appears twice.")
        seen.add(document.doc_id)


def _segment(document: Document, cap: int) -> tuple[Strategy, FallbackReason | None, Segmentation]:
    """The document's strategy and segmentation, degraded to generic text on failure."""
    strategy, segmenter = segmenter_for(document.metadata["language"])
    try:
        return strategy, fallback_reason(strategy), segmenter(document.text, max_chars=cap)
    except (RecursionError, ValueError) as error:
        logger.warning(
            "chunking.degraded",
            extra={"strategy": strategy.value, "error_type": type(error).__name__},
        )
        logger.debug("chunking.degraded_file", extra={"path": document.doc_id})
        segmentation = split_text(document.text, max_chars=cap)
        return Strategy.TEXT, FallbackReason.SEGMENTATION_FAILED, segmentation


def chunk_documents_with_report(
    documents: Sequence[Document], *, rules: ChunkingRules | None = None
) -> ChunkResult:
    """Chunk ``documents`` and explain the result; see the module docstring."""
    started = time.perf_counter()
    rules = rules if rules is not None else ChunkingRules()
    _validate(documents)
    chunks: list[DocumentChunks] = []
    degraded: list[str] = []
    for document in documents:
        strategy, reason, segmentation = _segment(document, rules.max_chunk_chars)
        if reason is FallbackReason.SEGMENTATION_FAILED:
            degraded.append(document.doc_id)
        nodes = build_nodes(document, segmentation, SYMBOL_STYLES[strategy])
        chunks.append(
            DocumentChunks(
                file_path=document.doc_id,
                language=document.metadata["language"],
                strategy=strategy.value,
                fallback_reason=reason.value if reason is not None else None,
                status=segmentation.status,
                error_regions=segmentation.error_regions,
                spans=segmentation.spans,
                nodes=nodes,
            )
        )
    report = build_chunk_report(chunks, degraded)
    result = ChunkResult(
        nodes=tuple(node for item in chunks for node in item.nodes),
        spans=tuple(span for item in chunks for span in item.spans),
        report=report,
    )
    logger.info(
        "chunking.done",
        extra={
            "component": "chunking",
            "documents": report.documents,
            "nodes": report.nodes,
            "by_strategy": {k: t.nodes for k, t in report.by_strategy.items()},
            "forced": report.forced,
            "degraded_files": len(report.degraded_files),
            "syntax_error_files": len(report.syntax_error_files),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return result


def chunk_documents(
    documents: Sequence[Document], *, rules: ChunkingRules | None = None
) -> list[BaseNode]:
    """The nodes of ``documents`` (Phase 2 contract; Phases 3-6 call this)."""
    return list(chunk_documents_with_report(documents, rules=rules).nodes)


# --- CLI ------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.chunker",
        description="Load a repository, chunk it with the Phase 2 strategies and report.",
    )
    parser.add_argument("--repo", type=Path, help="repository root (default: REPO_PATH)")
    parser.add_argument(
        "--rules-file", type=Path, help="corpus rules file (default: INGEST_IGNORE_FILE)"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the report as JSON")
    output.add_argument(
        "--list", action="store_true", help="one line per node: path, lines, symbol, kind..."
    )
    output.add_argument(
        "--show", metavar="FILE_PATH", help="print the chunks of one loaded document (content)"
    )
    return parser


def _escape(text: str) -> str:
    """Control and other non-printable characters escaped; newlines and tabs kept."""
    return "".join(
        char if char in "\n\t" or char.isprintable() else char.encode("unicode_escape").decode()
        for char in text
    )


def _fail(out: TextIO, code: int, what: str, error: Exception | str) -> int:
    logger.error("chunking.failed", extra={"component": "chunking", "exit_code": code})
    out.write(f"[FAIL] {what}: {error}\n")
    out.write(f"RESULT: FAIL (exit {code})\n")
    return code


def _describe(node: TextNode, span: Span) -> str:
    metadata = node.metadata
    return (
        f"{metadata['file_path']}\t{metadata['start_line']}-{metadata['end_line']}\t"
        f"{metadata.get('symbol', '-')}\t{span.kind.value}\t{len(node.text)}\t"
        f"{'forced' if span.forced else '-'}"
    )


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging()
    out = stdout if stdout is not None else sys.stdout
    try:
        config = importlib.import_module("src.config")
        repo: Path = args.repo if args.repo is not None else config.REPO_PATH
        rules_file: Path = (
            args.rules_file if args.rules_file is not None else config.INGEST_IGNORE_FILE
        )
    except ConfigurationError as error:
        return _fail(out, ExitCode.CONFIGURATION, "configuration", error)
    with NetworkGuard() as guard:
        try:
            documents = load_repository_with_report(repo, rules_file=rules_file).documents
            result = chunk_documents_with_report(documents)
        except ExternalNetworkAccessError as error:
            return _fail(out, ExitCode.EXTERNAL_NETWORK, "network", error)
        except IngestionError as error:
            return _fail(out, ExitCode.INGESTION, "chunking", error)
    if guard.violations:
        first = guard.violations[0]
        blocked = ExternalNetworkAccessError(first.operation, first.destination)
        return _fail(out, ExitCode.EXTERNAL_NETWORK, "network", blocked)
    if not documents:
        return _fail(out, ExitCode.NO_DOCUMENTS, "chunking", "no file passed the filters")
    pairs = list(zip(result.nodes, result.spans, strict=True))
    if args.json:
        out.write(json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    elif args.list:
        out.write("".join(_describe(node, span) + "\n" for node, span in pairs))
    elif args.show is not None:
        selected = [(n, s) for n, s in pairs if n.metadata["file_path"] == args.show]
        if not any(d.doc_id == args.show for d in documents):
            return _fail(out, USAGE_ERROR, "show", f"{args.show!r} is not a loaded document")
        for position, (node, span) in enumerate(selected, start=1):
            out.write(f"--- #{position} {_describe(node, span)}\n{_escape(node.text)}\n")
    else:
        out.write(result.report.render_text())
        out.write(f"RESULT: OK (exit {ExitCode.OK.value} OK)\n")
    return int(ExitCode.OK)


if __name__ == "__main__":
    raise SystemExit(main())
