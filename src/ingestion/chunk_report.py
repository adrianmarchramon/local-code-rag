"""Immutable, deterministic summary of one chunking run (Phase 2, Step 6).

Counts, reason codes and repository-relative paths only: no timings, no file or chunk
contents, no symbol names or headings. Two runs over the same documents give equal
reports; timings go to the logs.

Size percentiles use the nearest-rank method on node text lengths in characters: the
p-th percentile of n sorted values is the value at 1-based rank ``ceil(p / 100 * n)``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from llama_index.core.schema import MetadataMode, TextNode

from src.ingestion.spans import ParseStatus, Span


@dataclass(frozen=True, slots=True)
class NodeTally:
    documents: int
    nodes: int


@dataclass(frozen=True, slots=True)
class SizeStats:
    """Node text lengths in characters (nearest-rank percentiles; 0 when empty)."""

    p50: int
    p90: int
    max: int


@dataclass(frozen=True, slots=True)
class DocumentChunks:
    """What one document produced; the input the report is built from."""

    file_path: str
    language: str
    strategy: str
    fallback_reason: str | None
    status: ParseStatus
    error_regions: int
    spans: tuple[Span, ...]
    nodes: tuple[TextNode, ...]


@dataclass(frozen=True, slots=True)
class ChunkReport:
    """What a chunking run produced, per language, strategy and span kind."""

    documents: int
    nodes: int
    by_language: Mapping[str, NodeTally]
    by_strategy: Mapping[str, NodeTally]
    by_kind: Mapping[str, int]
    forced: int
    forced_by_kind: Mapping[str, int]
    fallback_reasons: Mapping[str, int]  # reason code -> documents
    syntax_error_files: tuple[str, ...]
    syntax_error_regions: int
    degraded_files: tuple[str, ...]  # strategy failed; segmented as generic text
    empty_documents: tuple[str, ...]  # documents that produced no node
    size: SizeStats
    max_embed_chars: int  # longest node as the embedding model sees it

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form, with deterministic key order."""
        return {
            "documents": self.documents,
            "nodes": self.nodes,
            "by_language": _tallies(self.by_language),
            "by_strategy": _tallies(self.by_strategy),
            "by_kind": dict(self.by_kind),
            "forced": self.forced,
            "forced_by_kind": dict(self.forced_by_kind),
            "fallback_reasons": dict(self.fallback_reasons),
            "syntax_error_files": list(self.syntax_error_files),
            "syntax_error_regions": self.syntax_error_regions,
            "degraded_files": list(self.degraded_files),
            "empty_documents": list(self.empty_documents),
            "size_chars": {"p50": self.size.p50, "p90": self.size.p90, "max": self.size.max},
            "max_embed_chars": self.max_embed_chars,
        }

    def render_text(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [f"Chunked: {self.documents} documents -> {self.nodes} nodes"]
        lines += _tally_lines("By language", self.by_language)
        lines += _tally_lines("By strategy", self.by_strategy)
        lines += _count_lines("By kind", self.by_kind)
        lines += _count_lines("Forced", self.forced_by_kind)
        lines += _count_lines("Fallback reasons (documents)", self.fallback_reasons)
        lines.append(
            f"Syntax errors: {len(self.syntax_error_files)} files, "
            f"{self.syntax_error_regions} regions"
        )
        lines += [f"  {path}" for path in self.syntax_error_files]
        lines.append(f"Degraded to generic text: {len(self.degraded_files)}")
        lines += [f"  {path}" for path in self.degraded_files]
        lines.append(f"Documents without nodes: {len(self.empty_documents)}")
        lines += [f"  {path}" for path in self.empty_documents]
        lines.append(
            f"Node size (chars): p50 {self.size.p50:,}, p90 {self.size.p90:,}, "
            f"max {self.size.max:,}; longest embedding input {self.max_embed_chars:,}"
        )
        return "\n".join(lines) + "\n"


def percentile(values: Sequence[int], percent: int) -> int:
    """Nearest-rank percentile of ``values`` (0 for no values)."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def _tallies(tallies: Mapping[str, NodeTally]) -> dict[str, dict[str, int]]:
    return {key: {"documents": t.documents, "nodes": t.nodes} for key, t in tallies.items()}


def _tally_lines(title: str, tallies: Mapping[str, NodeTally]) -> list[str]:
    return [f"{title}:"] + [
        f"  {key}: {t.documents} documents, {t.nodes} nodes" for key, t in tallies.items()
    ]


def _count_lines(title: str, counts: Mapping[str, int]) -> list[str]:
    return [f"{title}: {sum(counts.values())}"] + [f"  {k}: {v}" for k, v in counts.items()]


def _sorted(counter: Counter[str]) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(counter.items())))


def _group(chunks: Sequence[DocumentChunks], key: str) -> Mapping[str, NodeTally]:
    documents: Counter[str] = Counter()
    nodes: Counter[str] = Counter()
    for item in chunks:
        value = getattr(item, key)
        documents[value] += 1
        nodes[value] += len(item.nodes)
    return MappingProxyType({k: NodeTally(documents[k], nodes[k]) for k in sorted(documents)})


def build_chunk_report(chunks: Sequence[DocumentChunks], degraded: Sequence[str]) -> ChunkReport:
    """Aggregate one chunking run; every mapping is read-only and sorted by key."""
    spans = [span for item in chunks for span in item.spans]
    nodes = [node for item in chunks for node in item.nodes]
    sizes = [len(node.text) for node in nodes]
    reasons = Counter(item.fallback_reason for item in chunks if item.fallback_reason)
    with_errors = [item for item in chunks if item.status is ParseStatus.HAS_ERRORS]
    return ChunkReport(
        documents=len(chunks),
        nodes=len(nodes),
        by_language=_group(chunks, "language"),
        by_strategy=_group(chunks, "strategy"),
        by_kind=_sorted(Counter(span.kind.value for span in spans)),
        forced=sum(1 for span in spans if span.forced),
        forced_by_kind=_sorted(Counter(span.kind.value for span in spans if span.forced)),
        fallback_reasons=_sorted(reasons),
        syntax_error_files=tuple(sorted(item.file_path for item in with_errors)),
        syntax_error_regions=sum(item.error_regions for item in with_errors),
        degraded_files=tuple(sorted(degraded)),
        empty_documents=tuple(sorted(item.file_path for item in chunks if not item.nodes)),
        size=SizeStats(percentile(sizes, 50), percentile(sizes, 90), max(sizes, default=0)),
        max_embed_chars=max(
            (len(node.get_content(metadata_mode=MetadataMode.EMBED)) for node in nodes),
            default=0,
        ),
    )
