"""Content-free validation of a chunking run (Phase 2, Step 7; decision P6).

Everything here reports **records** (code, path, offsets, lines, length, kind), never file
text, symbols or headings, so a failing assertion on the frozen corpus cannot copy corpus
content into a terminal or a log. Checkers whose messages carry names (I5 and I7 in
``chunking_invariants``) are re-implemented here without them.

Independent of the code under test where it matters: node IDs are recomputed from the
literal namespace, line numbers from ``text.split("\\n")``, definitions from ``ast``.

Not a test module (pytest does not collect it), and not part of ``src/``.
"""

from __future__ import annotations

import os
import stat
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chunking_invariants import (
    check_bounds,
    check_cap,
    check_coverage,
    check_forced_where_needed,
    check_line_boundaries,
    check_order,
    check_units_whole,
    config_top_regions,
    definition_regions,
    fence_regions,
    markdown_section_regions,
    statement_regions,
)
from llama_index.core import Document
from llama_index.core.schema import NodeRelationship, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from src.ingestion.chunker import ChunkResult
from src.ingestion.spans import Span

# Literal, not imported from src/: a changed namespace must fail here (decision P4).
NAMESPACE: Final = uuid.UUID("41a05dbb-07dd-4b83-840e-8c25648f281a")
KEY_ORDER: Final = (
    "file_path",
    "language",
    "size_bytes",
    "num_lines",
    "content_hash",
    "directory",
    "symbol",
    "start_line",
    "end_line",
)
VISIBLE: Final = ("file_path", "language", "symbol")
_INT_KEYS: Final = frozenset({"size_bytes", "num_lines", "start_line", "end_line"})
_CONFIG: Final = frozenset({"yaml", "toml", "dockerfile", "makefile"})


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    """One broken expectation, located without content."""

    code: str
    path: str
    start: int = -1
    end: int = -1
    start_line: int = 0
    end_line: int = 0
    length: int = 0
    kind: str = ""


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _at(code: str, path: str, text: str, start: int, end: int, kind: str = "") -> Finding:
    return Finding(
        code,
        path,
        start,
        end,
        _line(text, start),
        _line(text, max(start, end - 1)),
        end - start,
        kind,
    )


def _invariant_findings(document: Document, spans: Sequence[Span], cap: int) -> list[Finding]:
    """I1-I4, I6 for every strategy; F1/F2 for fallbacks; name-free I5/I7 for Python."""
    text, path = document.text, document.doc_id
    language = document.metadata["language"]
    violations = (
        check_bounds(text, spans)
        + check_order(spans)
        + check_coverage(text, spans)
        + check_line_boundaries(text, spans, cap)
        + check_cap(text, spans, cap)
    )
    if language == "markdown":
        fences = fence_regions(text)
        violations += check_units_whole(text, spans, fences, cap)
        violations += check_forced_where_needed(
            text, spans, fences + markdown_section_regions(text), cap
        )
    elif language in _CONFIG:
        blocks = config_top_regions(text, language)
        violations += check_units_whole(text, spans, blocks, cap)
        violations += check_forced_where_needed(text, spans, blocks, cap)
    # Only the code and the path: the details of these checkers carry offsets only, but the
    # record stays minimal on purpose.
    findings = [Finding(v.code, path) for v in violations]
    if language == "python":
        findings += _python_findings(text, path, spans, cap)
    return findings


def _python_findings(text: str, path: str, spans: Sequence[Span], cap: int) -> list[Finding]:
    regions = definition_regions(text)
    statements = statement_regions(text)
    if regions is None or statements is None:
        return []  # not valid Python: the ast oracle does not apply
    findings = []
    for region in regions:
        touching = [s for s in spans if s.start < region.end and region.start < s.end]
        if region.size <= cap and not any(
            s.start <= region.start and region.end <= s.end for s in spans
        ):
            findings.append(_at("I5", path, text, region.start, region.end))
        if region.is_function and len(touching) > 1 and not all(s.forced for s in touching):
            findings.append(_at("I7", path, text, region.start, region.end))
    lines = []
    position = 0
    for row in text.split("\n"):
        lines.append((position, position + len(row.rstrip())))
        position += len(row) + 1
    too_big = [(a, b) for a, b in statements + lines if b - a > cap]
    for span in spans:
        if span.forced and not any(a < span.end and span.start < b for a, b in too_big):
            findings.append(_at("I7", path, text, span.start, span.end, span.kind.value))
    return findings


def _node_findings(document: Document, node: TextNode, span: Span) -> list[Finding]:
    text, path = document.text, document.doc_id
    at = _at("", path, text, span.start, span.end, span.kind.value)
    findings: list[Finding] = []

    def add(code: str) -> None:
        findings.append(
            Finding(code, path, at.start, at.end, at.start_line, at.end_line, at.length, at.kind)
        )

    if (node.start_char_idx, node.end_char_idx) != (span.start, span.end):
        add("offsets")
    if node.text != text[span.start : span.end]:
        add("slice")
    metadata = node.metadata
    expected_keys = [k for k in KEY_ORDER if k != "symbol" or "symbol" in metadata]
    if list(metadata) != expected_keys:
        add("keys")
    for key, value in metadata.items():
        wanted = int if key in _INT_KEYS else str
        empty_allowed = key == "directory"  # "" at the repository root (Phase 1 contract)
        if type(value) is not wanted or (value == "" and not empty_allowed):
            add(f"type:{key}")
    inherited = {k: document.metadata[k] for k in KEY_ORDER[:6]}
    if {k: metadata.get(k) for k in inherited} != inherited:
        add("inherited")
    expected_visible = [k for k in VISIBLE if k in metadata]
    for excluded in (node.excluded_embed_metadata_keys, node.excluded_llm_metadata_keys):
        if [k for k in metadata if k not in excluded] != expected_visible:
            add("visible")
    if set(node.relationships) != {NodeRelationship.SOURCE} or node.ref_doc_id != path:
        add("relationship")
    expected_id = str(uuid.uuid5(NAMESPACE, "\0".join((path, str(span.start), str(span.end)))))
    if node.node_id != expected_id:
        add("id")
    start_line, end_line = metadata.get("start_line"), metadata.get("end_line")
    if (start_line, end_line) != (_line(text, span.start), _line(text, span.end - 1)):
        add("lines")
    elif isinstance(start_line, int) and isinstance(end_line, int):
        rows = text.split("\n")[start_line - 1 : end_line]
        whole_lines = "\n".join(rows).rstrip()
        if node.text != whole_lines and not (
            span.forced and len(rows) == 1 and node.text in rows[0]
        ):
            add("line_text")
    stored = node_to_metadata_dict(node, remove_text=True, flat_metadata=True)
    back = metadata_dict_to_node(stored, text=node.text)
    if (back.node_id, back.hash, list(back.metadata)) != (node.node_id, node.hash, list(metadata)):
        add("round_trip")
    return findings


def validate(documents: Sequence[Document], result: ChunkResult, cap: int) -> list[Finding]:
    """Every broken expectation of one chunking run, as content-free records."""
    findings: list[Finding] = []
    nodes_by_path: dict[str, list[tuple[TextNode, Span]]] = {}
    order: list[str] = []
    for node, span in zip(result.nodes, result.spans, strict=True):
        path = str(node.metadata.get("file_path"))
        if not order or order[-1] != path:
            order.append(path)
        nodes_by_path.setdefault(path, []).append((node, span))
    if len(order) != len(set(order)) or order != [
        d.doc_id for d in documents if d.doc_id in nodes_by_path
    ]:
        findings.append(Finding("order", "*"))
    for document in documents:
        pairs = nodes_by_path.get(document.doc_id, [])
        spans = [span for _, span in pairs]
        if [s.start for s in spans] != sorted({s.start for s in spans}):
            findings.append(Finding("order", document.doc_id))
        findings += _invariant_findings(document, spans, cap)
        for node, span in pairs:
            findings += _node_findings(document, node, span)
    ids = Counter(node.node_id for node in result.nodes)
    hashes = Counter(node.hash for node in result.nodes)
    findings += [Finding("duplicate_id", "*", length=c) for c in ids.values() if c > 1]
    findings += [Finding("duplicate_hash", "*", length=c) for c in hashes.values() if c > 1]
    findings += [Finding("degraded", path) for path in result.report.degraded_files]
    return sorted(findings)


@dataclass(frozen=True, slots=True)
class SplitDefinitions:
    """Python definitions (``ast``, any depth) and how many of them the chunker split."""

    total: int
    over_cap: int
    split: int
    split_that_fit: int


def split_definitions(
    documents: Sequence[Document], result: ChunkResult, cap: int
) -> SplitDefinitions:
    total = over_cap = split = split_that_fit = 0
    for document in documents:
        if document.metadata["language"] != "python":
            continue
        spans = [
            s
            for n, s in zip(result.nodes, result.spans, strict=True)
            if n.metadata["file_path"] == document.doc_id
        ]
        for region in definition_regions(document.text) or []:
            total += 1
            over_cap += region.size > cap
            if sum(1 for s in spans if s.start < region.end and region.start < s.end) > 1:
                split += 1
                split_that_fit += region.size <= cap
    return SplitDefinitions(total, over_cap, split, split_that_fit)


Snapshot = dict[str, tuple[int, int, int]]


def tree_snapshot(root: Path) -> Snapshot:
    """(mode, size, mtime_ns) of every entry below ``root``; symlinks are not followed."""
    entries: Snapshot = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as scan:
            for entry in scan:
                info = entry.stat(follow_symlinks=False)
                entries[os.path.relpath(entry.path, root)] = (
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                )
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
    return entries


@dataclass(frozen=True, slots=True)
class ReviewPick:
    path: str
    criterion: str


def review_sample(documents: Sequence[Document], result: ChunkResult) -> list[ReviewPick]:
    """Deterministic, stratified sample for the owner's review (Gate 3 Q2): paths only.

    Each criterion takes the best file not chosen yet; ties go to the smaller path.
    """
    stats: dict[str, Counter[str]] = {d.doc_id: Counter() for d in documents}
    language = {d.doc_id: d.metadata["language"] for d in documents}
    for document in documents:
        if language[document.doc_id] == "markdown":
            stats[document.doc_id]["fences"] = len(fence_regions(document.text))
    for node, span in zip(result.nodes, result.spans, strict=True):
        counts = stats[str(node.metadata["file_path"])]
        counts["nodes"] += 1
        counts["forced"] += span.forced
        counts[span.kind.value] += 1
    criteria: list[tuple[str, str, str | None]] = [
        ("python", "most forced spans", "forced"),
        ("python", "split class (class_header)", "class_header"),
        ("python", "most module blocks", "module_block"),
        ("markdown", "most nodes", "nodes"),
        ("markdown", "most forced spans", "forced"),
        ("markdown", "most code fences", "fences"),
        ("markdown", "root README", None),
        ("dockerfile", "the Dockerfile", None),
        ("makefile", "the Makefile", None),
        ("toml", "the TOML file", None),
        ("yaml", "most forced spans", "forced"),
    ]
    picks: list[ReviewPick] = []
    chosen: set[str] = set()
    for label, criterion, key in criteria:
        candidates = [p for p in sorted(stats) if language[p] == label and p not in chosen]
        if label == "markdown" and key is None:
            candidates = [p for p in candidates if p.lower() == "readme.md"]
        if key is not None:
            candidates = sorted(candidates, key=lambda p: (-stats[p][key], p))
            candidates = [p for p in candidates if stats[p][key] > 0]
        if candidates:
            picks.append(ReviewPick(candidates[0], criterion))
            chosen.add(candidates[0])
    return picks
