"""Step 1 made executable: what a coherent unit of code is, as pure, reusable checkers.

rag_phase2.md Step 1 explains why blind chunking destroys code: it cuts functions in half
and leaves orphan fragments. These invariants turn that into something a test can fail on.
Each checker returns its violations (an empty list means the invariant holds):

* I1 bounds: ``0 <= start < end <= len(text)``; the span is the literal slice and does not
  end on whitespace.
* I2 order: spans are sorted and do not overlap.
* I3 coverage: every non-whitespace character belongs to exactly one span.
* I4 lines: a span starts at the beginning of a line and ends at the end of its content,
  unless it is a forced last-resort piece of a single over-long line.
* I5 whole definitions: a function, method or class that fits within the cap is never
  split across spans.
* I6 cap: no span is longer than the cap.
* I7 forced: a function or method split across spans has all those spans ``forced``, and
  a forced span always overlaps something that does not fit.
* I8 determinism: two runs give identical results.

The oracle for I5 and I7 is the standard library's ``ast`` (decorator lines and
``end_lineno``), not the tree-sitter tree that produced the spans. Only its line numbers
are used: its column offsets are UTF-8 bytes. When the text is not valid Python the
oracle does not apply and :func:`definition_regions` returns None.

Not a test module (pytest does not collect it), and not part of ``src/``.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.ingestion.spans import Segmentation, Span, SpanKind

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken invariant: its code (I1..I8) and where."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class Region:
    """Character range of an ``ast`` node, from column 0 of its first line (decorators
    included) to the end of the content of its last line."""

    start: int
    end: int
    name: str
    is_function: bool

    @property
    def size(self) -> int:
        return self.end - self.start


class _LineIndex:
    def __init__(self, text: str) -> None:
        self.starts: list[int] = []
        self.content_ends: list[int] = []
        position = 0
        for row in text.split("\n"):
            self.starts.append(position)
            self.content_ends.append(position + len(row.rstrip()))
            position += len(row) + 1

    def region(self, first_line: int, last_line: int) -> tuple[int, int]:
        return self.starts[first_line - 1], self.content_ends[last_line - 1]


def _node_lines(node: ast.stmt) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", [])
    first = min([node.lineno] + [d.lineno for d in decorators])
    return first, node.end_lineno or node.lineno


def definition_regions(text: str) -> list[Region] | None:
    """Every function, method and class (any depth), or None if ``text`` is not Python."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    index = _LineIndex(text)
    regions = [
        Region(*index.region(*_node_lines(node)), node.name, isinstance(node, FUNCTIONS))
        for node in ast.walk(tree)
        if isinstance(node, DEFINITIONS)
    ]
    return sorted(regions, key=lambda r: (r.start, r.end))


def statement_regions(text: str) -> list[tuple[int, int]] | None:
    """Character ranges of every statement (any depth), or None if not Python."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    index = _LineIndex(text)
    return [index.region(*_node_lines(n)) for n in ast.walk(tree) if isinstance(n, ast.stmt)]


def _content(text: str, span: Span) -> str:
    return text[span.start : span.end]


def _line_start(text: str, position: int) -> bool:
    return position == 0 or text[position - 1] == "\n"


def _line_rest(text: str, position: int) -> str:
    newline = text.find("\n", position)
    return text[position : len(text) if newline < 0 else newline]


def check_bounds(text: str, spans: Sequence[Span]) -> list[Violation]:
    """I1."""
    violations = []
    for span in spans:
        if not 0 <= span.start < span.end <= len(text):
            violations.append(Violation("I1", f"[{span.start}, {span.end}) out of bounds"))
        elif _content(text, span)[-1].isspace():
            violations.append(Violation("I1", f"[{span.start}, {span.end}) ends on whitespace"))
    return violations


def check_order(spans: Sequence[Span]) -> list[Violation]:
    """I2."""
    return [
        Violation("I2", f"[{a.start}, {a.end}) and [{b.start}, {b.end}) overlap or are unsorted")
        for a, b in itertools.pairwise(spans)
        if a.end > b.start
    ]


def check_coverage(text: str, spans: Sequence[Span]) -> list[Violation]:
    """I3."""
    owners = [0] * len(text)
    for span in spans:
        for position in range(max(span.start, 0), min(span.end, len(text))):
            owners[position] += 1
    bad = [i for i, ch in enumerate(text) if not ch.isspace() and owners[i] != 1]
    if not bad:
        return []
    return [Violation("I3", f"{len(bad)} non-whitespace characters covered != 1, first {bad[0]}")]


def check_line_boundaries(text: str, spans: Sequence[Span], cap: int) -> list[Violation]:
    """I4 (the only exception: forced pieces of a single line longer than the cap)."""
    violations = []
    for span in spans:
        if not _line_rest(text, span.start).strip():
            violations.append(Violation("I4", f"span at {span.start} starts on a blank line"))
        if span.forced and _is_piece(text, span) and _line_length(text, span.start) > cap:
            continue
        if not _line_start(text, span.start):
            violations.append(Violation("I4", f"span at {span.start} starts mid-line"))
        if _line_rest(text, span.end).strip():
            violations.append(Violation("I4", f"span ending at {span.end} ends mid-line"))
    return violations


def _is_piece(text: str, span: Span) -> bool:
    """A span inside a single line."""
    return "\n" not in _content(text, span)


def _line_length(text: str, position: int) -> int:
    start = text.rfind("\n", 0, position) + 1
    return len(_line_rest(text, start).rstrip())


def check_definitions_whole(text: str, spans: Sequence[Span], cap: int) -> list[Violation]:
    """I5 (ast oracle; nothing to check when the text is not Python)."""
    regions = definition_regions(text)
    if regions is None:
        return []
    return [
        Violation("I5", f"{region.name!r} fits ({region.size} <= {cap}) but is split")
        for region in regions
        if region.size <= cap
        and not any(s.start <= region.start and region.end <= s.end for s in spans)
    ]


def check_cap(text: str, spans: Sequence[Span], cap: int) -> list[Violation]:
    """I6."""
    return [
        Violation("I6", f"[{s.start}, {s.end}) has {s.end - s.start} > {cap} characters")
        for s in spans
        if s.end - s.start > cap
    ]


def check_forced(text: str, spans: Sequence[Span], cap: int) -> list[Violation]:
    """I7 (ast oracle; nothing to check when the text is not Python)."""
    regions = definition_regions(text)
    statements = statement_regions(text)
    if regions is None or statements is None:
        return []
    violations = []
    for region in regions:
        if not region.is_function:
            continue
        touching = [s for s in spans if s.start < region.end and region.start < s.end]
        if len(touching) > 1 and not all(s.forced for s in touching):
            violations.append(Violation("I7", f"{region.name!r} split without forced spans"))
    index = _LineIndex(text)
    lines = [(index.starts[i], index.content_ends[i]) for i in range(len(index.starts))]
    too_big = [(a, b) for a, b in statements + lines if b - a > cap]
    for span in spans:
        if span.forced and not any(a < span.end and span.start < b for a, b in too_big):
            violations.append(Violation("I7", f"forced span at {span.start} but everything fits"))
    return violations


def check_deterministic(run: Callable[[], Segmentation]) -> list[Violation]:
    """I8."""
    first, second = run(), run()
    return [] if first == second else [Violation("I8", "two runs differ")]


def check_all(
    text: str, segmentation: Segmentation, cap: int, rerun: Callable[[], Segmentation]
) -> list[Violation]:
    """I1-I8 together."""
    spans = segmentation.spans
    return (
        check_bounds(text, spans)
        + check_order(spans)
        + check_coverage(text, spans)
        + check_line_boundaries(text, spans, cap)
        + check_definitions_whole(text, spans, cap)
        + check_cap(text, spans, cap)
        + check_forced(text, spans, cap)
        + check_deterministic(rerun)
    )


# --- naive baselines (Step 1: what blind splitting does) ------------------------------------


def _trimmed(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    while not _line_start(text, start) and text[start - 1].isspace():
        start -= 1  # keep the indentation of the first line, like the segmenter
    return Span(start, end, SpanKind.MODULE_BLOCK, False, ())


def naive_line_split(text: str, lines: int) -> tuple[Span, ...]:
    """Cut every ``lines`` lines, blind to the code (rag_phase2.md Step 1 uses 4)."""
    index = _LineIndex(text)
    spans = []
    for first in range(0, len(index.starts), lines):
        last = min(first + lines, len(index.starts)) - 1
        span = _trimmed(text, index.starts[first], index.content_ends[last])
        if span is not None:
            spans.append(span)
    return tuple(spans)


def naive_char_split(text: str, chars: int) -> tuple[Span, ...]:
    """Cut every ``chars`` characters, blind to the code."""
    spans = []
    for start in range(0, len(text), chars):
        span = _trimmed(text, start, min(start + chars, len(text)))
        if span is not None:
            spans.append(span)
    return tuple(spans)
