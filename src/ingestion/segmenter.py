"""Syntax-aware segmentation of one source file into spans (Phase 2, Step 2).

``segment(text, spec, max_chars=...)`` parses the whole text once with tree-sitter and
cuts it into :class:`~src.ingestion.spans.Span` objects:

* every top-level definition (function or class, decorated or ``async``) is one span,
  with its decorators, docstring and leading comments;
* the remaining module-level lines form contiguous module blocks, computed as the
  complement of the definitions, so every non-whitespace character is covered;
* a class over the cap becomes a header span, one span per method and contiguous class
  blocks (nested classes recurse); a definition over the cap is split where a line starts
  a statement or clause, packing whole units greedily up to the cap and stepping down to
  lines and then characters only for what still does not fit. Those spans are ``forced``.

Policies (decisions Gate 1 and Gate 2, docs/decisions.md section 10):

* work on whole lines: rows come from the tree and are mapped to character offsets with a
  line-start index of the ``str`` (rows split on ``\\n`` only); byte offsets are never
  used to index the ``str``; an end-of-line comment stays with its line;
* a span starts at column 0 of its first non-blank line and ends after its last
  non-whitespace character (character pieces of a single over-long line excepted);
* leading comments: comment-only runs before a definition, even across blank lines, join
  it when only comments separate it from the previous code; a shebang or encoding line
  never does; if they would push a fitting definition over the cap they get their own
  span and the definition is not split;
* syntax errors: top-level nodes without errors are segmented normally; a top-level node
  containing errors is treated as module-level code (split at whole lines if needed) and
  counted in ``error_regions``;
* definitions inside a module-level ``if``/``try``/``with`` are module-level code.

Pure function: no I/O, no network, no global state besides the grammar cache.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import tree_sitter

from src.errors import InvalidChunkingRulesError
from src.ingestion.lines import LineIndex, character_pieces
from src.ingestion.parser_registry import GrammarSpec, new_parser
from src.ingestion.spans import ParseStatus, Segmentation, Span, SpanKind

logger = logging.getLogger(__name__)

# PEP 263 encoding declaration (only meaningful on the first two lines).
_ENCODING: Final = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


class _Category(Enum):
    DEFINITION = "definition"
    COMMENT = "comment"
    CODE = "code"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class _Unit:
    """Inclusive row range; ``node`` is set when the unit can be split structurally.

    ``header`` marks the lines that open a definition or compound statement (decorators,
    ``def``/``class``/``if``... line): when it is split, the header stays with its first part.
    """

    first: int
    last: int
    node: tree_sitter.Node | None = None
    header: bool = False


@dataclass(slots=True)
class _Item:
    """Sibling nodes sharing rows, merged into one row range and categorized."""

    first: int
    last: int
    nodes: list[tree_sitter.Node] = field(default_factory=list)
    category: _Category = _Category.CODE


@dataclass(frozen=True, slots=True)
class _Definition:
    comments: tuple[_Item, ...]
    item: _Item


_Element = _Definition | tuple[_Item, ...]  # a definition, or a block of plain items


class _Segmenter:
    def __init__(self, text: str, source: bytes, spec: GrammarSpec, max_chars: int) -> None:
        self.lines = LineIndex(text)
        self.source = source
        self.spec = spec
        self.cap = max_chars

    # --- tree helpers ---------------------------------------------------------------------

    def rows(self, node: tree_sitter.Node) -> tuple[int, int]:
        first, last = node.start_point.row, node.end_point.row
        if node.end_point.column == 0 and last > first:
            last -= 1  # the node ends at the very start of the next row
        while first < last and self.lines.blank[first]:
            first += 1
        return first, self.lines.trim(first, last)

    def definition_of(self, node: tree_sitter.Node) -> tree_sitter.Node:
        if node.type in self.spec.decorated_types:
            inner = node.child_by_field_name(self.spec.definition_field)
            if inner is not None:
                return inner
        return node

    def is_definition(self, node: tree_sitter.Node) -> bool:
        target = self.definition_of(node)
        return target.type in self.spec.function_types | self.spec.class_types

    def name_of(self, node: tree_sitter.Node) -> str | None:
        name = self.definition_of(node).child_by_field_name(self.spec.name_field)
        if name is None or name.start_byte == name.end_byte:
            return None
        return self.source[name.start_byte : name.end_byte].decode("utf-8")

    def inner_nodes(self, node: tree_sitter.Node) -> list[tree_sitter.Node]:
        """Statement-level children: block statements and clauses, in order."""
        inner: list[tree_sitter.Node] = []
        for child in self.definition_of(node).children:
            if child.type in self.spec.block_types:
                inner.extend(child.children)
            elif child.type in self.spec.clause_types:
                inner.append(child)
        return inner

    def body_of(self, node: tree_sitter.Node) -> tree_sitter.Node | None:
        for child in self.definition_of(node).children:
            if child.type in self.spec.block_types:
                return child
        return None

    # --- items: siblings grouped by rows ----------------------------------------------------

    def items(self, nodes: list[tree_sitter.Node], first: int, last: int) -> list[_Item]:
        """Group sibling nodes sharing rows, categorize them, and cover every row."""
        grouped: list[_Item] = []
        for node in nodes:
            if node.start_byte == node.end_byte:
                continue  # MISSING or empty nodes have no text
            start, end = self.rows(node)
            if grouped and start <= grouped[-1].last:
                grouped[-1].last = max(grouped[-1].last, end)
                grouped[-1].nodes.append(node)
            else:
                grouped.append(_Item(start, end, [node]))
        covered: list[_Item] = []
        cursor = first
        for item in grouped:
            covered.extend(self.gap(cursor, item.first - 1))
            covered.append(item)
            cursor = item.last + 1
        covered.extend(self.gap(cursor, last))
        for item in covered:
            item.category = self.category(item)
        return covered

    def gap(self, first: int, last: int) -> list[_Item]:
        """Non-blank rows no node claims (defensive: keeps coverage by construction)."""
        rows = [row for row in range(first, last + 1) if not self.lines.blank[row]]
        return [_Item(rows[0], rows[-1])] if rows else []

    def category(self, item: _Item) -> _Category:
        if not item.nodes:
            return _Category.CODE
        if any(node.has_error for node in item.nodes):
            return _Category.ERROR
        comments = [n for n in item.nodes if n.type in self.spec.comment_types]
        definitions = [n for n in item.nodes if self.is_definition(n)]
        if (
            len(definitions) == 1
            and len(definitions) + len(comments) == len(item.nodes)
            and self.name_of(definitions[0]) is not None
        ):
            return _Category.DEFINITION
        if len(comments) == len(item.nodes):
            return _Category.COMMENT
        return _Category.CODE

    def definition_node(self, item: _Item) -> tree_sitter.Node:
        return next(n for n in item.nodes if self.is_definition(n))

    def unit(self, item: _Item) -> _Unit:
        if item.category is not _Category.ERROR and len(item.nodes) == 1:
            node = item.nodes[0]
            if self.inner_nodes(node) and self.rows(node) == (item.first, item.last):
                return _Unit(item.first, item.last, node)
        return _Unit(item.first, item.last)

    # --- elements: definitions with their leading comments, and plain blocks ----------------

    def attachable(self, item: _Item, module_level: bool) -> bool:
        if item.category is not _Category.COMMENT:
            return False
        if module_level and item.first <= 1:
            line = self.lines.row(item.first)
            if (item.first == 0 and line.startswith("#!")) or _ENCODING.match(line):
                return False
        return True

    def elements(self, items: list[_Item], module_level: bool) -> list[_Element]:
        elements: list[_Element] = []
        block: list[_Item] = []
        pending: list[_Item] = []
        for item in items:
            if self.attachable(item, module_level):
                pending.append(item)
            elif item.category is _Category.DEFINITION:
                if block:
                    elements.append(tuple(block))
                block = []
                elements.append(_Definition(tuple(pending), item))
                pending = []
            else:
                block.extend(pending)
                pending = []
                block.append(item)
        block.extend(pending)
        if block:
            elements.append(tuple(block))
        return elements

    def emit(self, elements: list[_Element], prefix: tuple[str, ...], in_class: bool) -> list[Span]:
        block_kind = SpanKind.CLASS_BLOCK if in_class else SpanKind.MODULE_BLOCK
        spans: list[Span] = []
        for element in elements:
            if isinstance(element, _Definition):
                spans.extend(self.definition(element, prefix, in_class))
            else:
                units = [self.unit(item) for item in element]
                spans.extend(self.pack(units, block_kind, prefix, forced=False))
        return spans

    # --- definitions ------------------------------------------------------------------------

    def definition(
        self, element: _Definition, prefix: tuple[str, ...], in_class: bool
    ) -> list[Span]:
        item, comments = element.item, element.comments
        node = self.definition_node(item)
        path = (*prefix, self.name_of(node) or "")
        is_class = self.definition_of(node).type in self.spec.class_types
        kind = SpanKind.CLASS if is_class else SpanKind.METHOD if in_class else SpanKind.FUNCTION
        block_kind = SpanKind.CLASS_BLOCK if in_class else SpanKind.MODULE_BLOCK
        first = comments[0].first if comments else item.first
        if self.lines.size(first, item.last) <= self.cap:
            return [self.span(first, item.last, kind, False, path)]
        spans: list[Span] = []
        comment_units = [self.unit(c) for c in comments]
        if comments and self.lines.size(item.first, item.last) <= self.cap:
            spans.extend(self.pack(comment_units, block_kind, prefix, forced=False))
            spans.append(self.span(item.first, item.last, kind, False, path))
            return spans
        if not is_class:
            units = self.split_units(_Unit(item.first, item.last, node))
            return spans + self.forced_with_comments(
                comment_units, units, item.first, kind, path, block_kind, prefix
            )
        return self.big_class(item, node, comment_units, path, block_kind, prefix)

    def forced_with_comments(
        self,
        comment_units: list[_Unit],
        units: list[_Unit],
        first_row: int,
        kind: SpanKind,
        path: tuple[str, ...],
        block_kind: SpanKind,
        prefix: tuple[str, ...],
    ) -> list[Span]:
        """A forced split of a definition, its leading comments kept with its first part.

        When the comments cannot share a span with that first part they get their own
        unforced block span (no name), as when they would push a fitting definition over
        the cap: a comment-only chunk is neither part of the definition nor forced.
        """
        combined = self.pack(comment_units + units, kind, path, forced=True)
        if comment_units and combined[0].end <= self.lines.starts[first_row]:
            comments = self.pack(comment_units, block_kind, prefix, forced=False)
            return comments + self.pack(units, kind, path, forced=True)
        return combined

    def big_class(
        self,
        item: _Item,
        node: tree_sitter.Node,
        comment_units: list[_Unit],
        path: tuple[str, ...],
        parent_block_kind: SpanKind,
        prefix: tuple[str, ...],
    ) -> list[Span]:
        body = self.body_of(node)
        elements: list[_Element] = []
        if body is not None:
            body_items = self.items(body.children, self.rows(body)[0], item.last)
            elements = self.elements(body_items, module_level=False)
        # Header: everything before the first method or nested class (Gate 1, Q2).
        first_definition = next(
            (i for i, e in enumerate(elements) if isinstance(e, _Definition)), None
        )
        rest: list[_Element] = []
        header_last = item.last  # a class without methods: the header is the whole class
        if first_definition is not None:
            start = self.element_first(elements[first_definition])
            if start > item.first:
                rest = elements[first_definition:]
                header_last = self.lines.trim(item.first, start - 1)
        spans: list[Span] = []
        first = comment_units[0].first if comment_units else item.first
        if self.lines.size(first, header_last) <= self.cap:
            spans.append(self.span(first, header_last, SpanKind.CLASS_HEADER, False, path))
        elif comment_units and self.lines.size(item.first, header_last) <= self.cap:
            spans.extend(self.pack(comment_units, parent_block_kind, prefix, forced=False))
            spans.append(self.span(item.first, header_last, SpanKind.CLASS_HEADER, False, path))
        else:
            header = self.split_units(_Unit(item.first, header_last, node), stop=header_last)
            spans.extend(
                self.forced_with_comments(
                    comment_units,
                    header,
                    item.first,
                    SpanKind.CLASS_HEADER,
                    path,
                    parent_block_kind,
                    prefix,
                )
            )
        return spans + self.emit(rest, path, in_class=True)

    def element_first(self, element: _Element) -> int:
        if isinstance(element, _Definition):
            return (element.comments or (element.item,))[0].first
        return element[0].first

    # --- splitting and packing --------------------------------------------------------------

    def split_units(self, unit: _Unit, stop: int | None = None) -> list[_Unit]:
        """Units one level finer than ``unit``: statements and clauses, else lines."""
        last = unit.last if stop is None else stop
        if unit.node is not None:
            inner = [n for n in self.inner_nodes(unit.node) if self.rows(n)[0] <= last]
            groups = [g for g in self.items(inner, unit.first, last) if g.nodes]
            boundaries = [unit.first] + [g.first for g in groups if g.first > unit.first]
            has_header = bool(groups) and groups[0].first > unit.first
            units: list[_Unit] = []
            for index, start in enumerate(boundaries):
                end = boundaries[index + 1] - 1 if index + 1 < len(boundaries) else last
                end = self.lines.trim(start, end)
                match = next((g for g in groups if (g.first, g.last) == (start, end)), None)
                splittable = match is not None and len(match.nodes) == 1
                node = match.nodes[0] if match is not None and splittable else None
                node = node if node is not None and self.inner_nodes(node) else None
                units.append(_Unit(start, end, node, header=has_header and index == 0))
            if len(units) > 1:
                return units
        rows = [r for r in range(unit.first, last + 1) if not self.lines.blank[r]]
        return [_Unit(r, r) for r in rows]

    def finer(self, unit: _Unit) -> list[_Unit] | None:
        """``split_units`` when it makes progress; None for a single line without structure."""
        if unit.first == unit.last and unit.node is None:
            return None
        units = self.split_units(unit)
        return units if len(units) > 1 or units[0] != unit else None

    def pack(
        self, units: list[_Unit], kind: SpanKind, path: tuple[str, ...], forced: bool
    ) -> list[Span]:
        """Greedily pack contiguous units up to the cap; split whatever does not fit.

        The pieces of a unit that does not fit are packed on their own (they never merge
        with neighbouring units), except after a header: in a forced split, a unit that
        cannot join the header is split further so that the header keeps its first part.
        """
        spans: list[Span] = []
        queue = deque(units)
        current: tuple[int, int] | None = None
        ends_with_header = False
        while queue:
            unit = queue.popleft()
            if current is not None and self.lines.size(current[0], unit.last) <= self.cap:
                current = (current[0], unit.last)
                ends_with_header = unit.header
                continue
            if forced and ends_with_header and (finer := self.finer(unit)) is not None:
                queue.extendleft(reversed(finer))
                continue
            if self.lines.size(unit.first, unit.last) > self.cap:
                if current is not None:
                    spans.append(self.span(*current, kind, forced, path))
                current, ends_with_header = None, False
                finer = self.finer(unit)
                if finer is None:
                    spans.extend(self.characters(unit.first, kind, path))
                else:
                    spans.extend(self.pack(finer, kind, path, forced=True))
                continue
            if current is not None:
                spans.append(self.span(*current, kind, forced, path))
            current, ends_with_header = (unit.first, unit.last), unit.header
        if current is not None:
            spans.append(self.span(*current, kind, forced, path))
        return spans

    def characters(self, row: int, kind: SpanKind, path: tuple[str, ...]) -> list[Span]:
        """Last resort for one line over the cap: pieces broken after a word if possible."""
        pieces = character_pieces(self.lines, row, self.cap)
        return [Span(start, end, kind, True, path) for start, end in pieces]

    def span(
        self, first: int, last: int, kind: SpanKind, forced: bool, path: tuple[str, ...]
    ) -> Span:
        return Span(self.lines.starts[first], self.lines.content_ends[last], kind, forced, path)


def segment(text: str, spec: GrammarSpec, *, max_chars: int) -> Segmentation:
    """Spans of ``text`` (one source file), in order, with the state of its syntax tree."""
    if type(max_chars) is not int or max_chars <= 0:
        raise InvalidChunkingRulesError(
            f"max_chars must be a positive integer, got {max_chars!r}; "
            "pass ChunkingRules().max_chunk_chars."
        )
    if not text.strip():
        return Segmentation((), ParseStatus.CLEAN)
    source = text.encode("utf-8")
    tree = new_parser(spec).parse(source)
    segmenter = _Segmenter(text, source, spec, max_chars)
    last_row = len(segmenter.lines.starts) - 1
    items = segmenter.items(tree.root_node.children, 0, last_row)
    error_regions = sum(1 for item in items if item.category is _Category.ERROR)
    spans = segmenter.emit(segmenter.elements(items, module_level=True), (), in_class=False)
    status = ParseStatus.HAS_ERRORS if error_regions else ParseStatus.CLEAN
    result = Segmentation(tuple(sorted(spans, key=lambda s: s.start)), status, error_regions)
    logger.debug(
        "chunking.segmented",
        extra={
            "language": spec.label,
            "spans": len(result.spans),
            "forced_spans": sum(1 for s in result.spans if s.forced),
            "status": status.value,
            "error_regions": error_regions,
        },
    )
    return result
