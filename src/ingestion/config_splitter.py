"""Configuration fallbacks: YAML, TOML, Dockerfile, Makefile (Phase 2, Step 5; P3, Gate 1).

Each dialect finds its **top-level blocks**; column-0 comments directly above a block
(blank lines allowed in between) stay with it, and anything before the first block joins
the first one. Whole blocks are merged greedily up to the cap; a block that fits is never
split. A block that does not fit is split (every piece ``forced``):

* YAML: at the keys or list items of the next indentation level, recursively, each kept
  whole when it fits (Gate 1 Q4); a block without deeper structure falls back to blank
  lines, then lines, then characters. Top level = column-0 keys, list items and ``---``.
* TOML: tables and arrays of tables (``[t]``, ``[[t]]``) outside multi-line strings and
  arrays; an oversized table splits at blank lines between statements, then lines.
* Dockerfile: one block per instruction with its ``\\`` continuations and heredocs; each
  ``FROM`` starts a new chunk (stages are never merged); then blank lines and lines.
* Makefile: a rule with its recipe (tab-indented lines, blank lines between them
  included), ``define ... endef``, conditionals up to their ``endif``, and ``\\``
  continuations; then blank lines and lines.

Config spans carry no name path (no ``symbol``, Gate 1 Q2). Every scan is one pass over
the lines; the only regular expression (heredoc markers) has no nested repetition.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from src.ingestion.lines import Block, LineIndex, blank_line_blocks, pack_blocks, require_cap
from src.ingestion.spans import ParseStatus, Segmentation, Span, SpanKind

_HEREDOC: Final = re.compile(r"<<(-?)([\"']?)([A-Za-z_][A-Za-z0-9_]*)\2")
_CONDITIONALS: Final = frozenset({"ifeq", "ifneq", "ifdef", "ifndef"})


def _rows(index: LineIndex) -> list[str]:
    return [index.text[index.starts[r] : index.content_ends[r]] for r in range(index.last_row + 1)]


def _is_col0_comment(line: str) -> bool:
    return line.startswith("#")


def _top_ranges(index: LineIndex, starts: list[int], rows: list[str]) -> list[tuple[int, int, int]]:
    """``(first, start, last)`` of every top-level block: ``first`` includes the column-0
    comments directly above ``start`` (blank lines allowed in between)."""
    first_row = next(r for r in range(index.last_row + 1) if not index.blank[r])
    if not starts:
        return [(first_row, first_row, index.trim(first_row, index.last_row))]
    adjusted: list[int] = []
    for position, start in enumerate(starts):
        lower = starts[position - 1] + 1 if position else 0
        row, new_start = start - 1, start
        while row >= lower and (index.blank[row] or _is_col0_comment(rows[row])):
            if not index.blank[row]:
                new_start = row
            row -= 1
        adjusted.append(new_start)
    adjusted[0] = min(adjusted[0], first_row)
    ranges = []
    for k, first in enumerate(adjusted):
        last = (adjusted[k + 1] - 1) if k + 1 < len(adjusted) else index.last_row
        ranges.append((first, starts[k], index.trim(first, last)))
    return ranges


def _top_blocks(
    index: LineIndex,
    ranges: list[tuple[int, int, int]],
    cap: int,
    build: Callable[[int, int], Block],
) -> list[Block]:
    """One block per range; leading comments that would push a block that fits over the
    cap get their own chunk instead (as for Python definitions, Gate 1 policies)."""
    blocks: list[Block] = []
    for first, start, last in ranges:
        if first < start and index.size(first, last) > cap and index.size(start, last) <= cap:
            blocks.append(build(first, index.trim(first, start - 1)))
            blocks[-1] = Block(
                blocks[-1].first, blocks[-1].last, blocks[-1].children, starts_chunk=True
            )
            blocks.append(Block(start, last))
        else:
            blocks.append(build(first, last))
    return blocks


def _segmentation(spans: list[Span]) -> Segmentation:
    return Segmentation(tuple(spans), ParseStatus.CLEAN)


def _pack(index: LineIndex, blocks: list[Block], cap: int) -> list[Span]:
    return pack_blocks(
        index, blocks, cap=cap, kind=SpanKind.CONFIG_BLOCK, name_path=(), forced=False
    )


def _leaf(index: LineIndex, first: int, last: int, cap: int) -> Block:
    if index.size(first, last) <= cap:
        return Block(first, last)
    return Block(first, last, children=blank_line_blocks(index, first, last))


# --- YAML -----------------------------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _yaml_is_comment(line: str) -> bool:
    return line.lstrip(" ").startswith("#")


def _yaml_entry(line: str) -> bool:
    """A key (``key:`` / ``key: value``) or a list item (``-``), not scalar text."""
    stripped = line.lstrip(" ")
    if stripped == "-" or stripped.startswith("- "):
        return True
    key, colon, rest = stripped.partition(":")
    return bool(colon) and bool(key) and (not rest or rest[0] in " \t") and key[0] not in "{[|>"


def _opens_block_scalar(line: str) -> bool:
    """``key: |``, ``- >-`` ...: the lines below are one scalar value, not structure."""
    parts = line.split()
    token = parts[-1] if parts else ""
    return (
        token[:1] in {"|", ">"} and len(token) <= 3 and all(c in "+-0123456789" for c in token[1:])
    )


def _yaml_block(index: LineIndex, rows: list[str], first: int, last: int, cap: int) -> Block:
    """A YAML block; its children (computed only when it does not fit) are the keys or
    list items one indentation level deeper, each with the comments directly above it."""
    if index.size(first, last) <= cap:
        return Block(first, last)
    content = [
        r for r in range(first, last + 1) if not index.blank[r] and not _yaml_is_comment(rows[r])
    ]
    if len(content) < 2 or _opens_block_scalar(rows[content[0]]):
        return _leaf(index, first, last, cap)
    head, inner = content[0], content[1:]
    depth = min(_indent(rows[r]) for r in inner)
    child_starts = [r for r in inner if _indent(rows[r]) == depth and _yaml_entry(rows[r])]
    if not child_starts:
        return _leaf(index, first, last, cap)
    starts: list[int] = []
    lower = head + 1
    for start in child_starts:
        row, new_start = start - 1, start
        while row >= lower and (
            index.blank[row] or (_yaml_is_comment(rows[row]) and _indent(rows[row]) >= depth)
        ):
            if not index.blank[row]:
                new_start = row
            row -= 1
        starts.append(new_start)
        lower = start + 1
    header_last = index.trim(first, starts[0] - 1)
    children = [Block(first, header_last, header=True)]
    for k, start in enumerate(starts):
        end = index.trim(start, (starts[k + 1] - 1) if k + 1 < len(starts) else last)
        children.append(_yaml_block(index, rows, start, end, cap))
    return Block(first, last, children=tuple(children))


def split_yaml(text: str, *, max_chars: int) -> Segmentation:
    """Spans of a YAML document."""
    cap = require_cap(max_chars)
    if not text.strip():
        return _segmentation([])
    index = LineIndex(text)
    rows = _rows(index)
    starts = [r for r, line in enumerate(rows) if line and line[0] not in " \t#"]
    ranges = _top_ranges(index, starts, rows)
    blocks = _top_blocks(index, ranges, cap, lambda a, b: _yaml_block(index, rows, a, b, cap))
    return _segmentation(_pack(index, blocks, cap))


# --- TOML -----------------------------------------------------------------------------------


def _toml_scan(line: str, multiline: str | None, depth: int) -> tuple[str | None, int]:
    """Lexer state after ``line``: the open multi-line string delimiter and bracket depth."""
    position, length = 0, len(line)
    while position < length:
        if multiline is not None:
            close = line.find(multiline, position)
            if close < 0:
                return multiline, depth
            position, multiline = close + 3, None
            continue
        char = line[position]
        if char == "#":
            break
        if char in "\"'":
            if line.startswith(char * 3, position):
                multiline, position = char * 3, position + 3
                continue
            position += 1
            while position < length and line[position] != char:
                position += 2 if char == '"' and line[position] == "\\" else 1
            position += 1
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        position += 1
    return multiline, depth


def split_toml(text: str, *, max_chars: int) -> Segmentation:
    """Spans of a TOML document."""
    cap = require_cap(max_chars)
    if not text.strip():
        return _segmentation([])
    index = LineIndex(text)
    rows = _rows(index)
    starts: list[int] = []
    boundary = [True] * len(rows)  # row starts outside any string or array
    multiline: str | None = None
    depth = 0
    for row, line in enumerate(rows):
        boundary[row] = multiline is None and depth == 0
        if boundary[row] and line.lstrip().startswith("["):
            starts.append(row)
            continue  # a table header: its brackets are not values
        multiline, depth = _toml_scan(line, multiline, depth)

    def table(first: int, last: int) -> Block:
        if index.size(first, last) <= cap:
            return Block(first, last)
        pieces: list[Block] = []
        current: int | None = None
        for row in range(first, last + 1):
            if index.blank[row] and boundary[row]:
                if current is not None:
                    pieces.append(Block(current, index.trim(current, row - 1)))
                    current = None
            elif current is None and not index.blank[row]:
                current = row
        if current is not None:
            pieces.append(Block(current, index.trim(current, last)))
        return Block(first, last, children=tuple(pieces))

    blocks = _top_blocks(index, _top_ranges(index, starts, rows), cap, table)
    return _segmentation(_pack(index, blocks, cap))


# --- Dockerfile -----------------------------------------------------------------------------


def split_dockerfile(text: str, *, max_chars: int) -> Segmentation:
    """Spans of a Dockerfile: instructions merged up to the cap, never across a FROM."""
    cap = require_cap(max_chars)
    if not text.strip():
        return _segmentation([])
    index = LineIndex(text)
    rows = _rows(index)
    starts: list[int] = []
    stage_starts: set[int] = set()
    heredocs: list[tuple[str, bool]] = []
    continuing = False
    for row, line in enumerate(rows):
        if heredocs:
            delimiter, strip_tabs = heredocs[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate.rstrip() == delimiter:
                heredocs.pop(0)
            continue
        if continuing:
            if not line.lstrip().startswith("#"):
                heredocs += [(m.group(3), m.group(1) == "-") for m in _HEREDOC.finditer(line)]
                continuing = line.rstrip().endswith("\\")
            continue
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        starts.append(row)
        if stripped.split(maxsplit=1)[0].upper() == "FROM":
            stage_starts.add(row)
        heredocs += [(m.group(3), m.group(1) == "-") for m in _HEREDOC.finditer(line)]
        continuing = line.rstrip().endswith("\\")
    spans: list[Span] = []
    stage: list[Block] = []
    for first, start, last in _top_ranges(index, starts, rows):
        if start in stage_starts and stage:
            spans += _pack(index, stage, cap)
            stage = []
        stage += _top_blocks(
            index, [(first, start, last)], cap, lambda a, b: _leaf(index, a, b, cap)
        )
    spans += _pack(index, stage, cap)
    return _segmentation(spans)


# --- Makefile -------------------------------------------------------------------------------


def split_makefile(text: str, *, max_chars: int) -> Segmentation:
    """Spans of a Makefile: rules with their recipes, defines and conditionals."""
    cap = require_cap(max_chars)
    if not text.strip():
        return _segmentation([])
    index = LineIndex(text)
    rows = _rows(index)
    starts: list[int] = []
    in_define = False
    conditional_depth = 0
    continuing = False
    for row, line in enumerate(rows):
        if continuing:
            continuing = line.endswith("\\")
            continue
        stripped = line.lstrip()
        word = stripped.split(maxsplit=1)[0] if stripped else ""
        if in_define:
            in_define = word != "endef"
            continue
        if not stripped or line.startswith("\t") or stripped.startswith("#"):
            continue
        if conditional_depth:
            if word == "endif":
                conditional_depth -= 1
            elif word in _CONDITIONALS:
                conditional_depth += 1
            continuing = line.endswith("\\")
            continue
        starts.append(row)
        if word == "define":
            in_define = True
        elif word in _CONDITIONALS:
            conditional_depth = 1
        continuing = line.endswith("\\")
    ranges = _top_ranges(index, starts, rows)
    blocks = _top_blocks(index, ranges, cap, lambda a, b: _leaf(index, a, b, cap))
    return _segmentation(_pack(index, blocks, cap))


Splitter = Callable[..., Segmentation]

# Loader label -> configuration splitter (each satisfies strategies.Segmenter).
CONFIG_SPLITTERS: Final[Mapping[str, Splitter]] = MappingProxyType(
    {
        "yaml": split_yaml,
        "toml": split_toml,
        "dockerfile": split_dockerfile,
        "makefile": split_makefile,
    }
)
