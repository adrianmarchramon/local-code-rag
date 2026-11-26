"""Markdown fallback (Phase 2, Step 5; decision P3 and Gate 1 Q2-Q3).

A section starts at each ATX (``#``) or setext (underlined) heading found **outside**
fenced code blocks; the text before the first heading (front matter included) is its own
preamble section. Fences open with three or more backticks or tildes at any indentation
(so fences inside list items count), close with a run of the same character at least as
long and nothing else, and an unclosed fence runs to the end of the file. A ``---`` line
after a blank line is a thematic break and a leading ``---`` block is front matter, never
a setext underline.

* A heading with no body joins the following section, which keeps its own heading path
  (Gate 1 Q3); nothing else is merged.
* A section that fits the cap is one span. One that does not is split at blank lines
  outside fences (a fence is one block and is never split when it fits), then lines, then
  characters; its heading stays with its first part and every piece is ``forced``.
* The name path is the heading path (empty headings skipped); Step 4 renders it as the
  bounded ``symbol`` (Gate 1 Q2).

One pass over the lines with plain string operations (no regular expressions): linear in
the size of the text whatever it contains.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.lines import Block, LineIndex, pack_blocks, require_cap
from src.ingestion.spans import ParseStatus, Segmentation, Span, SpanKind

_FENCE_CHARS = "`~"
_MAX_ATX_LEVEL = 6


@dataclass(frozen=True, slots=True)
class _Heading:
    first: int  # first row of the heading (a setext heading spans its paragraph)
    last: int  # last row of the heading construct (the underline for setext)
    level: int
    text: str


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _fence_opener(line: str) -> str | None:
    """The fence marker (e.g. "```") if ``line`` opens a fenced code block."""
    stripped = line.lstrip(" \t")
    if not stripped or stripped[0] not in _FENCE_CHARS:
        return None
    char = stripped[0]
    run = len(stripped) - len(stripped.lstrip(char))
    if run < 3:
        return None
    if char == "`" and "`" in stripped[run:]:
        return None  # an info string with a backtick is inline code, not a fence
    return char * run


def _closes(line: str, marker: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(marker) and stripped == marker[0] * len(stripped)


def _atx(line: str) -> tuple[int, str] | None:
    """(level, text) of an ATX heading line, or None."""
    if _leading_spaces(line) > 3:
        return None
    stripped = line.lstrip(" ")
    level = len(stripped) - len(stripped.lstrip("#"))
    if not 1 <= level <= _MAX_ATX_LEVEL:
        return None
    rest = stripped[level:]
    if rest and rest[0] not in " \t":
        return None
    text = rest.strip()
    closing = text.rstrip("#")
    if closing != text and (not closing or closing[-1] in " \t"):
        text = closing.rstrip()  # optional closing sequence: "## Title ##"
    return level, text


def _setext_level(line: str) -> int | None:
    if _leading_spaces(line) > 3:
        return None
    stripped = line.strip()
    if not stripped or stripped[0] not in "=-" or stripped != stripped[0] * len(stripped):
        return None
    return 1 if stripped[0] == "=" else 2


def _is_list_item(line: str) -> bool:
    stripped = line.lstrip(" \t")
    if stripped[:2] in {"- ", "* ", "+ "} or stripped in {"-", "*", "+"}:
        return True
    digits = len(stripped) - len(stripped.lstrip("0123456789"))
    return 0 < digits <= 9 and stripped[digits : digits + 1] in {".", ")"}


def _scan(index: LineIndex) -> tuple[list[_Heading], list[bool]]:
    """Headings outside fences, and which rows are inside a fence (opener and closer too)."""
    rows = [index.text[index.starts[r] : index.content_ends[r]] for r in range(index.last_row + 1)]
    in_fence = [False] * len(rows)
    headings: list[_Heading] = []
    marker: str | None = None
    paragraph_start: int | None = None  # first row of the current paragraph
    first_content = next((r for r, line in enumerate(rows) if line.strip()), None)
    front_matter_end = -1
    if first_content == 0 and rows[0].strip() == "---":
        closer = next((r for r in range(1, len(rows)) if rows[r].strip() in {"---", "..."}), None)
        front_matter_end = closer if closer is not None else -1
    for row, line in enumerate(rows):
        if row <= front_matter_end:
            continue
        if marker is not None:
            in_fence[row] = True
            if _closes(line, marker):
                marker = None
            continue
        if not line.strip():
            paragraph_start = None
            continue
        opener = _fence_opener(line)
        if opener is not None:
            marker = opener
            in_fence[row] = True
            paragraph_start = None
            continue
        atx = _atx(line)
        if atx is not None:
            headings.append(_Heading(row, row, atx[0], atx[1]))
            paragraph_start = None
            continue
        level = _setext_level(line)
        if level is not None and paragraph_start is not None:
            text = " ".join(rows[r].strip() for r in range(paragraph_start, row))
            headings.append(_Heading(paragraph_start, row, level, text))
            paragraph_start = None
            continue
        if paragraph_start is None and _leading_spaces(line) >= 4:
            continue  # indented code does not start a paragraph (no setext heading)
        if paragraph_start is None and not _is_list_item(line) and level is None:
            paragraph_start = row
        elif _is_list_item(line):
            paragraph_start = None
    return headings, in_fence


def _children(
    index: LineIndex, first: int, last: int, heading_last: int | None, in_fence: list[bool]
) -> tuple[Block, ...]:
    """Blank-line blocks outside fences; the heading is a header block kept with the next."""
    blocks: list[Block] = []
    start = first
    if heading_last is not None:
        blocks.append(Block(first, heading_last, header=True))
        start = heading_last + 1
    current: int | None = None
    for row in range(start, last + 1):
        if index.blank[row] and not in_fence[row]:
            if current is not None:
                blocks.append(Block(current, index.trim(current, row - 1)))
                current = None
        elif current is None:
            current = row
    if current is not None:
        blocks.append(Block(current, index.trim(current, last)))
    return tuple(b for b in blocks if not index.blank[b.first])


def split_markdown(text: str, *, max_chars: int) -> Segmentation:
    """Spans of a Markdown document: one per section, split only when over the cap."""
    cap = require_cap(max_chars)
    if not text.strip():
        return Segmentation((), ParseStatus.CLEAN)
    index = LineIndex(text)
    headings, in_fence = _scan(index)

    # (first row, heading or None) for every section, preamble included.
    starts: list[tuple[int, _Heading | None]] = []
    first_row = next(r for r in range(index.last_row + 1) if not index.blank[r])
    if not headings or headings[0].first > first_row:
        starts.append((first_row, None))
    starts += [(h.first, h) for h in headings]

    stack: list[tuple[int, str]] = []
    spans: list[Span] = []
    pending: int | None = None  # first row of heading-only sections waiting to join
    for position, (start, heading) in enumerate(starts):
        end = starts[position + 1][0] - 1 if position + 1 < len(starts) else index.last_row
        end = index.trim(start, end)
        path: tuple[str, ...] = ()
        if heading is not None:
            while stack and stack[-1][0] >= heading.level:
                stack.pop()
            stack.append((heading.level, heading.text))
            path = tuple(name for _, name in stack if name)
        body_empty = heading is not None and all(
            index.blank[r] for r in range(heading.last + 1, end + 1)
        )
        is_last = position + 1 == len(starts)
        if body_empty and not is_last:
            pending = start if pending is None else pending
            continue
        first = start if pending is None else pending
        pending = None
        if index.size(first, end) <= cap:
            spans.append(
                Span(index.starts[first], index.content_ends[end], SpanKind.SECTION, False, path)
            )
            continue
        heading_last = heading.last if heading is not None else None
        children = _children(index, start, end, heading_last, in_fence)
        if first != start:  # joined heading-only rows lead and stay with what follows
            children = (Block(first, index.trim(first, start - 1), header=True), *children)
        spans.extend(
            pack_blocks(
                index, children, cap=cap, kind=SpanKind.SECTION, name_path=path, forced=True
            )
        )
    return Segmentation(tuple(spans), ParseStatus.CLEAN)
