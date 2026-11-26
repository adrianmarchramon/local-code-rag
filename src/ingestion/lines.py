"""Line arithmetic shared by every chunking strategy: one source of truth for offsets.

* :class:`LineIndex` maps rows to **character** offsets of the ``str`` (rows split on
  ``\\n`` only, as tree-sitter and the loader count them; never ``splitlines``) and offsets
  back to 1-based line numbers with ``bisect``.
* :func:`character_pieces` is the last resort for one line longer than the cap.
* :func:`pack_blocks` packs whole blocks of rows greedily up to the cap and splits what
  does not fit into its children, then lines, then characters (forced); the fallback
  splitters use it. The Python segmenter keeps its own tree-aware packing.

A span always starts at column 0 of its first row and ends after the last non-whitespace
character of its last row, except character pieces (decisions Gate 1, docs section 10.3).
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from typing import Final

from src.errors import InvalidChunkingRulesError
from src.ingestion.spans import Span, SpanKind


class LineIndex:
    """Rows of a text split on ``\\n`` only, with their character offsets."""

    def __init__(self, text: str) -> None:
        self.text = text
        starts: list[int] = []
        content_ends: list[int] = []
        blank: list[bool] = []
        position = 0
        for row in text.split("\n"):
            content = row.rstrip()
            starts.append(position)
            content_ends.append(position + len(content))
            blank.append(not content)
            position += len(row) + 1
        self.starts: Final = tuple(starts)
        self.content_ends: Final = tuple(content_ends)
        self.blank: Final = tuple(blank)

    @property
    def last_row(self) -> int:
        return len(self.starts) - 1

    def row(self, row: int) -> str:
        """The row's text without its trailing whitespace."""
        return self.text[self.starts[row] : self.content_ends[row]]

    def size(self, first: int, last: int) -> int:
        """Characters of the span covering rows ``first..last``."""
        return self.content_ends[last] - self.starts[first]

    def trim(self, first: int, last: int) -> int:
        """``last`` moved up past trailing blank rows (never above ``first``)."""
        while last > first and self.blank[last]:
            last -= 1
        return last

    def line_of(self, offset: int) -> int:
        """1-based line of the character at ``offset``: 1 + newlines before it."""
        return bisect.bisect_right(self.starts, offset)


def character_pieces(index: LineIndex, row: int, cap: int) -> list[tuple[int, int]]:
    """Pieces of one over-long row, each at most ``cap`` characters, cut after a word
    when possible; they never start or end on whitespace."""
    text = index.text
    position, end = index.starts[row], index.content_ends[row]
    while text[position].isspace():
        position += 1
    pieces: list[tuple[int, int]] = []
    while position < end:
        if end - position <= cap:
            pieces.append((position, end))
            break
        limit = position + cap
        cut = next(
            (
                k
                for k in range(limit, position, -1)
                if text[k].isspace() and not text[k - 1].isspace()
            ),
            limit,
        )
        pieces.append((position, cut))
        position = cut
        while position < end and text[position].isspace():
            position += 1
    return pieces


@dataclass(frozen=True, slots=True)
class Block:
    """Inclusive row range with optional finer structure.

    ``children`` (contiguous, in order) are used when the block does not fit; without
    them it steps down to lines. ``header`` marks a block that must stay with what follows
    when its parent is split (a Markdown heading, a YAML parent key). ``starts_chunk``
    forbids merging the block into the chunk before it (comments that belong to what
    follows them).
    """

    first: int
    last: int
    children: tuple[Block, ...] = ()
    header: bool = False
    starts_chunk: bool = False


def _finer(index: LineIndex, block: Block) -> tuple[Block, ...] | None:
    if len(block.children) > 1:
        return block.children
    rows = [r for r in range(block.first, block.last + 1) if not index.blank[r]]
    if len(rows) > 1:
        return tuple(Block(r, r) for r in rows)
    return None


def pack_blocks(
    index: LineIndex,
    blocks: tuple[Block, ...] | list[Block],
    *,
    cap: int,
    kind: SpanKind,
    name_path: tuple[str, ...],
    forced: bool,
) -> list[Span]:
    """Greedily merge whole blocks up to ``cap``; split a block that does not fit.

    Pieces of a split block are packed on their own and are ``forced``; after a header
    in a forced split, a block that cannot join it is split further so that the header
    keeps its first part.
    """
    spans: list[Span] = []
    queue = deque(blocks)
    current: tuple[int, int] | None = None
    ends_with_header = False

    def emit(first: int, last: int, is_forced: bool) -> None:
        spans.append(
            Span(index.starts[first], index.content_ends[last], kind, is_forced, name_path)
        )

    while queue:
        block = queue.popleft()
        joinable = current is not None and not block.starts_chunk
        if current is not None and joinable and index.size(current[0], block.last) <= cap:
            current = (current[0], block.last)
            ends_with_header = block.header
            continue
        if forced and ends_with_header and (finer := _finer(index, block)) is not None:
            queue.extendleft(reversed(finer))
            continue
        if index.size(block.first, block.last) > cap:
            if current is not None:
                emit(*current, forced)
            current, ends_with_header = None, False
            finer = _finer(index, block)
            if finer is None:
                for start, end in character_pieces(index, block.first, cap):
                    spans.append(Span(start, end, kind, True, name_path))
            else:
                spans.extend(
                    pack_blocks(index, finer, cap=cap, kind=kind, name_path=name_path, forced=True)
                )
            continue
        if current is not None:
            emit(*current, forced)
        current, ends_with_header = (block.first, block.last), block.header
    if current is not None:
        emit(*current, forced)
    return spans


def blank_line_blocks(index: LineIndex, first: int, last: int) -> tuple[Block, ...]:
    """Runs of non-blank rows separated by blank rows, within ``first..last``."""
    blocks: list[Block] = []
    start: int | None = None
    for row in range(first, last + 1):
        if index.blank[row]:
            if start is not None:
                blocks.append(Block(start, row - 1))
                start = None
        elif start is None:
            start = row
    if start is not None:
        blocks.append(Block(start, index.trim(start, last)))
    return tuple(blocks)


def require_cap(max_chars: object) -> int:
    """``max_chars`` if it is a positive int; otherwise :class:`InvalidChunkingRulesError`."""
    if type(max_chars) is not int or max_chars <= 0:
        raise InvalidChunkingRulesError(
            f"max_chars must be a positive integer, got {max_chars!r}; "
            "pass ChunkingRules().max_chunk_chars."
        )
    return max_chars
