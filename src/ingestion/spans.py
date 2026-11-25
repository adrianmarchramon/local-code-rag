"""Span model of the Phase 2 segmenter: what a file is cut into, before any LlamaIndex node.

A :class:`Span` is a half-open range ``[start, end)`` of **character** offsets into the
document text (never bytes): its content is the literal slice ``text[start:end]``. It
carries what Step 4 needs to build a node without re-parsing: its kind, whether it comes
from splitting something that did not fit (``forced``), and its name path (``("f",)``,
``("C",)``, ``("C", "m")``; empty for module-level code). Line numbers and IDs are derived
from the offsets later (Step 4) and are deliberately not stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from src.errors import ChunkingError


@unique
class SpanKind(StrEnum):
    """What a span holds. Closed set: consumers may match on it exhaustively."""

    FUNCTION = "function"  # a whole top-level function, or a forced fragment of one
    CLASS = "class"  # a whole class that fits within the cap (also a nested class)
    CLASS_HEADER = "class_header"  # decorators, comments, `class` line, docstring, ...
    METHOD = "method"  # a method of a class that did not fit, or a forced fragment of one
    CLASS_BLOCK = "class_block"  # class-body statements between or after its methods
    MODULE_BLOCK = "module_block"  # contiguous module-level code outside definitions


@unique
class ParseStatus(StrEnum):
    """Whether the syntax tree had errors (the file is still segmented: decision Gate 2)."""

    CLEAN = "clean"
    HAS_ERRORS = "has_errors"


@dataclass(frozen=True, slots=True)
class Span:
    """A contiguous, non-empty character range of the document, with its provenance."""

    start: int
    end: int
    kind: SpanKind
    forced: bool
    name_path: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ChunkingError(
                f"Invalid span [{self.start}, {self.end}): offsets must satisfy "
                "0 <= start < end. This is a segmenter bug; please report it."
            )
        if not isinstance(self.name_path, tuple) or not all(
            isinstance(part, str) and part for part in self.name_path
        ):
            raise ChunkingError(f"Invalid name path {self.name_path!r}: expected a tuple of names.")


@dataclass(frozen=True, slots=True)
class Segmentation:
    """The spans of one document, in text order, and the state of its syntax tree.

    ``error_regions`` counts the top-level nodes that contained syntax errors; their lines
    were segmented as module-level code (salvage policy, decision Gate 2).
    """

    spans: tuple[Span, ...]
    status: ParseStatus
    error_regions: int = 0

    def __post_init__(self) -> None:
        if (self.status is ParseStatus.CLEAN) != (self.error_regions == 0):
            raise ChunkingError(
                f"Inconsistent segmentation status {self.status.value!r} with "
                f"{self.error_regions} error regions."
            )
