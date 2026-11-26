"""Phase 2, Step 3: which segmentation strategy each language label gets, as data.

Every loader label has an explicit destination (decision P5 and Gate 1): a label with a
tree-sitter grammar (:func:`src.ingestion.parser_registry.grammar_for`, the only source of
that data) is segmented syntactically; ``markdown`` by headings; ``yaml``, ``toml``,
``dockerfile`` and ``makefile`` by configuration blocks; every other label, and any
unknown label, as generic text. Every non-syntactic strategy carries the fallback reason
``no_parser``. Adding a language or a format means adding data here plus its
implementation; the orchestrator only depends on the :class:`Segmenter` protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from types import MappingProxyType
from typing import Final, Protocol

from src.ingestion.config_splitter import CONFIG_SPLITTERS
from src.ingestion.markdown_splitter import split_markdown
from src.ingestion.parser_registry import GrammarSpec, grammar_for
from src.ingestion.segmenter import segment
from src.ingestion.spans import Segmentation
from src.ingestion.text_splitter import split_text


@unique
class Strategy(StrEnum):
    """How a document is cut into spans."""

    SYNTACTIC = "syntactic"  # tree-sitter grammar (Python)
    MARKDOWN = "markdown"  # fence-aware headings
    CONFIG = "config"  # top-level configuration blocks
    TEXT = "text"  # blank-line blocks (G1-Q4)


@unique
class FallbackReason(StrEnum):
    """Why a document was not segmented syntactically."""

    NO_PARSER = "no_parser"  # its label has no grammar (P5)
    SEGMENTATION_FAILED = "segmentation_failed"  # its strategy failed on it (Gate 1, Q5)


# Labels without a grammar, each with its strategy. `python` is absent on purpose: its
# strategy comes from the parser registry. tests/test_chunking.py fails when the loader
# emits a label that is in neither place.
LABEL_STRATEGIES: Final[Mapping[str, Strategy]] = MappingProxyType(
    {
        "markdown": Strategy.MARKDOWN,
        "yaml": Strategy.CONFIG,
        "toml": Strategy.CONFIG,
        "dockerfile": Strategy.CONFIG,
        "makefile": Strategy.CONFIG,
        "javascript": Strategy.TEXT,
        "typescript": Strategy.TEXT,
        "java": Strategy.TEXT,
        "go": Strategy.TEXT,
        "rust": Strategy.TEXT,
        "c": Strategy.TEXT,
        "cpp": Strategy.TEXT,
        "ruby": Strategy.TEXT,
        "csharp": Strategy.TEXT,
        "php": Strategy.TEXT,
    }
)


def strategy_for(label: str) -> Strategy:
    """The strategy for a loader label; unknown labels get generic text."""
    if grammar_for(label) is not None:
        return Strategy.SYNTACTIC
    return LABEL_STRATEGIES.get(label, Strategy.TEXT)


def fallback_reason(strategy: Strategy) -> FallbackReason | None:
    """``no_parser`` for every strategy that is not syntactic (P5)."""
    return None if strategy is Strategy.SYNTACTIC else FallbackReason.NO_PARSER


class Segmenter(Protocol):
    """Anything that cuts a document's text into a :class:`Segmentation`."""

    def __call__(self, text: str, *, max_chars: int) -> Segmentation: ...


@dataclass(frozen=True, slots=True)
class SyntacticSegmenter:
    """Adapts :func:`src.ingestion.segmenter.segment` to the :class:`Segmenter` protocol."""

    spec: GrammarSpec

    def __call__(self, text: str, *, max_chars: int) -> Segmentation:
        return segment(text, self.spec, max_chars=max_chars)


@dataclass(frozen=True, slots=True)
class SymbolStyle:
    """How a span's name path becomes the ``symbol`` metadata value (P4, Gate 1 Q2)."""

    separator: str
    max_chars: int | None = None  # None: no bound


SYMBOL_STYLES: Final[Mapping[Strategy, SymbolStyle]] = MappingProxyType(
    {
        Strategy.SYNTACTIC: SymbolStyle("."),  # Class.method
        Strategy.MARKDOWN: SymbolStyle(" > ", max_chars=120),  # heading path, bounded
        Strategy.CONFIG: SymbolStyle(" > "),  # config spans carry no name path
        Strategy.TEXT: SymbolStyle(" > "),  # text spans carry no name path
    }
)


def segmenter_for(label: str) -> tuple[Strategy, Segmenter]:
    """The strategy for a loader label and the segmenter that implements it."""
    strategy = strategy_for(label)
    spec = grammar_for(label)
    if spec is not None:
        return strategy, SyntacticSegmenter(spec)
    if strategy is Strategy.MARKDOWN:
        return strategy, split_markdown
    if strategy is Strategy.CONFIG:
        return strategy, CONFIG_SPLITTERS[label]
    return strategy, split_text
