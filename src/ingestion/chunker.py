"""Phase 2 chunking parameters.

Steps 1-2 only define the parameters, as an immutable, validated dataclass (decision log:
"chunker parameters in a frozen dataclass in the module"). ``chunk_documents`` arrives in
Step 6; the modules that do the work (:mod:`src.ingestion.segmenter`,
:mod:`src.ingestion.parser_registry`) receive these values as arguments and never import
this module, so there is no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.errors import InvalidChunkingRulesError

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
