"""Generic text fallback (Phase 2, Step 5; decision Gate 1 Q4).

For every label without a grammar and without a dedicated splitter (javascript,
typescript, java, go, rust, c, cpp, ruby, csharp, php, any unknown label) and for a file
whose own strategy failed (Gate 1 Q5): blocks separated by blank lines, merged whole up to
the cap; a block that does not fit is split into lines, and a line into characters, all
``forced``. Linear in the size of the text.
"""

from __future__ import annotations

from src.ingestion.lines import LineIndex, blank_line_blocks, pack_blocks, require_cap
from src.ingestion.spans import ParseStatus, Segmentation, SpanKind


def split_text(text: str, *, max_chars: int) -> Segmentation:
    """Spans of a generic text document."""
    cap = require_cap(max_chars)
    if not text.strip():
        return Segmentation((), ParseStatus.CLEAN)
    index = LineIndex(text)
    blocks = blank_line_blocks(index, 0, index.last_row)
    spans = pack_blocks(
        index, blocks, cap=cap, kind=SpanKind.TEXT_BLOCK, name_path=(), forced=False
    )
    return Segmentation(tuple(spans), ParseStatus.CLEAN)
