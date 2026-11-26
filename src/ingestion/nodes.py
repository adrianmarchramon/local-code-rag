"""Phase 2, Step 4: from spans to LlamaIndex ``TextNode`` objects (decision P4, Gate 1).

Node contract (consumed by Phases 3-6; docs/decisions.md section 11):

* ``text`` is the literal slice ``document.text[span.start:span.end]`` (never stripped);
  ``start_char_idx``/``end_char_idx`` are the span's character offsets.
* ``metadata`` is a fresh dict per node, in this fixed order (the node hash, which
  Phase 4's fusion deduplicates on, depends on keys, values, types and order): the six
  document keys, ``symbol`` (only when the span has a name path; never ``None`` or ``""``),
  ``start_line`` and ``end_line`` (int, 1-based, inclusive, never omitted).
* Only ``file_path``, ``language`` and ``symbol`` reach the embedding model and the LLM;
  every other key is listed in fresh exclusion lists per node.
* ``id_`` = UUID5 of a fixed namespace and ``doc_id``, start and end offsets: the same
  document text gives the same IDs on every run, so re-chunking at Phase 6 start-up
  matches what Phase 3 persisted. It shifts when the offsets move (any edit above the span).
* Relationships: ``SOURCE`` only (the document); no previous/next links.

Pure functions: no re-parsing, no I/O. Lines are counted on ``\\n`` only (``LineIndex``).
"""

from __future__ import annotations

import uuid
from typing import Final

from llama_index.core import Document
from llama_index.core.schema import NodeRelationship, TextNode

from src.ingestion.documents import HIDDEN_METADATA_KEYS, METADATA_KEYS
from src.ingestion.lines import LineIndex
from src.ingestion.spans import Segmentation
from src.ingestion.strategies import SymbolStyle

# Part of the contract: changing it changes every node ID (and the golden test).
NODE_ID_NAMESPACE: Final = uuid.UUID("41a05dbb-07dd-4b83-840e-8c25648f281a")

SYMBOL_KEY: Final = "symbol"
START_LINE_KEY: Final = "start_line"
END_LINE_KEY: Final = "end_line"
NODE_METADATA_KEYS: Final = (*METADATA_KEYS, SYMBOL_KEY, START_LINE_KEY, END_LINE_KEY)
VISIBLE_NODE_KEYS: Final = ("file_path", "language", SYMBOL_KEY)
HIDDEN_NODE_KEYS: Final = (*HIDDEN_METADATA_KEYS, START_LINE_KEY, END_LINE_KEY)
_ELLIPSIS: Final = "…"


def node_id(doc_id: str, start: int, end: int) -> str:
    """Deterministic node ID: UUID5 of the namespace and ``doc_id``, start, end."""
    return str(uuid.uuid5(NODE_ID_NAMESPACE, "\0".join((doc_id, str(start), str(end)))))


def render_symbol(name_path: tuple[str, ...], style: SymbolStyle) -> str | None:
    """The ``symbol`` value of a name path, or None when the span has no name.

    A bounded style keeps the most specific end of the path: ``"… > Section"``.
    """
    if not name_path:
        return None
    full = style.separator.join(name_path)
    limit = style.max_chars
    if limit is None or len(full) <= limit:
        return full
    marker = _ELLIPSIS + style.separator
    kept = name_path[-1]
    if len(marker) + len(kept) > limit:
        return _ELLIPSIS + kept[len(kept) - (limit - len(_ELLIPSIS)) :]
    for part in reversed(name_path[:-1]):
        candidate = part + style.separator + kept
        if len(marker) + len(candidate) > limit:
            break
        kept = candidate
    return marker + kept


def build_nodes(
    document: Document, segmentation: Segmentation, style: SymbolStyle
) -> tuple[TextNode, ...]:
    """One node per span, in span order; see the module docstring for the contract."""
    text = document.text
    lines = LineIndex(text)
    source = document.as_related_node_info()  # hashes the document once, not per node
    base = {key: document.metadata[key] for key in METADATA_KEYS}
    nodes: list[TextNode] = []
    for span in segmentation.spans:
        metadata: dict[str, str | int] = dict(base)
        symbol = render_symbol(span.name_path, style)
        if symbol is not None:
            metadata[SYMBOL_KEY] = symbol
        metadata[START_LINE_KEY] = lines.line_of(span.start)
        metadata[END_LINE_KEY] = lines.line_of(span.end - 1)
        nodes.append(
            TextNode(
                id_=node_id(document.doc_id, span.start, span.end),
                text=text[span.start : span.end],
                metadata=metadata,
                excluded_embed_metadata_keys=list(HIDDEN_NODE_KEYS),
                excluded_llm_metadata_keys=list(HIDDEN_NODE_KEYS),
                start_char_idx=span.start,
                end_char_idx=span.end,
                relationships={NodeRelationship.SOURCE: source.model_copy(deep=True)},
            )
        )
    return tuple(nodes)
