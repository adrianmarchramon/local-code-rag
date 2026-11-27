"""Digests of chunking output (Phase 2 definition, moved to ``src/`` by decision P6).

A document's digest is the sha256 of the UTF-8 lines ``"<node id>:<node hash>"`` of its
nodes, in order, joined by ``"\\n"``; the corpus digest is the sha256 of the lines
``"<path>:<document digest>"`` in document order. A node hash covers its text and its
metadata (LlamaIndex ``TextNode.hash``), so any changed boundary, text or metadata changes
the digest. Phase 3 stores it with the index (decision P5); the Phase 2 key test keeps
comparing it with its versioned manifest, an independent expected value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from llama_index.core.schema import BaseNode

DIGEST_DEFINITION: Final = (
    "document = sha256 of the lines '<node id>:<node hash>' of its nodes in order, joined "
    "by \\n (UTF-8); corpus = sha256 of the lines '<path>:<document digest>' in document "
    "order, joined by \\n"
)


def _sha256(lines: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def document_digest(nodes: Sequence[BaseNode]) -> str:
    """Digest of one document's nodes (see the module docstring)."""
    return _sha256(f"{node.node_id}:{node.hash}" for node in nodes)


def corpus_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """Digest of ``{"path", "digest"}`` rows, in order."""
    return _sha256(f"{row['path']}:{row['digest']}" for row in rows)


def nodes_digest(nodes: Sequence[BaseNode]) -> str:
    """The corpus digest straight from nodes, grouped by ``file_path`` in order of appearance.

    Equals the Phase 2 manifest's corpus digest for the same chunking run.
    """
    groups: dict[str, list[BaseNode]] = {}
    for node in nodes:
        groups.setdefault(str(node.metadata["file_path"]), []).append(node)
    return corpus_digest(
        [{"path": path, "digest": document_digest(group)} for path, group in groups.items()]
    )
