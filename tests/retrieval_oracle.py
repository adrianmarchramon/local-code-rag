"""Exact cosine kNN in pure Python, to check Chroma's HNSW ranking (Phase 3, Step 6).

``entries`` are the stored ``(node, vector)`` pairs of a generation (vectors come back at
float32 precision), the query vector is the embedder's, and the comparison tolerates ties:
positions whose cosines differ by less than ``tolerance`` may swap. Not a test module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQuery


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def exact_ranking(
    entries: Iterable[tuple[BaseNode, list[float]]], query: Sequence[float], k: int
) -> list[tuple[str, float]]:
    scored = [(node.node_id, _cosine(query, vector)) for node, vector in entries]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:k]


def hnsw_ranking(
    store: BasePydanticVectorStore, query: Sequence[float], k: int
) -> list[tuple[str, float, float]]:
    """(node id, cosine from the score, raw score), in Chroma's order."""
    result = store.query(VectorStoreQuery(query_embedding=list(query), similarity_top_k=k))
    ids = result.ids or []
    scores = result.similarities or []
    return [(i, 1.0 + math.log(s), s) for i, s in zip(ids, scores, strict=True)]


def disagreements(
    exact: Sequence[tuple[str, float]],
    hnsw: Sequence[tuple[str, float, float]],
    tolerance: float = 1e-5,
) -> list[str]:
    """Content-free description of every difference beyond ties (empty: they agree)."""
    problems: list[str] = []
    if len(exact) != len(hnsw):
        problems.append(f"length {len(hnsw)} from HNSW, {len(exact)} exact")
    for position, ((_, want), (_, got, _)) in enumerate(zip(exact, hnsw, strict=False), start=1):
        if abs(want - got) > tolerance:
            problems.append(f"position {position}: cosine {got:.6f} from HNSW, {want:.6f} exact")
    if exact and hnsw and len(exact) == len(hnsw):
        cut = exact[-1][1]
        strict_exact = {i for i, c in exact if c > cut + tolerance}
        strict_hnsw = {i for i, c, _ in hnsw if c > cut + tolerance}
        if strict_exact != strict_hnsw:
            problems.append(f"{len(strict_exact ^ strict_hnsw)} node IDs differ above the cut-off")
    return problems
