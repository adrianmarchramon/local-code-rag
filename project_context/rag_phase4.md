# Phase 4 — Retrieval: Hybrid Search and Reranking

> This phase refines the **core of the system**. Remember the idea we repeated from the fundamentals: in RAG, quality is determined by retrieval. The similarity search you introduced in Phase 3 is a good starting point, but it has significant limitations, especially for code. Here, you will turn it into a highly effective retrieval process using two techniques that robust RAG solutions incorporate to improve accuracy: **hybrid search** (combining semantic search with keyword search) and **reranking** (reordering candidates using a more precise model). This is likely the phase where you gain the most quality per unit of effort, because improving retrieval improves everything that follows.

**Phase objective:** Retrieve the correct snippets for each question.
**Duration:** Week 4.
**By the end, you will have:** A retrieval system that combines semantic and lexical search, reorders results with a reranker, allows metadata filtering, and returns, for any given question, the actual code snippets that answer it, in the correct order.

---

## The Big Picture

Enhanced retrieval consists of several stages that, combined, retrieve much better than simple search:

```
   User query
        │
        ├──────────────────────┬───────────────────────┐
        ▼                      ▼                         │
   [ Vector Search ]      [ BM25 Search ]         (optional metadata
      (semantic)            (keywords)                 filter)
        │                      │
        └──────────┬───────────┘
                   ▼
        [ RRF Fusion ]  ──►  combines both lists into a single ranking
                   │
                   ▼
        [ Reranking (cross-encoder) ]  ──►  reorders by actual relevance
                   │
                   ▼
        The correct snippets, in the correct order → (feeds Phase 5 generation)
```

We will use LlamaIndex's `BM25Retriever` for lexical search, `QueryFusionRetriever` to combine both searches with *Reciprocal Rank Fusion*, and a cross-encoder reranker (a BGE model) for the final reordering. Add the dependencies:

```bash
uv add llama-index-retrievers-bm25 llama-index-llms-ollama sentence-transformers
```

And, very importantly, configure LlamaIndex to use your **local models** by default. This avoids a classic issue that catches many people off guard: by default, LlamaIndex attempts to use OpenAI for the LLM and embeddings, and without an API key, some components will fail to instantiate. By configuring the models globally, everything points locally. Create `src/settings.py`:

```python
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.config import CODE_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL


def configure_local_models() -> None:
    """LlamaIndex will use local models by default, not OpenAI."""
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
    )
    Settings.llm = Ollama(
        model=CODE_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0
    )
```

Also add the configuration for this phase to `src/config.py` (the `CODE_MODEL` is the one you downloaded in Phase 0):

```python
CODE_MODEL = "qwen3-coder:30b"   # or "qwen2.5-coder:7b" depending on your hardware
RETRIEVE_TOP_K = 20              # candidates per retriever before reranking
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 5                 # final snippets after reranking
```

---

## Step 1 — The Limits of Purely Semantic Search

To understand why we need more than the vector search from Phase 3, it helps to see where it fails, as this is precisely where code has a unique characteristic. Semantic search is excellent for conceptual questions (e.g., "how are payments managed?") because it captures meaning. However, it has a blind spot: **exact identifiers**.

Imagine looking for a specific function by its name: `validate_token`. This query has very little semantic "meaning" (it is basically a name), so the vector search might not hit the mark: the vector of `validate_token` as a query is not necessarily positioned close to the vector of the function, because there is very little conceptual content to vectorize. In contrast, a **keyword** search would find the exact match of the name immediately.

And here lies the key: **code is full of exact identifiers** that matter a great deal (names of functions, variables, classes, methods, parameters). You will constantly want to find "where is `parse_config` defined" or "where is `MAX_RETRIES` used", and those are lexical, exact-match searches, not semantic ones. A retrieval system that relies solely on vector search will consistently fail on these types of queries, which are common in code repositories. Because of this, hybrid search is not a luxury for a code RAG—it is a necessity.

---

## Step 2 — Hybrid Search: Semantic Plus Lexical

The solution is to combine two complementary search approaches. **Dense search** (vector-based, semantic) captures meaning and is robust to paraphrasing; **sparse search** (lexical, using the BM25 algorithm) captures exact term matches. Each retrieves what the other misses: dense search finds conceptual queries, while sparse search finds exact identifiers. Together, they cover both types of queries.

We build both retrievers. You already have the vector retriever (from Phase 3); the BM25 retriever is built directly from the snippets:

```python
from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=RETRIEVE_TOP_K)
```

Here is a practical detail worth knowing, because it demonstrates that you understand how the pieces fit together: the BM25Retriever requires the **nodes** (the snippets) to build its lexical index. Since we are using ChromaDB—which stores nodes in its own database rather than in LlamaIndex's internal document store—we build the BM25 retriever by passing the nodes directly (`nodes=nodes`), instead of passing them through the index. You already have these nodes from the chunking phase. (For queries in subsequent runs where the index is already persisted, you can retrieve the nodes again by running the ingestion pipeline, which is deterministic, or you can persist the BM25 index itself, which it also supports.)

Now comes the interesting part: how to **combine** the two lists of results returned by the retrievers. The standard technique is **Reciprocal Rank Fusion (RRF)**, and it is worth understanding because it is elegant and robust. The idea is: for each snippet, we look at the position (rank) where it appears in each list, and we give it a combined score by summing the reciprocal of its ranks, using the formula `1 / (k + rank)` (with a constant `k`, typically 60). Then, all snippets are re-sorted by this combined score. The beauty of RRF is that it uses the *positions* rather than the raw scores from each retriever (which are not directly comparable, since a vector similarity score and a BM25 score are on completely different scales). Consequently, a snippet that appears near the top in both lists climbs significantly, combining the best of both worlds without needing an additional model.

LlamaIndex implements this using the `QueryFusionRetriever`:

```python
from llama_index.core.retrievers import QueryFusionRetriever

fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=RETRIEVE_TOP_K,
    num_queries=1,             # 1 = no query expansion (no extra LLM call)
    mode="reciprocal_rerank",  # RRF fusion
    use_async=False,
)
```

The `mode="reciprocal_rerank"` is what applies RRF. Setting `num_queries=1` disables *query expansion* (an optional technique where the LLM generates variants of the question to improve recall). We keep it disabled for now so that retrieval is fast and independent of the LLM, though you could enable it later by increasing this number.

---

## Step 3 — Reranking: The Second Precision Filter

Hybrid fusion gives us good **recall**: it is highly likely that the correct snippets are among the, say, 20 returned candidates. However, the *ordering* of those 20 can still be improved—and order matters, because we will only pass the very best ones to the model. This is where **reranking** comes in, acting as a second filter focused on **precision**.

To understand why reranking helps, it is useful to know an important distinction between two types of models. The embedding model we use for searching is a **bi-encoder**: it encodes the query and each snippet separately into independent vectors and compares them. This is fast and scalable (you can precalculate the vectors for the entire repository) but less precise because it never looks at the query and the snippet *together*. A **reranker** is a **cross-encoder**: it takes the query and a snippet at the same time, analyzes them jointly, and produces a much finer relevance score. It is far more precise but also slower, which is why you cannot run it across the entire repository.

This gives rise to the **two-stage retrieval** pattern, which is fundamental in production-grade RAG: first, a broad and inexpensive retrieval step (hybrid fusion) casts a wide net to fetch the top 20 candidates, optimizing recall; second, a computationally expensive and precise reranking step (the cross-encoder) is applied only to those 20 candidates to reorder them, keeping the top 5, optimizing precision. You only use the resource-intensive model where it yields the best return: on the short list. We will use a BGE reranker:

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)
```

This reranker is downloaded once and **runs locally** (on your machine, via sentence-transformers), keeping our privacy promise: just like the LLM and the embeddings, the reordering occurs without sending any data externally. An observation that highlights its value for code: cross-encoder reranking is particularly useful in domains like code, where exact phrasing matters and generic embeddings sometimes fall short. This is precisely where it closes the gap the most.

---

## Step 4 — Metadata Filtering

Sometimes you do not want to search the entire repository, but rather narrow down the search: only within Python files, or only within a specific directory. This is what **metadata filters** are for, leveraging the metadata (`language`, `file_path`) you have kept since the ingestion phase. This is another area where those metadata fields show their value:

```python
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

filters = MetadataFilters(
    filters=[MetadataFilter(key="language", value="python", operator=FilterOperator.EQ)]
)
vector_retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K, filters=filters)
```

With this, the vector search is restricted to snippets that match the filter (in this example, only Python). This is useful in multi-language repositories (e.g., searching only within the Python backend while ignoring the JavaScript frontend) or to narrow things down to a specific section of the codebase. It is a simple yet powerful capability that once again demonstrates why investing in solid metadata from the beginning was worth the effort.

---

## Step 5 — The Complete Retriever

Now, you combine everything into `src/retrieval/retriever.py`: building the hybrid retriever (with an optional filter), retrieving the candidates, and reordering them with the reranker.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

from src.config import RERANK_MODEL, RERANK_TOP_N, RETRIEVE_TOP_K


def build_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    language: str | None = None,
) -> QueryFusionRetriever:
    """Hybrid retriever (vector + BM25) with RRF fusion and optional filtering."""
    filters = None
    if language is not None:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)]
        )

    vector_retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K, filters=filters)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=RETRIEVE_TOP_K)

    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=RETRIEVE_TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def get_reranker() -> SentenceTransformerRerank:
    """Cross-encoder reranker (BGE) that runs locally."""
    return SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)


def retrieve(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    query: str,
    language: str | None = None,
) -> list[NodeWithScore]:
    """Retrieves and reorders the most relevant snippets for a query."""
    retriever = build_hybrid_retriever(index, nodes, language=language)
    candidates = retriever.retrieve(query)
    reranked = get_reranker().postprocess_nodes(candidates, query_str=query)
    return reranked


if __name__ == "__main__":
    from src.config import REPO_PATH
    from src.indexing.build_index import build_index
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository
    from src.settings import configure_local_models

    configure_local_models()
    nodes = chunk_documents(load_repository(REPO_PATH))
    index = build_index(REPO_PATH)

    results = retrieve(index, nodes, "how is input data validated?")
    for r in results:
        m = r.metadata
        print(f"\n--- {m.get('file_path')} [{m.get('symbol', '?')}] "
              f"(score {r.score:.3f}) ---")
        print(r.get_content()[:250])
```

The workflow is straightforward: the hybrid retriever is built, the top 20 candidates are retrieved by combining semantic and lexical searches, and the reranker reorders them to keep the top 5 most relevant. Those 5 are what we will pass to the model as context in the next phase.

---

## Step 6 — Testing Retrieval

Run the retriever (`uv run python -m src.retrieval.retriever`) and observe the results. You will see the most relevant snippets for the question, now better selected and better ordered than with the simple search in Phase 3. To truly appreciate the improvement, try two types of queries. With a **conceptual** question (e.g., "how are connection errors handled?"), you will verify that the semantic side finds the relevant code even if you do not use the exact words. With an **exact identifier** query (the name of a specific function in your repository), you will see that the BM25 side finds it instantly, which is something vector search alone would fail to do. Seeing this combination work is proof of the value that hybrid search adds. And when you see the correct snippets appearing at the very top, you can verify the impact of the reranking step.

---

## Step 7 — Tests

We verify that retrieval works for both types of queries. Since it requires local models (embeddings and reranker), this is an integration test that is skipped if Ollama is not available. Complete `tests/test_retrieval.py`:

```python
import pytest

from src.indexing.embeddings import get_embedding_model


def _ollama_available() -> bool:
    try:
        get_embedding_model().get_text_embedding("test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Requires Ollama running")
def test_hybrid_retrieval_finds_exact_identifier(tmp_path):
    from src.indexing.build_index import build_index
    from src.indexing.vector_store import get_vector_store
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository
    from src.retrieval.retriever import retrieve
    from src.settings import configure_local_models

    configure_local_models()
    (tmp_path / "auth.py").write_text(
        "def validate_token(token):\n"
        "    '''Verifies a session token.'''\n"
        "    return len(token) == 32\n"
    )
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")

    nodes = chunk_documents(load_repository(tmp_path))
    store = get_vector_store(persist_dir=tmp_path / "vs", collection="test")
    index = build_index(tmp_path, vector_store=store)

    # Query for the exact identifier: BM25 should find it
    results = retrieve(index, nodes, "validate_token")
    assert results[0].metadata["file_path"] == "auth.py"
```

This test verifies the contribution of hybrid search: when querying for the exact name `validate_token`, the most relevant snippet is the one containing the correct function, thanks to the lexical component. This is precisely the case where purely semantic search would struggle. Run it with `make test`.

---

## Verification: Definition of Done

The phase is complete when the following conditions are met:

- [ ] `retriever.py` performs vector (semantic) search on the database.
- [ ] Hybrid search is implemented, combining vector search with BM25 via RRF fusion.
- [ ] A cross-encoder reranker (BGE) reorders candidates by relevance and runs locally.
- [ ] Filtering by metadata (language, file) is supported.
- [ ] Tests verify that retrieval successfully finds the relevant code.
- [ ] **The key test:** when presented with a specific question about the code, the retrieval process returns the actual snippets that answer it, in the correct order.

The key test centers on relevance and order: if, when faced with both conceptual and exact-identifier queries, your system returns the snippets that truly answer them and places them at the top, you have built a quality retrieval system. Since we know that in RAG retrieval is the most critical element, reaching this point means you have resolved the component that most heavily determines the overall system quality.

---

## Deliverables and Next Steps

By completing Phase 4, you have refined the core of the system: a retrieval process that combines semantic and lexical search, merges them using RRF, reorders them with a cross-encoder reranker, and allows metadata filtering—all locally. You have understood and applied the techniques that truly distinguish a quality RAG system (the complementary nature of dense and sparse searches, the two-stage retrieval pattern), and you have resolved the component that plays the largest role in the final quality.

The next step, **Phase 5**, finally closes the question-to-answer loop: **generation with the local LLM**. You will take the high-quality snippets you now retrieve, combine them with the query in a prompt, and pass them to Qwen3-Coder via Ollama so it can generate an answer grounded in that code, complete with citations to the source files. This is where the meticulous care taken during retrieval pays off: because you provide the model with the correct context, its response will be accurate. You have moved from "retrieving the correct snippets" to being on the verge of "answering questions about the code, grounded in the codebase."
