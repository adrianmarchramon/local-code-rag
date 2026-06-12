# Phase 3 — Embeddings and the Vector Database

> In this phase, the process that makes searching by meaning possible takes place. Until now, you have the code split into coherent chunks, but you cannot *search* them by meaning yet. Here, you convert each chunk into an **embedding** (a vector that captures its meaning) and store it in a **vector database** optimized to find, given a query vector, the most similar ones. With this, your repository stops being just a collection of files and becomes a **knowledge base searchable by meaning**. And here, you will finally see the reward of having chunked the code properly in the previous phase: since each chunk is a meaningful unit, its embedding will accurately capture what it does, and the search will truly work.

**Phase objective:** Vectorize the chunks and store them for searching.  
**Duration:** Week 3.  
**Upon completion, you will have:** The complete indexing pipeline (repo → ingestion → chunking → embeddings → storage), a persisted repository index in the local vector store, and the ability to perform similarity searches that return relevant code snippets.

---

## The Big Picture

This phase completes the **indexing** half of the system, assembling all the pieces you have built so far and adding two new ones (embeddings and storage):

```
   Repository
      │
      ▼  (Phase 1)
   Ingestion  ──►  clean code documents
      │
      ▼  (Phase 2)
    Chunking  ──►  coherent chunks with metadata
      │
      ▼  (Phase 3, new)
   Embeddings ──►  each chunk is converted into a vector (local model)
      │
      ▼  (Phase 3, new)
  Vector Store (ChromaDB) ──► stores vectors + metadata, persisted on disk
      │
      ▼
Searchable index by meaning → (feeds the retrieval in Phase 4)
```

We will use a **local embedding model** served by Ollama (preserving privacy) and **ChromaDB** as the vector store, all orchestrated with LlamaIndex. Add the dependencies (you already installed LlamaIndex core in Phase 1):

```bash
uv add llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb
```

And add the configuration for this phase to your `src/config.py`:

```python
from pathlib import Path

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
VECTOR_STORE_DIR = Path("data/vector_store")
COLLECTION_NAME = "code"
```

---

## Step 1 — What is an embedding and why does it make semantic search possible?

We saw the concept in the fundamentals, but it is worth making it concrete, as it is what gives meaning to this entire phase. An **embedding** is the conversion of text (here, a code snippet) into a **vector**: a long list of numbers. The remarkable part is that this vector captures the *meaning* of the text, so that two fragments doing similar things will have similar vectors (close in space), even if they use different words. That is, meaning is converted into *geometry*: the proximity between vectors reflects the proximity of meaning.

This is exactly what makes **semantic search** possible, which is far superior to keyword search for many questions. Imagine someone asks, "how do I authenticate?". A keyword search would literally look for the word "authenticate" and would fail to find a function named `validate_token` because it does not contain that word. However, with embeddings, the vector of the query and the vector of that function will be close (because they *mean* related things), so the search will find it. The embedding captures intent, not just words.

To search, the process is symmetrical: the user's query is converted into a vector using the same model, and the system looks for the fragments whose vectors are closest to the query's vector. Those are the most relevant by meaning. This entire phase is about building the infrastructure to make that work.

---

## Step 2 — The local embedding model

The first step is to configure the model that converts text into vectors, and, keeping the project's promise, we do this locally using Ollama. Complete `src/indexing/embeddings.py`:

```python
from llama_index.embeddings.ollama import OllamaEmbedding

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_embedding_model() -> OllamaEmbedding:
    """Local embedding model, served by Ollama."""
    return OllamaEmbedding(
        model_name=EMBEDDING_MODEL,   # "nomic-embed-text"
        base_url=OLLAMA_BASE_URL,     # "http://localhost:11434"
    )
```

This creates an embedding model that uses the `nomic-embed-text` model you downloaded in Phase 0, through Ollama. Since Ollama runs on your machine, each code snippet is vectorized locally, without sending anything to an external server. This keeps the promise of total privacy intact even during this step (which is exactly where many systems break it by sending text to a cloud embedding API).

There is a **critical** point that should be established right now: you must use the **same embedding model** for both indexing and querying. The reason is that search compares the query vector with the chunk vectors, and this comparison only makes sense if they all inhabit the same "vector space"—meaning they were generated by the same model. Using one model for indexing and another for querying would be like comparing apples to oranges: the vectors would not be comparable and the search would fail. That is why we centralize the model in a single function used on both sides.

---

## Step 3 — The vector store

Once you have the vectors, you need a place to store them and, above all, a way to search them efficiently. That is the job of the **vector database**, which specializes in one operation: given a query vector, quickly find the most similar vectors among potentially millions. We will use ChromaDB. Complete `src/indexing/vector_store.py`:

```python
from pathlib import Path

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import COLLECTION_NAME, VECTOR_STORE_DIR


def get_vector_store(
    persist_dir: Path = VECTOR_STORE_DIR,
    collection: str = COLLECTION_NAME,
) -> ChromaVectorStore:
    """Persistent disk-based ChromaDB vector store."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = client.get_or_create_collection(collection)
    return ChromaVectorStore(chroma_collection=chroma_collection)
```

The key piece here is the `PersistentClient`: unlike an in-memory client (which would lose everything when the program closes), the persistent client **saves data to disk** in the directory you specify. This is what allows your index to survive between runs, as we will see in the persistence step. Along with each vector, the database stores its **metadata** (the `file_path`, `symbol`, and the line range you set in previous phases), which travel with the vector and will form the basis for citations and filtering in subsequent phases.

---

## Step 4 — The complete indexing pipeline

Now you assemble all the pieces of the system into a single pipeline running from the repository to the index. This is where everything built so far comes together: Phase 1 ingestion, Phase 2 chunking, and the embeddings and storage of this phase. Complete `src/indexing/build_index.py`:

```python
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex

from src.indexing.embeddings import get_embedding_model
from src.indexing.vector_store import get_vector_store
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_repository


def build_index(
    repo_path: Path,
    vector_store=None,
    embed_model=None,
) -> VectorStoreIndex:
    """Complete pipeline: repo → ingestion → chunking → embeddings → storage."""
    # 1. Ingestion (Phase 1) and chunking (Phase 2)
    documents = load_repository(repo_path)
    nodes = chunk_documents(documents)

    # 2. Embeddings and storage (Phase 3)
    vector_store = vector_store or get_vector_store()
    embed_model = embed_model or get_embedding_model()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Build the index: vectorizes each chunk and stores it
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index
```

It is helpful to understand what `VectorStoreIndex` does, as this is where the heavy lifting happens. When building it from the chunks (`nodes`), LlamaIndex takes each chunk, passes it through the embedding model to get its vector, and stores it in the vector database alongside its text and metadata. Setting `show_progress=True` displays a progress bar, which is useful because vectorizing an entire repository with a local model can take some time (since you are generating hundreds or thousands of embeddings on your machine). Once finished, the vector database contains your entire repository converted into searchable vectors. The optional `vector_store` and `embed_model` parameters allow you to inject alternative versions, which is useful for testing.

---

## Step 5 — Persistence: index once, query many times

Indexing is the slow and resource-intensive part of the system: vectorizing all the chunks takes time. It would be impractical to rebuild it every time you start the application. Therefore, it is essential to **persist** the index: build it once and reuse it later. This is one of the most important mistakes to avoid (reindexing everything on every run), and solving it properly demonstrates good engineering judgment.

There is an elegant detail of ChromaDB worth noting: because we use its `PersistentClient`, **persistence is automatic**. The vector store saves the vectors and their text to its disk directory as you insert them, so once the index is built, it is already persisted without requiring any further actions. This differs from LlamaIndex's default in-memory store (which would require an explicit call to save); with a real vector database like Chroma, the data already resides there.

To **reload** the index in subsequent runs (which you will do when querying in the following phases), you simply reconnect to the existing Chroma collection and rebuild the index interface on top of it, without re-vectorizing anything:

```python
from llama_index.core import VectorStoreIndex

from src.indexing.embeddings import get_embedding_model
from src.indexing.vector_store import get_vector_store


def load_index() -> VectorStoreIndex:
    """Loads the pre-built index, reconnecting to the vector store."""
    vector_store = get_vector_store()          # reconnects to the persisted collection
    embed_model = get_embedding_model()        # the same model, for querying
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```

The difference between `build_index` and `load_index` is the key to persistence: the former vectorizes and stores (slow, done once); the latter connects to what is already stored (instant, done every time you query). Additionally, notice that `load_index` uses the same embedding model for the consistency reasons we discussed: queries must be vectorized with the same model as the chunks.

---

## Step 6 — The first similarity search

Now it is time to check that everything works, which serves as the success criterion for this phase. Build the index and perform a similarity search: give it a query and see if it returns relevant code snippets. Add this to the end of `build_index.py`:

```python
if __name__ == "__main__":
    from src.config import REPO_PATH

    index = build_index(REPO_PATH)

    # Similarity search: the most relevant chunks for a question
    retriever = index.as_retriever(similarity_top_k=3)
    results = retriever.retrieve("how are input data validated?")

    for r in results:
        m = r.metadata
        print(f"\n--- {m.get('file_path')} [{m.get('symbol', '?')}] "
              f"(score {r.score:.3f}) ---")
        print(r.get_content()[:250])
```

Upon execution (`uv run python -m src.indexing.build_index`), the system will index the repository and then search for the three most relevant chunks for the question. If everything is correct, you will see code snippets that genuinely relate to validating data, each with its source file, symbol, and relevance score. **This is the moment you see the reward of all the previous steps:** because you split the code into coherent units, their embeddings accurately capture their meaning, and the search returns entire, pertinent functions, rather than meaningless pieces. If the chunking had been poor, you would see confusing results here; seeing good results validates the entire chain.

---

## Step 7 — Tests

Since this phase requires local models to actually vectorize, the test is an integration test: it indexes a minimal repository and checks that the search returns the expected results, skipping the test if Ollama is not available. Complete `tests/test_indexing.py`:

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
def test_index_and_search(tmp_path):
    # Minimal repo: one validation file and one math file
    (tmp_path / "validation.py").write_text(
        "def validate_input(data):\n"
        "    '''Checks that the input data are valid.'''\n"
        "    if not data:\n"
        "        raise ValueError('empty')\n"
        "    return True\n"
    )
    (tmp_path / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")

        from src.indexing.build_index import build_index
        from src.indexing.vector_store import get_vector_store

        # Isolated vector store in a temporary directory
        store = get_vector_store(persist_dir=tmp_path / "vs", collection="test")
        index = build_index(tmp_path, vector_store=store)

        results = index.as_retriever(similarity_top_k=1).retrieve(
            "how are input data validated"
        )
        assert len(results) >= 1
        # The most relevant chunk must come from the validation file
        assert results[0].metadata["file_path"] == "validation.py"
```

This test indexes a miniature repository (isolated in a temporary directory to avoid touching your actual index) and verifies the essentials: that when searching for "how are input data validated," the most relevant snippet is from the validation function, and not the addition function. That is, it verifies that semantic search works: it finds the relevant code by its meaning. It is skipped automatically if Ollama is not running. Run it with `make test`.

---

## Verification: The Definition of "Done"

The phase is complete when the following are met:

- [ ] `embeddings.py` configures the local embedding model (Ollama), used for both indexing and querying.
- [ ] `vector_store.py` configures ChromaDB with disk persistence.
- [ ] The complete indexing pipeline (repo → ingestion → chunking → embeddings → storage) works end-to-end.
- [ ] The index is persisted and can be reloaded without reindexing.
- [ ] The tests verify that the search returns relevant code.
- [ ] **The key test:** You index a repo and a similarity search returns relevant code snippets.

The key test is the search: if, after indexing a repository, you ask a question and the system returns code snippets that genuinely relate to it, you have built a system searchable by meaning. This is the foundation upon which everything else will be built: without a functioning search, there is no RAG; with it, you already have the heart of the system beating.

---

## Deliverables and what comes next

By completing Phase 3, you have finished the **indexing half** of the system: a pipeline that converts any repository into a persisted, vector index searchable by meaning, entirely locally. You have assembled the work of the three phases (ingestion, chunking, embeddings, and storage) into an end-to-end chain, and you have seen the reward of proper chunking: the search returns relevant code. You have also resolved persistence, enabling you to index once and query many times.

The next step, **Phase 4**, refines the heart of the system: the **retrieval**. The similarity search you just launched is a solid starting point, but it has limitations, especially for code. You will enhance it using two techniques that significantly increase precision: **hybrid search** (combining semantic search with keyword search, which is crucial for exact code identifiers) and **reranking** (reordering results with a more precise model). You have gone from "the repository is searchable by meaning" to being on the verge of "retrieving exactly the correct snippets for each query," which is what truly determines the quality of the RAG.
