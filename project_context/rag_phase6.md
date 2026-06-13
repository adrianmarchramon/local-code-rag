# Phase 6 — The API and the Interface

> In this phase, you convert the engine into a product. Up to this point, your RAG assistant works, but only from the command line; from here on, anyone will be able to use it from a chat screen. There are two complementary pieces: the **API**, which turns the system into a service that can receive requests, and the **chat interface**, which gives it a face to converse with. And here, as in other phases, the benefit of having previously completed the MLOps project becomes clear: the API directly reuses the FastAPI best practices you already master, allowing you to set it up in a fraction of the time and with consistency between your two projects.

**Phase objective:** Convert the system into a usable product.  
**Duration:** Week 6.  
**Upon completion, you will have:** A FastAPI API that serves the assistant (with endpoints to index a repository and to ask questions), and a functional chat interface to converse with it and see citations to the files for each answer.

---

## The Big Picture

The architecture of this phase clearly separates the brain (the API) from the face (the interface):

```
     User
      │
      ▼
Chat Interface (Streamlit)    ──► writes questions, views answers and sources
      │ (HTTP calls)
      ▼
  API (FastAPI)
      ├── POST /ask     → executes the RAG pipeline and returns answer + sources
      ├── POST /index   → (re)indexes a repository
      └── GET  /health  → checks status
      │
      ▼
 RAG Pipeline (Phase 5)       ──► retrieval + generation
      │
      ▼
Index + nodes (loaded once at startup)
```

We will use FastAPI for the API and Streamlit for the chat interface. Add the dependencies:

```bash
uv add fastapi "uvicorn[standard]" streamlit httpx
```

And add the configuration for this phase to `src/config.py`:

```python
API_HOST = "0.0.0.0"
API_PORT = 8000
API_URL = "http://localhost:8000"   # for the interface to call the API
```

---

## Step 1 — The Architecture: Separated API and Interface

Before diving into coding, it is worth understanding why we separate the API from the interface, as this is a design decision that demonstrates good engineering judgment. You could have the interface call the pipeline directly, but separating the backend (the API) from the frontend (the interface) has clear advantages, and it is how real-world systems are built.

The **separation of concerns** is the main reason: the API is the brain (it receives a question, executes the RAG pipeline, and returns the response), and the interface is just the face (it collects the user's question and displays the response beautifully). Each does one thing. Furthermore, the API is **reusable**: once the assistant is a service with an API, any client can use it (the chat interface, an editor extension, a script), not just the UI you build. The interface is simply one of the potential consumers. This architecture, identical in spirit to the one you set up in the MLOps project (where the API served the model), is the professional way to do it, and knowing how to explain why you chose it scores extra points.

---

## Step 2 — Loading the Index Once: The Lifespan

There is a crucial performance point here, and it is exactly the same lesson you learned in Phase 4 of the MLOps project. The index and nodes are expensive to load (nodes must be derived from the repository, and the index must be connected). It would be catastrophic for performance to do this on every request: each question would take an eternity. The solution is to load them **only once, when starting the API**, and reuse them across all requests. FastAPI offers the `lifespan` mechanism for this, the same one you used to load the model at startup in MLOps:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import REPO_PATH
from src.indexing.build_index import load_index
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_repository
from src.settings import configure_local_models

# Shared state between requests: the index and the nodes
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # At startup: configure local models and load the index only once
    configure_local_models()
    state["nodes"] = chunk_documents(load_repository(REPO_PATH))  # for BM25
    state["index"] = load_index()  # reconnects to the persisted index (fast, does not re-embed)
    yield
    # At shutdown: clean up
    state.clear()
```

It is worth understanding what happens here. At API startup, the nodes (derived from the repository, which the BM25 retriever needs) are loaded, and it reconnects to the vector index already persisted in ChromaDB. Note that we use `load_index` (which reconnects to what was already built, instantaneously), not `build_index` (which would re-vectorize everything): this is where the persistence you set up in Phase 3 pays off. This does require, however, that you have indexed the repository at least once (you will do this via the `/index` endpoint or by running the indexing once). From then on, all requests reuse the index already loaded in memory, responding quickly.

---

## Step 3 — The Pydantic Schemas

Just like in the MLOps project, we define input and output contracts using Pydantic, which gives us automatic validation, documentation, and type safety. Complete `src/api/schemas.py`:

```python
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the code")
    language: str | None = Field(None, description="Filter by language (optional)")


class Source(BaseModel):
    file_path: str | None = None
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IndexRequest(BaseModel):
    repo_path: str = Field(..., description="Path to the repository to index")


class IndexResponse(BaseModel):
    indexed_files: int
    chunks: int
    message: str
```

These schemas describe exactly what each endpoint receives and returns. `AskRequest` contains the question (mandatory) and an optional language filter; `AskResponse` returns the response and the list of sources; `Source` structures each citation (file, symbol, lines, score). Defining this with Pydantic ensures that FastAPI automatically validates requests, generates interactive documentation, and keeps the service contract crystal clear, just as in your previous project.

---

## Step 4 — The API with FastAPI

Now, the endpoints. Complete `src/api/main.py` by building upon the lifespan base we already wrote:

```python
from pathlib import Path

from fastapi import FastAPI

from src.api.schemas import (
    AskRequest,
    AskResponse,
    IndexRequest,
    IndexResponse,
    Source,
)
from src.generation.answer import answer_question
from src.indexing.build_index import build_index

# ... (the lifespan block and 'state' from Step 2 go here) ...

app = FastAPI(title="RAG Code Assistant (local)", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": "index" in state}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = answer_question(
        state["index"], state["nodes"], request.question, language=request.language
    )
    return AskResponse(
        answer=result.text,
        sources=[Source(**s) for s in result.sources],
    )


@app.post("/index", response_model=IndexResponse)
def index_repo(request: IndexRequest) -> IndexResponse:
    documents = load_repository(Path(request.repo_path))
    nodes = chunk_documents(documents)
    state["nodes"] = nodes
    state["index"] = build_index(Path(request.repo_path))
    return IndexResponse(
        indexed_files=len(documents),
        chunks=len(nodes),
        message="Repository indexed successfully",
    )
```

The three endpoints cover the essentials. `/ask` is the heart: it receives a question, executes the full Phase 5 RAG pipeline (using the already-loaded index and nodes), and returns the answer with its sources. `/index` allows on-demand indexing (or re-indexing) of a repository, running the indexing pipeline and updating the state. Finally, `/health`, like in MLOps, checks that the service is healthy and the index is loaded. Start the API with `uv run uvicorn src.api.main:app --reload` and, as a gift from FastAPI, you will have interactive documentation at `http://localhost:8000/docs`, where you can test the endpoints directly from your browser.

---

## Step 5 — The Chat Interface

Now for the face of the system. Streamlit allows you to build a chat interface with very little code, thanks to its specific chat-oriented components. Complete `ui/app.py`:

```python
import httpx
import streamlit as st

from src.config import API_URL

st.set_page_config(page_title="RAG Code Assistant", page_icon="🤖")
st.title("Code Assistant (Local RAG)")


def render_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(
                    f"- `{s['file_path']}` **{s.get('symbol') or ''}** "
                    f"(lines {s.get('start_line')}–{s.get('end_line')})"
                )


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

# User input
if pregunta := st.chat_input("Ask something about the code..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Searching the code and generating the response..."):
            resp = httpx.post(
                f"{API_URL}/ask", json={"question": pregunta}, timeout=300
            )
            data = resp.json()
        st.markdown(data["answer"])
        render_sources(data["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
    )
```

It is worth understanding the Streamlit building blocks. `st.session_state` maintains the conversation history across interactions so the chat remembers what has been said. `st.chat_input` is the text box at the bottom where the user types, and `st.chat_message` displays each message according to its role (user or assistant), giving it a conversational look. When the user asks a question, the interface calls the `/ask` API endpoint (with a generous timeout because the local model takes time to generate), displays the response, and lists the sources underneath. The `st.spinner` shows a loading indicator while the response is being generated, which improves the user experience by giving feedback that the system is working. Start the interface with `uv run streamlit run ui/app.py`.

---

## Step 6 — Showing Sources: Transparency

It is worth pausing to consider the **citations in the interface**, as this is where the entire chain of transparency you have been building culminates. As we have seen, transparency is key in RAG: an answer without sources forces the user to blindly trust the assistant; an answer with sources allows them to verify it. In the interface, this translates to showing, beneath each response, exactly which code snippets support it: the file, the symbol, and the lines.

This is where the metadata you carefully preserved since Phase 1 pays off: the `file_path` you included during ingestion, and the `symbol` and line range you added during chunking. All of this traveled through the embeddings, retrieval, and generation steps, and now appears to the user as precise, verifiable citations. This complete traceability, from the source code to the citation on screen, is one of the features that sets a well-built RAG apart. When you show it in the demo, demonstrating that each response is accompanied by its sources immediately communicates that your system is transparent and reliable, rather than a black box spitting out answers.

---

## Step 7 — Testing the Complete Application

With the API and the interface running (each in its own terminal), open the interface in your browser and test the complete assistant. Type a question about the indexed repository and watch how the response appears in the chat, grounded in the code, with expandable sources listed below. For the first time, you have the entire system working as a product: a smooth chat experience to converse with your code assistant.

Test the experience as a user would: ask several consecutive questions (the chat maintains history), ask explanatory questions, request code generation, and expand the sources to verify that each response cites the correct code. If the conversation flows and responses are returned with their citations, you have met the phase's goal. This is also the perfect format for the demonstration video you will record in Phase 8.

---

## Verification: The Definition of "Done"

The phase is complete when the following conditions are met:

- [ ] The FastAPI API exposes endpoints for indexing (`/index`) and asking (`/ask`), utilizing Pydantic schemas.
- [ ] The index and nodes are loaded only once at startup (lifespan), not on every request.
- [ ] The interactive documentation (`/docs`) is available.
- [ ] The chat interface allows you to converse with the assistant.
- [ ] Each response displays its sources (file, symbol, lines).
- [ ] **The key test:** You open the interface, write a question about the indexed repository, and get a response with its sources, in a smooth chat experience.

The key test lies in the end-to-end experience: if you open the interface, ask a question, and receive a grounded response with its sources in a fluid chat interface, you have turned your RAG engine into a real product. That transition—from "it works in the terminal" to "anyone can use it from a screen"—is what makes the project shift from a simple experiment into something you can demonstrate and use.

---

## Deliverables and What's Next

By closing Phase 6, you have turned the system into a usable product: an API serving the assistant as a service and a chat interface where anyone can converse with it and view the citations for each response. You have successfully reused FastAPI best practices from your previous project (using the lifespan to load the index once, Pydantic schemas, the health endpoint, automatic documentation), adding cohesion to your portfolio, and you have completed the chain of transparency by showing the sources to the user.

The next step, **Phase 7**, is what elevates this project above average: **evaluation with RAGAS**. Up until now, you have evaluated the system's quality "by eye," testing questions and checking if the answers seem good. However, a rigorous project needs to measure its quality objectively. You will build an evaluation dataset and measure metrics like faithfulness (does the answer actually rely on the context?), relevance, and the quality of the retrieved context, using these metrics to improve the system in a measurable way. You have moved from "the assistant is a usable product" to "I can prove, with numbers, how good it is"—applying the exact same rigor you brought to models in your MLOps project.
