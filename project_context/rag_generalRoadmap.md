# Roadmap: Local LLM with RAG for Code

> A code assistant that runs 100% on your machine, without a single line of your code leaving for the internet. It indexes entire repositories, understands their structure, and answers questions and generates code in a *project-aware* manner, relying on what is actually in them. It demonstrates mastery of today's most in-demand AI architectures (RAG), local LLM deployment, and the engineering required to turn it into a real product.

**Estimated Duration:** 6-8 weeks (part-time)  
**Level:** Intermediate  
**Final Outcome:** A local, private RAG system that, given any repository, allows you to ask questions like "how does authentication work in this project?" or "write a test for this function following the project's style" and obtain grounded answers based on the actual code, complete with citations to the source files and functions.

---

## 1. What Exactly You Will Build

A **Retrieval-Augmented Generation (RAG)** system specialized in code, operating in two stages. It is best to view it this way because it has a preparation phase (indexing) and a usage phase (querying), running like two clocks at different speeds:

```
== INDEXING (once per repository) ==

   Code repository (GitHub or local)
        │
        ▼
   [ Ingestion ]  ──►  load and filter files (ignore binaries, builds, vendored)
        │
        ▼
   [ Code-aware Chunking ]  ──►  split by functions/classes, not blindly
        │
        ▼
   [ Embeddings (local model) ]  ──►  vectorize each chunk
        │
        ▼
   [ Vector DB (Chroma / Qdrant) ]  ──►  store with metadata (file, symbol, lines)


== QUERY (for each question) ==

   User question
        │
        ▼
   [ Retrieval ]  ──►  hybrid search (vector + keyword)
        │
        ▼
   [ Reranking ]  ──►  reorder chunks by actual relevance
        │
        ▼
   [ Prompt ]  ──►  retrieved context + question
        │
        ▼
   [ Local LLM (Qwen3-Coder via Ollama) ]  ──►  generate response
        │
        ▼
   Grounded response + citations (which file and function support it)
```

The core idea of RAG is that, instead of expecting the model to "know" your code (it doesn't, and retraining it would be extremely expensive), you **provide the relevant context on the fly**: you search your repository for the chunks that matter for the question and pass them to the model along with the query. This way, the model responds based on your actual code, not on what it remembers from its training. And since everything (the model, the embeddings, the database) runs locally, your code never leaves your machine.

---

## 2. The Use Case and Why It Is Impressive

The assistant solves a problem that anyone who has worked on a large project knows well: understanding an unfamiliar or extensive codebase. It allows asking in natural language things like "where are payments handled?", "how is the data layer structured?", or "generate a new function following the patterns already used in the project," and getting answers grounded in the actual code, with references to specific files.

Why this is an excellent choice for your profile:

**Demonstrates mastery of the most in-demand AI architecture of 2026.** RAG is currently the standard pattern for connecting an LLM with proprietary data, and building one from end to end (ingestion, chunking, embeddings, retrieval, reranking, generation, evaluation) shows that you understand the full stack, not just how to call an API.

**Total privacy is a real business argument.** Having the code never leave the machine is not just a whim: many companies are forbidden from sending their code to external APIs due to data governance reasons. A 100% local assistant solves exactly that problem, and being able to explain this demonstrates that you think about real-world business constraints.

**It connects with your interest in AI and data science**, and it is visually appealing: a chat assistant that answers questions about real code makes for a very engaging demo.

> **What to index:** For the demo, index a real repository. An excellent and elegant choice is to index **your own MLOps project** (the first one in your plan): you demonstrate both projects at once, and the assistant answers questions about code you know intimately. Alternatively, choose a popular, medium-sized open-source repository. The system should work when pointed at any repository.

---

## 3. Tech Stack (Updated to 2026)

LLM tooling changes very rapidly, so these recommendations reflect the current state. Important: many guides mention models like CodeLlama, which **have been outdated**; the recommendations below are the current ones.

| Layer | Tool | Why This One |
|------|-------------|--------------|
| Language | Python 3.11+ | Standard in AI |
| Environment | uv | Fast and reproducible dependencies (just like in MLOps) |
| LLM Server | **Ollama** | The standard for running local LLMs; serves both model and embeddings |
| Code Model | **Qwen3-Coder** | The best open-source code model for local use in 2026 |
| Embeddings Model | **nomic-embed-text** or **BGE-M3** (local) | Local vectorization, without sending anything outside |
| Vector Database | **ChromaDB** (to start) → **Qdrant** (production) | Chroma is zero-config; Qdrant is more robust |
| RAG Framework | **LlamaIndex** (or LangChain) | LlamaIndex is designed specifically for indexing and RAG |
| Code Chunking | Syntax-aware splitter (tree-sitter / `CodeSplitter`) | Blindly splitting code ruins retrieval |
| Search | Hybrid (vector + BM25) + **reranking** (BGE-reranker) | Boosts precision, key for code identifiers |
| Evaluation | **RAGAS** | Rigorously measures RAG quality |
| API | **FastAPI** | Serve the assistant as a service |
| Interface | **Streamlit** or **Gradio** | A chat UI that is quick to build |
| Quality & Delivery | pytest, ruff, Docker, CI/CD | Reuse practices from the MLOps project |

A few notes on these choices, as knowing the reasoning demonstrates strong judgment:

**The model (Qwen3-Coder).** In 2026, it is the benchmark among open-source code models for local execution: it offers excellent performance on consumer hardware with a permissive license. Choose the size according to your hardware (a 7-8B variant runs on 16 GB laptops; larger sizes if you have more VRAM). Alternatives depending on the case include **GPT-OSS 20B** (good all-rounder), **DeepSeek-Coder** (strong in algorithmic tasks), or **Codestral** (optimized for autocomplete). Check the Ollama library for the exact names, as they change.

**Local embeddings.** To maintain total privacy, embeddings must also be local. `nomic-embed-text` (available in Ollama) is an accessible and widely used option; **BGE-M3** or **GTE-Qwen2** are more powerful alternatives. The key is that no chunk of your code is sent to an external API to be vectorized.

**The vector database.** Start with **ChromaDB**, which is embedded and configuration-free, making it ideal for getting started. When you want something more robust and production-oriented, **Qdrant** (self-hosted) is the reference option, featuring hybrid search and better performance at scale.

**The framework.** **LlamaIndex** is designed specifically for indexing and RAG, offering highly convenient abstractions for this use case; **LangChain** is more general and also valid. We will use LlamaIndex as our primary choice.

---

## 4. Repository Architecture

A clean structure, like in the MLOps project, communicates professionalism. This is the foundation:

```
rag-code-assistant/
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # load and filter repo files
│   │   └── chunker.py           # code-aware chunking
│   ├── indexing/
│   │   ├── embeddings.py        # local embeddings model
│   │   └── vector_store.py      # interaction with vector store
│   ├── retrieval/
│   │   ├── retriever.py         # hybrid search (vector + BM25)
│   │   └── reranker.py          # result reranking
│   ├── generation/
│   │   ├── llm.py               # local LLM client (Ollama)
│   │   └── prompt.py            # RAG prompt construction
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   └── schemas.py           # Pydantic models
│   ├── evaluation/
│   │   └── ragas_eval.py        # RAG quality evaluation
│   └── config.py                # centralized configuration
├── ui/
│   └── app.py                   # chat interface (Streamlit/Gradio)
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_api.py
├── eval/
│   └── questions.yaml           # evaluation question set
├── data/
│   └── vector_store/            # persisted index (managed, not in Git)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml       # app + Ollama + (Qdrant)
├── pyproject.toml
├── Makefile
└── README.md
```

The organization follows the system's workflow: `ingestion` (load and split), `indexing` (vectorize and store), `retrieval` (search and reorder), `generation` (build prompt and generate), plus `api`, `ui`, and `evaluation`. Each folder has a clear responsibility, making the system easy to understand and extend.

---

## 5. Phase-by-Phase Roadmap

Each phase has an objective, tasks, a deliverable, and a definition of "done." Since this project is shorter than the MLOps project, the phases are lighter, but the principle is the exact same: do not move forward without meeting the criteria.

---

### 🔹 Phase 0 — Setup & Fundamentals (Week 1, First Few Days)

**Objective:** Environment ready, Ollama running, and use case defined.

**Tasks:**
- [ ] Create the repo with the folder structure, a `uv` environment, code quality tools (ruff, pre-commit), and a Makefile, just like in the MLOps project.
- [ ] Install **Ollama** and download the models: the code model (Qwen3-Coder in the size your hardware allows) and the embeddings model (nomic-embed-text).
- [ ] Verify you can perform local inference with the model (`ollama run`).
- [ ] Choose the repository you are going to index (your MLOps project or another).
- [ ] Document the use case and stack decisions.

**Deliverable:** Reproducible environment + Ollama serving the models locally.  
**Definition of Done:** You can generate text with the local LLM and obtain an embedding, both without an internet connection.

---

### 🔹 Phase 1 — Ingestion: Loading and Filtering Code (Weeks 1-2)

**Objective:** Turn a repository into a clean collection of code documents.

**Tasks:**
- [ ] Write `loader.py` to traverse the repository and load the code files.
- [ ] Filter intelligently: ignore binaries, dependency folders (`node_modules`, `.venv`), build artifacts, and respect `.gitignore`.
- [ ] Associate metadata with each file (path, language, size).
- [ ] Handle multiple programming languages.

**Deliverable:** A function that, given a repository, returns its relevant code files along with their metadata.  
**Definition of Done:** You point the ingestion at a real repository and get only the relevant code files, without any clutter.

---

### 🔹 Phase 2 — Code-Aware Chunking (Week 2)

**Objective:** Split code while respecting its structure. **This is the key phase of the project.**

**Tasks:**
- [ ] Understand why naive chunking (cutting every N characters) ruins code: it splits functions in half, separates a signature from its body, and breaks retrieval.
- [ ] Write `chunker.py` using a **syntax-aware** splitter: chunk by logical units (functions, classes, methods) using syntax trees (tree-sitter or LlamaIndex's `CodeSplitter`).
- [ ] Associate rich metadata with each chunk: source file, symbol (function/class name), line range.
- [ ] Handle overly large chunks (splitting while respecting logic).

**Deliverable:** A chunking system that produces coherent fragments aligned with the code structure.  
**Definition of Done:** Generated chunks are meaningful code units (a full function, a class), not arbitrary cuts mid-line.

---

### 🔹 Phase 3 — Embeddings and the Vector Database (Week 3)

**Objective:** Vectorize the chunks and store them for searching.

**Tasks:**
- [ ] Write `embeddings.py` to vectorize each chunk using the local model.
- [ ] Write `vector_store.py` to store the vectors in ChromaDB along with their metadata.
- [ ] Build the complete indexing pipeline: repo → ingestion → chunking → embeddings → storage.
- [ ] Persist the index so you do not have to re-index every time.

**Deliverable:** A pipeline that indexes an entire repository into the local vector database.  
**Definition of Done:** You index a repository and can run a similarity search that returns relevant code chunks.

---

### 🔹 Phase 4 — Retrieval: Hybrid Search and Reranking (Week 4)

**Objective:** Retrieve the *correct* chunks for each question.

**Tasks:**
- [ ] Write `retriever.py` with vector (semantic) search over the database.
- [ ] Add **hybrid search**: combine vector search with keyword search (BM25). This is particularly valuable for code, where exact identifiers (function names, variables) matter immensely and keyword search excels at capturing them.
- [ ] Add a **reranker** (BGE-reranker) to reorder retrieved chunks by actual relevance, improving precision.
- [ ] Allow metadata filtering (e.g., searching only within a specific file or language).

**Deliverable:** A retrieval system that, given a question, returns the most relevant code chunks.  
**Definition of Done:** When asked a specific question about the code, the retrieval returns the chunks that actually answer it, sorted in the correct order.

---

### 🔹 Phase 5 — Generation with the Local LLM (Week 5)

**Objective:** Generate responses grounded in the retrieved code.

**Tasks:**
- [ ] Write `llm.py`, the client that interacts with Qwen3-Coder via Ollama.
- [ ] Write `prompt.py` to construct the RAG prompt: combining retrieved chunks (context) with the user's question.
- [ ] Design the prompt so the model is **grounded in the context** and does not hallucinate: instruct it to answer only using what the code supports and to cite the source files/functions.
- [ ] Handle project-aware code generation (following the repository's patterns).

**Deliverable:** The complete RAG pipeline: question → retrieval → generation → response with citations.  
**Definition of Done:** You ask a question about the code and get an accurate, grounded response that cites the specific files on which it is based.

---

### 🔹 Phase 6 — API and Interface (Week 6)

**Objective:** Convert the system into a usable product.

**Tasks:**
- [ ] Write the API with **FastAPI**: an endpoint to index a repository and another to ask questions, using Pydantic schemas.
- [ ] Build a **chat interface** with Streamlit or Gradio, allowing you to converse with the assistant and view file citations.
- [ ] Display the sources for each response (which code chunks support it), as transparency is key in RAG.

**Deliverable:** A functional chat application for the code assistant.  
**Definition of Done:** You open the interface, type a question about the indexed repository, and receive a response with its sources in a smooth chat experience.

---

### 🔹 Phase 7 — Evaluation with RAGAS (Week 7, First Half)

**Objective:** Rigorously measure the quality of the system. **This is what elevates the project.**

**Tasks:**
- [ ] Build an evaluation dataset of questions about the repository along with their expected answers.
- [ ] Write `ragas_eval.py` to evaluate the system using **RAGAS**, measuring metrics such as *faithfulness* (is the answer truly grounded in the context without hallucinating?), *answer relevance*, and *context precision/recall*.
- [ ] Use the results to iterate: adjust the chunking, the number of retrieved chunks, the reranking, or the prompt, and measure if it improves.

**Deliverable:** An evaluation report with RAG quality metrics.  
**Definition of Done:** You have objective metrics for your system and have used the evaluation to improve it measurably, rather than guessing.

---

### 🔹 Phase 8 — Packaging and Presentation (Week 8)

**Objective:** Showcase the value of the system.

**Tasks:**
- [ ] Containerize with **Docker** and a `docker-compose.yml` that spins up the app alongside Ollama (and Qdrant if you use it).
- [ ] Write the **definitive README**: architecture diagram, video demo, design decisions, RAGAS metrics, and setup instructions.
- [ ] Record a **video** showing the assistant answering questions about a real repository, including citations. The argument for total privacy (fully local) is a great selling point.
- [ ] Write a blog post explaining what you built and the reasoning behind it.

**Deliverable:** Containerized system + excellent README + video + post.  
**Definition of Done:** Someone can understand the project in two minutes using the README and video, and spin it up using Docker.

---

## 6. The Perfect README (Brief Checklist)

Just like in MLOps, the README has the greatest impact. It should include: a catchy title and description; architecture diagram; a GIF or video of the assistant in action; the problem and why it matters (highlighting the privacy aspect); the tech stack with badges; working instructions; a **design decisions** section (why Qwen3-Coder, why code-aware chunking, why hybrid search, why 100% local); **RAGAS metrics** as proof of quality; and what you learned. (This section can be expanded in detail later, as done in the previous project.)

---

## 7. Common Mistakes to Avoid

| Mistake | Why It Is Bad | What to Do |
|-------|-----------------|-----------|
| Naive code chunking | Ruins retrieval: splits functions, separates signatures | Syntax-aware chunking |
| Only vector search | Fails with exact code identifiers | Hybrid search (vector + BM25) |
| Not citing sources | Users cannot verify answers; feels like opaque magic | Always display source chunks |
| Not evaluating (guessing everything) | You don't know if it works or if changes improve it | Evaluate with RAGAS and measure |
| Letting the model hallucinate | It fabricates answers not backed by the code | Use a prompt that forces grounding in context |
| Sending embeddings to an external API | Breaks the promise of total privacy | Local embeddings (the key selling point of the project) |
| Re-indexing everything every time | Slow and unfeasible for large repositories | Persist the index; use incremental indexing |

---

## 8. Stretch Goals (If You Want to Go Further)

- **Incremental indexing:** Detect which files have changed and re-index only those, rather than the entire repository.
- **Agentic assistant:** Enable the model to perform multiple sequential searches to answer complex questions (reasoning about what to look for next), moving closer to an AI coding agent.
- **Multi-repository support:** Index and query multiple repositories simultaneously, with project-level filtering.
- **Code generation with verification:** Attempt to run or lint generated code to validate that it works.
- **Model comparison:** Evaluate several local LLMs (Qwen3-Coder, DeepSeek-Coder, GPT-OSS) using RAGAS and document which performs best for your use case.
- **Editor integration:** A VS Code extension that queries your assistant directly from the editor.

---

## 9. Learning Resources

- **LlamaIndex and LangChain documentation:** The standard references for building RAG pipelines; LlamaIndex is particularly useful for this use case.
- **Ollama documentation:** For serving local models and checking the current names and sizes.
- **RAGAS documentation:** For rigorously evaluating RAG systems.
- **Qdrant and Chroma:** Their guides cover vector, hybrid, and reranking searches with clear examples.
- **Concepts to master:** What RAG is and why it outperforms fine-tuning for proprietary data; the importance of chunking; dense vs. sparse vs. hybrid search; the role of reranking; and how to evaluate RAG quality (faithfulness, relevance, and context precision).

---

## 10. Milestone Summary

| Week | Milestone | Status |
|--------|------|--------|
| 1 | Environment + Ollama + local models | ⬜ |
| 1-2 | Repository ingestion and filtering | ⬜ |
| 2 | Code-aware chunking | ⬜ |
| 3 | Embeddings + vector database | ⬜ |
| 4 | Hybrid retrieval + reranking | ⬜ |
| 5 | Grounded generation with the local LLM | ⬜ |
| 6 | API + chat interface | ⬜ |
| 7 | Evaluation with RAGAS | ⬜ |
| 8 | Packaging, README, video | ⬜ |

---

**The ultimate goal:** When someone looks at this project, they should think, *"this person truly understands modern AI architectures."* This is not just a script calling an OpenAI API; it is a complete, local, and private RAG system with sophisticated retrieval, grounded generation, and rigorous evaluation. This, combined with the MLOps project, profiles you as someone who understands both production engineering and applied AI—a highly powerful combination.
