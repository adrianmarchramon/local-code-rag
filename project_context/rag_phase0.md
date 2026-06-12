# Phase 0 — Setup and Fundamentals

> This is the first phase and, just like in the MLOps project, you dedicate it to building foundations before touching the system's logic. It has two halves: a **familiar** one, preparing the Python environment using the best practices you already mastered in the previous project, and a **new, project-specific** one, setting up the local models with Ollama and verifying that you can indeed generate text and compute embeddings without an internet connection. This second part is what makes the central promise of the project (running everything locally) possible, so it deserves special attention.

**Phase objective:** Environment ready, Ollama running with local models, and the use case defined.  
**Duration:** First days of week 1.  
**By the end, you will have:** A structured and reproducible repository, Ollama serving the code and embedding models on your machine, the certainty that both work without internet, your chosen repository to index, and documented design decisions.

---

## Before you begin: Hardware and tools

This project has a unique characteristic that the MLOps project did not: it runs language models on your own machine, which imposes hardware requirements that you should clarify from the beginning to make the right choice of model.

The key is **memory** (RAM, or VRAM if you have a dedicated GPU). A language model must be loaded entirely into memory to run, so the size of the model you can execute depends on how much memory you have. As a reference: with about 16 GB of RAM, you can comfortably run a 7B code model (around 5 GB); with 24–32 GB or more, you can run better models. Macs with Apple Silicon (unified memory) work especially well for this. If your hardware is very limited, don't worry: choose a smaller model and the project will work just the same, only with slightly less powerful responses. The important thing is to be realistic about your machine when choosing a model, which we will do in the download step.

The base tools you need installed are the same as in MLOps, plus Ollama:

**Git, uv, and an editor** (VS Code recommended), just like in the previous project. If you are coming from the MLOps project, you already have them.

**Ollama**, the tool that runs the models locally. You will install it in Step 2.

**Docker Desktop**, which you will need for containerization in Phase 8; it is best to have it ready, even if you won't use it until the end.

---

## Step 1 — The Repository and the Environment

This part directly reuses what you learned in Phase 0 of the MLOps project, so we will cover it quickly, focusing on the specifics of this project.

Create the repository on GitHub (with a README, Python `.gitignore`, and license), clone it, and initialize it with uv and the Python version:

```bash
git clone https://github.com/YOUR_USERNAME/rag-code-assistant.git
cd rag-code-assistant
uv init
uv python pin 3.11
```

Set up the folder structure we designed in the fundamentals, which follows the flow of the system:

```bash
mkdir -p src/ingestion src/indexing src/retrieval src/generation src/api src/evaluation
mkdir -p ui tests eval data/vector_store docker
touch src/__init__.py src/config.py
touch src/ingestion/__init__.py src/ingestion/loader.py src/ingestion/chunker.py
touch src/indexing/__init__.py src/indexing/embeddings.py src/indexing/vector_store.py
touch src/retrieval/__init__.py src/retrieval/retriever.py src/retrieval/reranker.py
touch src/generation/__init__.py src/generation/llm.py src/generation/prompt.py
touch src/api/__init__.py src/api/main.py src/api/schemas.py
touch src/evaluation/__init__.py src/evaluation/ragas_eval.py
touch ui/app.py eval/questions.yaml docker/Dockerfile docker/docker-compose.yml
```

For this phase, the minimum dependencies are the Ollama Python client (to verify embeddings from code) and the quality tools. The rest of the stack (LlamaIndex, ChromaDB, FastAPI, RAGAS, the interface) will be added in their corresponding phases, just as you did in MLOps:

```bash
uv add ollama
uv add --dev pytest ruff pre-commit
```

Configure `ruff` and `pre-commit` exactly as you did in the previous project (the ruff configuration in `pyproject.toml`, `.pre-commit-config.yaml` with the ruff hooks, and `uv run pre-commit install`), and create a `Makefile` with common commands. Also make sure your `.gitignore` excludes the persisted index, which is a generated and potentially large artifact:

```gitignore
data/vector_store/
.venv/
__pycache__/
```

Reusing this entire MLOps project foundation is one of the advantages of doing the projects in order: you already master the structure, version control, and quality tools, allowing you to set up the foundations in a fraction of the time.

---

## Step 2 — Install Ollama

This is where the project-specific tasks begin. **Ollama** is the tool that runs LLMs on your machine and serves them through a simple local API. Install it:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

(On Windows and macOS, you can also download the desktop application from `ollama.com`.)

Once installed, Ollama runs as a background service exposing a local API at `http://localhost:11434`. This is the endpoint your application will connect to in order to talk to the models. If it is not running at any point, you can start it with `ollama serve`. You can verify that it is active by opening that address in your browser (it should respond with a message indicating that Ollama is running) or by running any Ollama command.

It is worth internalizing Ollama's role because it is the cornerstone of the project's privacy: everything that goes through it (text generation and embedding calculation) happens on your machine, without sending anything to external servers. This is what makes the "100% local" promise real.

---

## Step 3 — Download the Models

Now you will download the two models needed by the system: the code model (to generate responses) and the embedding model (to vectorize). Downloading a model is a single command, and Ollama takes care of everything.

For the **code model**, the choice depends on your hardware, as discussed. The current recommendation (2026), if your machine allows it (around 24–32 GB of memory), is **Qwen3-Coder** in its 30B variant, which is a Mixture of Experts (MoE) model: although it has 30B total parameters, it only activates about 3.3B per inference, meaning it performs like a large model but runs with the efficiency of a small one. Additionally, it is designed to understand entire repositories:

```bash
ollama pull qwen3-coder:30b
```

If your hardware is more modest (around 16 GB), a comfortable and highly capable option is a 7B code model, which takes up about 5 GB:

```bash
ollama pull qwen2.5-coder:7b
```

Both are excellent for this project; the first provides more powerful responses in exchange for more memory. Check `ollama.com/library` to see the exact sizes and tags available, as they change frequently. If in doubt, start with the 7B model: it allows you to progress through the entire project, and you can always switch to a larger one later (remember that switching models is, thanks to Ollama, just a matter of changing a name in the configuration).

For the **embedding model**, download `nomic-embed-text`, a lightweight, fast, and widely used option ideal for local vectorization:

```bash
ollama pull nomic-embed-text
```

Verify that both models are installed:

```bash
ollama list
```

You should see your code model and `nomic-embed-text` listed.

---

## Step 4 — Verify Local Inference

This is the step that confirms the foundation of the project works, and serves as the success criterion for the phase: verifying that you can **generate text** and **compute embeddings**, both locally and without internet.

First, generation. Launch an interactive session with your code model:

```bash
ollama run qwen3-coder:30b
```

(Replace with the model you downloaded.) Type a request, for example, "write a Python function that reverses a string", and verify that the model generates a response. To exit, type `/bye`. If you see a response, your local LLM works. As a definitive proof that it is running locally, you can disconnect your Wi-Fi and try again: it will keep working, because everything happens on your machine.

Second, embeddings. You can verify this from code or with a request to the local API. Using the Python client you installed:

```python
import ollama

result = ollama.embed(model="nomic-embed-text", input="def greet(): return 'hello'")
vector = result["embeddings"][0]
print(f"Embedding generated, dimension: {len(vector)}")
```

Or directly via the local API, without writing code:

```bash
curl http://localhost:11434/api/embed -d '{"model": "nomic-embed-text", "input": "def greet(): return hello"}'
```

If you get a vector of numbers (a long list of values), the embedding model is working. With this, you have confirmed both core capabilities that power the entire project: generating and vectorizing, both locally.

---

## Step 5 — The Critical Detail of the Context Window

There is an Ollama configuration detail that is easy to overlook and matters greatly **specifically in a RAG project**, so it is best to address it from the beginning. By default, Ollama limits the **context window** of models to 4,096 tokens. The context window determines how much text the model can "read" at once (the input you pass to it).

Why this is a problem in RAG: the essence of your system is to inject retrieved code snippets into the prompt, along with the question. If you retrieve multiple code chunks, it is very easy to exceed those 4,096 tokens. If the context overflows, the model won't see some of the information you provided, ruining the responses. Since the code model you use supports much larger windows (Qwen3-Coder supports up to 256,000 tokens), the limitation is not from the model itself, but from Ollama's default configuration.

The solution is to **expand the context window**. There are several ways to do this. The cleanest approach for this project is to create a custom model with a `Modelfile` that sets a larger window:

```
# File: Modelfile
FROM qwen3-coder:30b
PARAMETER num_ctx 32768
```

And create the model from it:

```bash
ollama create qwen3-coder-rag -f Modelfile
```

Alternatively, when you call the model from your code (in Phase 5, via LlamaIndex), you can specify the context window as a parameter. The important thing is to keep this limit in mind from now on: a sufficient context window (for example, 32,000 tokens) is necessary for your RAG to pass all the retrieved code to the model without truncation. Anticipating this detail—which many discover late after receiving inexplicably poor answers—demonstrates that you understand the interaction between RAG and the model.

---

## Step 6 — Choose the Repository to Index

Before closing this phase, decide which codebase your assistant will operate on, as you will need it in the next phase. As discussed in the fundamentals, an excellent and elegant option is to **index your own MLOps project**: you showcase both projects at once, and since you know that codebase inside out, you will easily be able to judge if the assistant's responses are good (you know what the correct answer should be). Alternatively, a popular, medium-sized open-source repository works just as well.

Clone the chosen repository to an accessible local path (outside of your project directory, to avoid mixing them up):

```bash
git clone https://github.com/YOUR_USERNAME/mlops-fraud-pipeline.git ~/repos/project-to-index
```

Take a look at it to get an idea of its size and the languages it contains: how many code files it has, what languages they are in, and whether there are folders that should be ignored (dependencies, builds). This familiarity with the corpus will help you make good decisions regarding ingestion and chunking in the upcoming phases.

---

## Step 7 — Document the Use Case and Decisions

Just like in the MLOps project, take a moment to document your decisions in writing, as articulating them is what demonstrates solid judgment during an interview. Create a document (for example, `docs/decisions.md`) where you record the following, along with your reasoning:

**The use case.** What the assistant does and for whom: answering questions about a codebase and generating project-aware code, all locally for privacy.

**Model choice and rationale.** Which code model you are using and why (balancing capability and your hardware), mentioning that Qwen3-Coder is the current benchmark and that you avoided outdated models like CodeLlama. Also document your hardware setup, as it constrains the choice.

**The privacy argument.** Why everything is local (the model, embeddings, database) and what business problem this solves (companies that cannot send their code to external APIs). This is your main differentiator, so having it well-articulated is key.

**Corpus selection.** Which repository you are going to index and why, along with its characteristics (size, languages).

**Stack decisions.** A summary of why you chose the main components (Ollama, ChromaDB, LlamaIndex, hybrid search, reranking, RAGAS), which you will refer to when writing the design decisions section of the README.

---

## Verification: The "Definition of Done"

The phase is complete when the following criteria are met:

- [ ] The repository is created, structured according to the system flow, with a reproducible environment (uv, ruff, pre-commit, Makefile), reusing MLOps practices.
- [ ] Ollama is installed and running, serving its local API.
- [ ] The code model and `nomic-embed-text` are downloaded (`ollama list` displays them).
- [ ] You are aware of the context window limit and know how to expand it for RAG.
- [ ] You have cloned the repository to be indexed.
- [ ] The decisions (use case, model, privacy, corpus, stack) are documented.
- [ ] **The key test:** You can generate text with the local LLM and obtain an embedding, both without an internet connection.

The key test is the double offline verification: if, with Wi-Fi disconnected, you can ask the model to generate code and retrieve an embedding vector, you have confirmed that the project's local foundation works. That complete independence from the internet is, quite literally, the project's core promise realized from day one.

---

## Deliverables and What's Next

By closing Phase 0, you have the project's foundations: a structured and reproducible repository, Ollama serving models locally, confirmation that you generate and vectorize without internet, your chosen corpus, and documented decisions. Additionally, you have solved the part that differs the most from the previous project (setting up local models) and anticipated a subtle yet important detail (the context window) that will make a difference down the road.

The next step, **Phase 1**, is where the system begins to take shape: **ingestion**, which will traverse your chosen repository, load its code files, and carefully filter them, ignoring irrelevant content. It is the first link in the indexing pipeline, relying directly on the corpus you just cloned and the familiarity you gained during this phase. You have gone from "having the environment and models ready" to being on the verge of "starting to turn a repository into searchable knowledge".
