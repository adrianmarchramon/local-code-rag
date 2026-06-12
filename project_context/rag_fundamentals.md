# Local LLM with RAG for Code — Fundamentals and Design

> This document is the conceptual design work prepared prior to writing code. It covers what RAG is and why it is the architecture you are going to build, what exactly the system does and how, the use case and why it matters, the technology stack justified piece by piece, and the repository architecture. Understanding these foundations is what separates a project copied from a tutorial from one designed with deliberate engineering criteria.

---

## Introduction: What This Project Really Is

To understand this project, you have to start with a fundamental limitation of language models. No matter how powerful an LLM is, it has two major constraints: it does not know your private data (your code was never part of its training), and its knowledge is frozen at the moment of its training. If you ask a model, "How does authentication work in my project?", it has no idea because it has never seen your codebase [INDEX]. It might generate a plausible-sounding answer, but it won't be about *your* code.

There are two ways to give a model knowledge it doesn't have. The first is **fine-tuning**: retraining the model with your data so that it "learns" it. This sounds appealing, but in practice, it is expensive, slow, requires expertise, and suffers from a major drawback: as soon as your code changes (and it changes constantly), the model's knowledge becomes obsolete, requiring you to retrain it all over again. The second approach is **RAG (Retrieval-Augmented Generation)**, and that is what you are going to build [INDEX].

The concept behind RAG is elegant and can be easily understood using a metaphor: instead of forcing the model to memorize all your code (like a closed-book exam), you give it an **open-book exam** [INDEX]. When a question arrives, you first *search* your codebase for the specific code snippets relevant to that question, and then you *pass them to the model* alongside the query, essentially instructing it: "Answer this question using these accompanying code snippets" [INDEX]. The model doesn't need prior knowledge of your code; you provide the necessary context at the exact moment it is needed [INDEX]. This solves both problems of fine-tuning at once: it is cost-effective (no retraining required) and always up to date (you search against your current codebase, whatever state it is in) [INDEX].

This is why, in 2026, RAG has become the **industry-standard architecture** for connecting an LLM to proprietary data, representing one of the most sought-after patterns in applied AI [INDEX]. Building an end-to-end RAG system demonstrates that you understand how to work with LLMs in real-world enterprise environments.

### The Most Important Concept: Retrieval is Paramount

There is one central idea about RAG that is vital to internalize from the start, as it dictates where the actual engineering effort lies: **the quality of a RAG system is primarily determined by retrieval, not generation**. The logic is simple and mirrors the MLOps principle you saw earlier: "the model is the small part." If you retrieve the wrong code snippets, even the best LLM in the world will produce a poor answer because it is reasoning over incorrect context. Conversely, if you retrieve the correct snippets, even a modest model can produce an excellent response. "Garbage in, garbage out."

This has a direct consequence: most of your engineering effort—and the parts of the project that will stand out most—will not be spent on calling the LLM (which is the easy part, almost a single line of code). Instead, it will be spent on building a robust **retrieval pipeline**: how you chunk the code, embed it, search it, and rerank the results. A project that merely plugs in an LLM API does not impress; one that implements a sophisticated retrieval pipeline shows you truly understand how to make RAG work.

### The Differentiating Angle: Entirely Local

There is a second design decision that defines this project and makes it exceptionally valuable: **it runs 100% locally**. The model, the embeddings, and the database all run on your machine, and not a single byte of your code ever leaves your system to access the internet. This is not a technical whim; it is a direct response to a very real and widespread industry need: many companies are strictly **prohibited** from sending their code to external APIs due to intellectual property, regulatory compliance, or data governance constraints. For these organizations, an assistant that sends code to a third-party server is a non-starter. A system that runs entirely locally resolves this bottleneck.

Knowing how to articulate this demonstrates a quality that companies value as much as technical skills: the ability to design within real-world business constraints, rather than just focusing on whether something works. There is, of course, an honest trade-off that is worth acknowledging: local models that fit on consumer-grade hardware are less capable than frontier cloud-based models. However, local models in 2026 are highly capable of handling the vast majority of daily programming tasks. For sensitive codebases, the guarantee of complete privacy is often far more valuable than those final percentage points of model capability. Presenting this trade-off honestly is, in itself, a clear sign of professional maturity.

### Who This Impresses and Why

Just like in the MLOps project, it is worth being explicit about your audience because it guides your design decisions. The **engineer** interviewing you will not be impressed by a simple LLM API call (anyone can do that); they will be impressed to see that you understand the difficult parts of RAG: chunking code while respecting its structural syntax, combining semantic and lexical search, using a reranker, and *evaluating* the system's quality objectively instead of relying on subjective gut checks. The **technical recruiter**, who often doesn't dive deep into the codebase, will pick up on two strong signals: the total privacy argument (a differentiator anyone can grasp) and an engaging chat demo where the assistant answers questions about real code. Finally, the **team lead** will see that you have mastered the most in-demand AI architecture of the moment and know how to turn it into a functional product.

With this in mind, every decision in this document is directed toward a single goal: making anyone who reviews your project think, *"This person truly understands modern AI architectures, rather than just knowing how to plug in an API."*

---

## 1. What Exactly You Are Going to Build

The system operates in two highly distinct phases, and understanding this separation is fundamental to grasping the architecture. There is an **indexing** phase, which is run once per repository and is slow and computationally heavy, and a **querying** phase, which occurs on every question and must be fast. They act like two clocks ticking at different speeds over the same knowledge base, similar to how we distinguished the prediction path from the model lifecycle in the MLOps project.

```
== INDEXING (Once per repository) ==
   Repository → Ingestion → Chunking → Embeddings → Vector DB

== QUERYING (For each question) ==
   Question → Retrieval → Reranking → Prompt → LLM → Answer with Citations
```

### The Indexing Phase: Preparing the Knowledge Base

Indexing transforms your code—which is text structured for humans and compilers—into a format that can be *searched by meaning*. It occurs in four sequential steps.

**Ingestion** traverses the repository and loads the code files. While it sounds trivial, it requires careful filtering: you must ignore irrelevant files (binary files, dependency directories like `node_modules` or `.venv`, and build artifacts) while respecting what the project itself flags to be ignored. The output is a clean collection of code files, each annotated with metadata (file path, language).

**Chunking** splits those files into manageable fragments, and this is where the first critical, code-specific decision comes in. You cannot feed an entire thousand-line file to the model (it would exceed the context window and dilute the relevant details), so you must split it. However, as we will discuss, *how* you split it changes everything: naive text chunking ruins code, whereas chunking that respects its structural syntax is what makes retrieval work.

**Embeddings** convert each code fragment into a vector: a list of numbers representing its *semantic meaning*. This is the core mechanism that allows you to search by intent rather than just literal keyword matching; two fragments performing similar tasks will have similar vectors, even if they use entirely different function names or variables. A locally executed embedding model performs this conversion.

**Storage** saves these vectors in a vector database, along with their associated metadata (which file the fragment came from, what function or class it represents, and the line numbers it occupies). This database is optimized for a single operation: given a query vector, quickly find the most similar vectors. Once indexing is complete, you have your codebase converted into a semantically searchable knowledge base, persisted so you do not have to rebuild it every time.

### The Querying Phase: Generating Answers with Context

Querying occurs every time a user asks a question, and it must be responsive and fast. It also consists of several stages.

**Retrieval** is the heart of the system. The user's question is converted into a vector using the same local embedding model, and the vector database is searched for the code fragments with the most similar vectors—meaning those most semantically relevant to the question. However, as we will see, purely semantic search is insufficient for code: you must combine it with keyword-based search because exact names are critical in programming.

**Reranking** refines the initial results. The initial retrieval is fast but can be imprecise. The reranking step takes the candidate fragments and reorders them using a more precise model, ensuring that the fragments most relevant to answering the query rise to the top. It serves as a secondary quality filter that significantly improves accuracy.

**Prompt Construction** assembles the final prompt. The top-performing retrieved fragments are extracted and combined with the user's question into a structured instruction for the model. The instruction essentially tells the model: "Answer this question based on these provided code fragments, and cite where you found the information."

**Generation** is the final step where the LLM is invoked. The local model receives the constructed prompt (context + question) and generates the response. Because it is provided with the correct context and instructed to base its answer on it, the generated response is grounded in your actual code, rather than what the model vaguely recalls from its pre-training data.

And a final, crucial detail: **citations**. The response does not arrive in isolation; it is accompanied by its sources, detailing which files and functions support the answer [INDEX]. This allows the user to verify the response, which is the hallmark of a well-engineered RAG system. It transforms an opaque response ("trust me") into a transparent, verifiable answer ("I am saying this because of this code right here") [INDEX].

---

## 2. The Use Case: An Assistant That Understands Your Codebase

It is worth focusing on exactly what problem this assistant solves. A well-defined use case gives technical choices clear purpose and provides a compelling story to share during an interview.

The problem this system addresses is universal for anyone who has ever worked with code they didn't write, or with a codebase too large to keep entirely in their head: **understanding a codebase**. Where are payments processed? How is the data layer set up? What does this function do, and where is it called? How would I write a new function that aligns with the established patterns of the project? Answering these questions manually involves digging through files, reading lines of code, and mentally piecing together how they fit. This assistant, by contrast, allows you to ask these questions in natural language and receive grounded answers based on the real code, complete with references to the exact files. It is, in essence, a tool for rapid comprehension and developer onboarding.

### Why Total Privacy is the Central Value Proposition

We touched on this in the introduction, but it is worth expanding upon as it is the primary differentiator of this project. Excellent commercial code assistants exist, but nearly all of them function by sending your code to external servers. For many organizations, this is a strict dealbreaker: banks, healthcare providers, defense contractors, and any company with highly valuable proprietary code or strict compliance requirements often have **strict policies** prohibiting their code from leaving their infrastructure. It isn't a preference; it is a mandate.

An assistant that runs entirely locally eliminates this barrier. Since the code never leaves the host machine, it satisfies even the most stringent security policies. Building the system with this constraint (local model, local embeddings, local database) is not just a technical exercise; it addresses a concrete business challenge. Being able to articulate this value during an interview ("I built this with a focus on organizations that cannot use cloud-based AI tools due to strict data governance") demonstrates that you understand the business context behind technology decisions, not just the technology itself.

### Which Codebase to Index

To make the project demonstratable, the system needs to run on a real-world repository. You have several options, and one particularly elegant approach is to **index your own MLOps project**—the first project in your portfolio. This offers a dual advantage: you demonstrate both projects at once, and the assistant answers questions about a codebase you already know inside out, making it easy for you to evaluate the quality of its responses. Alternatively, any popular, mid-sized open-source repository serves just as well for demonstrations. The important design principle is that the system should not be coupled to a specific repository, but should be capable of indexing any codebase it is pointed to.

---

## 3. The Technology Stack, Justified Piece by Piece

Selecting the right tools and, more importantly, understanding *why* you chose them, is a key indicator of engineering judgment. In the rapidly evolving landscape of LLMs, this is especially true, as many guides still recommend tools that are already outdated. A quick but important warning before we begin: many older tutorials recommend models like **CodeLlama**, which by 2026 **have been clearly surpassed**. The recommendations below reflect the current state of the art.

The **governing philosophy** of the entire stack is simple: **everything runs locally**. Every tool is chosen to uphold the promise that your code never leaves the machine. This immediately rules out any cloud services for the model, embeddings, or database, guiding us toward open-source, self-hosted alternatives. This mirrors the MLOps principle of "lightweight tools tailored directly to the problem."

### The Model Server: Ollama

**Ollama** is the tool that allows you to run LLMs on your own machine with minimal friction. It downloads, manages, and serves models via a clean local API, abstracting away the underlying complexity of running large models on consumer hardware. In 2026, it is the de facto standard for running local LLMs, serving as the foundation of your system's privacy: both your generation model and your embedding model will run through it. Its primary benefit is simplicity; switching models is as simple as updating a configuration name.

### The Code Model: Qwen3-Coder

The model serves as the generative engine of your system, and staying current here is vital. In 2026, the benchmark for open-source code models suitable for local execution is **Qwen3-Coder**. It delivers outstanding performance on programming tasks while running efficiently on consumer hardware under a permissive license, effectively succeeding the models recommended in older guides.

The choice of **model size** depends on your available hardware, and it is important to understand the trade-offs: larger models are more capable but require significantly more GPU memory (VRAM). A variant in the 7-8B parameter range runs comfortably on mid-tier laptops (16 GB of RAM), whereas larger sizes require more VRAM but offer superior reasoning. Depending on your hardware and requirements, several local **alternatives** exist: **GPT-OSS 20B** (OpenAI’s open-weights model, a solid all-rounder), **DeepSeek-Coder** (exceptionally strong at algorithmic tasks and logical reasoning), **Codestral** (optimized for code autocompletion), or **Devstral** (designed for agentic workflows). The Ollama model library remains the definitive reference for current model names and sizes, as they evolve frequently. Demonstrating that you understand this landscape and can justify your model choice based on hardware and task requirements shows true technical currency.

### Embeddings: Also Local

For the system to remain fully private, the embeddings (converting code to vectors) must also be computed locally, rather than sending code to an external embedding API, which would violate the privacy guarantee. **nomic-embed-text**, accessible directly through Ollama, is a lightweight, widely adopted, and accessible option to get started. If you require higher retrieval quality, models like **BGE-M3** or **GTE-Qwen2** are highly capable local embedding models. Because the choice of embedding model directly impacts search quality, it is a crucial component of your pipeline, and keeping it local is non-negotiable.

### The Vector Database: From Chroma to Qdrant

Once you generate your vectors, you need a place to store and query them efficiently: a **vector database**, designed to quickly locate the closest vectors to a given query. To **start out**, **ChromaDB** is ideal: it is embedded (it runs within your application process without requiring a separate server) and requires virtually zero configuration, making it perfect for rapid prototyping. When you are ready for a more robust, production-ready solution, **Qdrant** (self-hosted) is the industry standard: it provides superior performance at scale and native support for hybrid search, which is exactly what a code-focused RAG system needs. Other options exist (such as Milvus, Weaviate, pgvector, or managed services like Pinecone), but for a local project, starting with Chroma and transitioning to Qdrant covers the spectrum beautifully. Starting simple and justifying an upgrade to a more robust system is a sign of practical engineering maturity.

### The Framework: LlamaIndex

Building a RAG pipeline completely from scratch would require a lot of boilerplate code to coordinate ingestion, chunking, embeddings, and retrieval. An **orchestration framework** provides clean abstractions for these tasks. The two primary options are **LlamaIndex** and **LangChain**. While LangChain is more general-purpose (designed for building all types of LLM applications), **LlamaIndex is tailored specifically for indexing and retrieval**, which is the core of this project. It offers highly effective out-of-the-box utilities, including a syntax-aware code splitter. For this reason, LlamaIndex is our primary choice, though LangChain remains a highly defensible alternative. Understanding the differences between these frameworks and selecting the one best suited to the task shows strong ecosystem knowledge.

### Syntax-Aware Chunking

This is not a specific library but a critical technique, and it is so central to code-based RAG that it warrants its own discussion. As mentioned, *how* you split your code dictates whether retrieval works. Naive chunking (splitting text strictly every N characters or lines) is disastrous for source code: it cuts functions in half, separates function signatures from their bodies, and breaks classes into meaningless fragments. These broken chunks are useless for retrieval because they lack logical unity, and they only serve to confuse the LLM.

The alternative is **syntax-aware chunking**: splitting code along its natural logical boundaries (functions, classes, methods) while preserving its structure. This is accomplished using tools that understand programming language syntax, such as **tree-sitter** (an incremental parser) or LlamaIndex's built-in `CodeSplitter`. Each resulting chunk is a coherent, self-contained unit of code (such as an entire function), making it far more valuable for retrieval and much easier for the LLM to comprehend. Implementing structural, syntax-aware chunking rather than naive splitting is a primary differentiator that elevates this project above generic RAG tutorials.

### Hybrid Search and Reranking

Two additional techniques significantly elevate retrieval quality. The first is **hybrid search**. There are two ways to search: **dense search** (semantic, using vectors) finds fragments by conceptual meaning, allowing a query like "how do I authenticate" to locate login logic even if the word "authenticate" never appears in the code; and **sparse search** (lexical, keyword-based, using algorithms like BM25) matches exact terms. In code-based RAG, both are essential: semantic search captures developer intent, while lexical search is what successfully finds an exact identifier like `validate_token` when searched by name. Hybrid search **combines both approaches**, which is exceptionally valuable for source code where exact names (functions, variables, classes) are highly common and would often be missed by semantic search alone.

The second technique is **reranking**. Initial retrieval prioritizes speed and may return some less-relevant fragments mixed with the high-quality ones. A **reranker** (such as BGE-reranker) is a specialized model that takes these candidate fragments and **re-evaluates** their exact relevance to the query, moving the highest-quality matches to the very top. This implements a "two-stage retrieval" pattern: a broad, fast initial retrieval followed by a precise re-ordering of the short-listed candidates. Adding a reranker dramatically increases the quality of the context fed to your LLM, and is a staple of production-grade RAG systems.

### Evaluation: RAGAS

A key characteristic that distinguishes a disciplined engineering project from an experimental one is **evaluation**. A RAG system has many tunable parameters (chunk size, retrieval size, reranking thresholds, prompt styling), and without objective measurement, it is impossible to know if changes are helping or hurting. **RAGAS** is an evaluation framework built specifically for RAG pipelines. It quantifies metrics such as **faithfulness** (does the answer stay grounded in the retrieved context, or does the model hallucinate?), **answer relevance** (does the model actually address the user's question?), and **context precision** (did the system retrieve the correct information?). Integrating an evaluation suite shows that you apply the same engineering rigor to generative AI that you applied to traditional modeling in the MLOps project: you measure and iterate based on data rather than intuition. This is often the single most impressive element of a portfolio project.

### The API and UI

Finally, to turn this system into a usable product, **FastAPI** serves the assistant's backend as an API (reusing the same patterns of speed, typing, and automatic documentation from the MLOps project), while a framework like **Streamlit** or **Gradio** allows you to quickly build a **chat interface** where users can converse with the assistant and view citations [INDEX]. These UI tools are designed to build data and AI applications with minimal code, making them perfect for creating an interactive, demonstratable frontend without needing to build a web application from scratch.

### Tech Stack Summary

| Layer | Tool | Purpose |
|------|-------------|--------------|
| LLM Server | Ollama | Runs the model and embeddings locally |
| Code Model | Qwen3-Coder | Generation engine (with hardware-based alternatives) |
| Embeddings | nomic-embed-text / BGE-M3 (local) | Converts code and queries into vectors |
| Vector Database | ChromaDB → Qdrant | Stores and queries vectors by similarity |
| Orchestration | LlamaIndex | Coordinates the RAG pipeline |
| Chunking | Syntax-Aware Splitter | Chunks code along its structural boundaries |
| Retrieval | Hybrid (Dense + BM25) + Reranking | Finds the most relevant code fragments |
| Evaluation | RAGAS | Quantifiably measures system quality |
| API & UI | FastAPI + Streamlit/Gradio | Exposes the backend and provides a chat interface |

---

## 4. Repository Architecture

Just like in the MLOps project, a clean directory structure communicates professional standards before anyone even looks at your code. The repository should be organized to mirror the system's operational flow:

```
rag-code-assistant/
├── src/
│   ├── ingestion/      # Loading and syntax-aware chunking
│   ├── indexing/       # Vectorization and storage
│   ├── retrieval/      # Querying, hybrid search, and reranking
│   ├── generation/     # Prompt engineering and LLM generation
│   ├── api/            # API endpoints (FastAPI)
│   ├── evaluation/     # Evaluation suite with RAGAS
│   └── config.py       # Centralized configuration
├── ui/                 # Chat interface (Streamlit/Gradio)
├── tests/              # Automated tests
├── eval/               # Evaluation datasets/questions
├── data/               # Local vector storage (git-ignored)
├── docker/             # Dockerfile and docker-compose configurations
├── pyproject.toml
├── Makefile
└── README.md
```

The organizing principle of this layout is **separation of concerns matching the system's pipeline flow**. The subdirectories under `src/` correspond directly to the architectural stages we explored: `ingestion` (loading and splitting code), `indexing` (vectorization and database writes), `retrieval` (searching and reranking), and `generation` (combining prompt context and generating answers), along with `api`, `evaluation`, and centralized configuration. This alignment makes it immediately clear to any reviewer how data flows through your application.

A few specific choices are worth noting. The **user interface lives in its own directory** (`ui/`), completely decoupled from the backend logic in `src/`, as they handle distinct responsibilities. The **evaluation questions** have a dedicated home (`eval/`) and are version-controlled alongside the project, as they define how system quality is measured. The **persisted index** is stored in `data/` but, like the data artifacts in the MLOps project, is excluded from Git tracking because it is a generated, potentially large artifact. Finally, building on your previous work, you maintain a **centralized configuration** (`config.py`), **automated tests** (`tests/`), **containerization** (`docker/`), and a **Makefile** as a unified command-line interface. A major advantage of building this project after completing an MLOps project is the direct transferability of these engineering best practices, which accelerates development and brings consistent professional rigor to your portfolio.

---

## Summary

Before writing any code, the fundamental design is clear: you are building a **fully local and private RAG system for code** that indexes a repository and answers questions grounded in the actual codebase, complete with precise citations. You understand the central concept of RAG (providing context at query time rather than relying on memorization), and the guiding principle that retrieval quality—not generation—is the primary driver of performance, which clarifies where to focus your engineering effort. You have a clear, high-value differentiator (total privacy as a solution to real-world corporate governance constraints). Your technology stack is current, featuring tools selected for concrete architectural reasons while avoiding outdated models. Finally, your repository structure mirrors the operational pipeline and implements robust, enterprise-grade engineering practices.

Each design decision ensures a consistent outcome: when a peer or reviewer inspects your repository, they will see that you understand the intricacies of modern AI systems—from syntax-aware chunking and hybrid search to reranking and systematic evaluation—rather than simply knowing how to query a public API. With these architectural foundations established, you are ready to begin Phase 0 with a clear understanding of both what you are building and why.
