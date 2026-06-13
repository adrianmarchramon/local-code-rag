# Phase 8 — Packaging and Presentation

> This is the final phase of the project, and although you will write very little code, it is the one that decides whether all your work from the previous eight phases gets noticed. Remember the principle you already applied in the MLOps project: most people, and recruiters in particular, will judge your project by its README and video before they look at the code. You have built a local and private RAG code assistant that is technically excellent and rigorously evaluated; now you need to package it so that anyone can run it, and present it so that anyone can understand what you have achieved in two minutes. There is, moreover, an important distinction compared to the previous project: while you deployed that one in the cloud, this one—whose value lies precisely in running locally—is "distributed" by making it trivial to run on anyone's machine. Here, packaging well is the way you share.

**Phase objective:** Showcase the system's value.  
**Duration:** Week 8.  
**Upon completion, you will have:** The system containerized and easy to spin up with Docker, an excellent README, an architecture diagram, a demo video, and a post. In short, a finished project ready to impress.

---

## The Big Picture

This phase has two halves, just like the last one in the MLOps project. The first is **packaging**: containerizing the system with Docker so that anyone can spin it up on their machine with a single command, including the model server (Ollama). The second is **presenting**: creating the materials (README, diagram, video, post) that communicate what you have built without anyone needing to dive deep into the code.

A clarification on the approach that is worth keeping in mind, as it shows you understand the nature of the project: unlike the MLOps project, **a public cloud deployment makes no sense here**. The value of this system is that it runs locally and keeps code private; hosting it on an external server would contradict its premise, and it would also require a powerful and expensive GPU to serve the model. That is why the correct way to "deliver" this project is to package it with Docker so that it runs on the user's own infrastructure, rather than hosting it publicly. Knowing how to explain this difference demonstrates sound judgment.

---

## Step 1 — Containerize: The App and Ollama Together

Containerization directly builds on what you learned in Phase 5 of the MLOps project, with one specific addition: here, besides the app, you need to orchestrate **Ollama** (the model server). The application's `Dockerfile` follows the same multi-stage pattern with uv that you already master:

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --no-dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

What is truly specific to this project is the `docker-compose.yml`, which spins up three coordinated services: Ollama (the model server), the API, and the interface:

```yaml
# docker/docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama   # persists downloaded models
    # For NVIDIA GPU (highly recommended; uncomment if you have it):
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434   # the API communicates with the Ollama container
    volumes:
      - ../data/vector_store:/app/data/vector_store
    depends_on:
      - ollama

  ui:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

volumes:
  ollama_data:
```

It is worth understanding the specific RAG components. The **Ollama** service uses the official image and has a volume (`ollama_data`) to ensure downloaded models persist between restarts. The **API** communicates with Ollama using the service name (`http://ollama:11434`), rather than `localhost`, which is injected via the `OLLAMA_BASE_URL` environment variable. This is where a practice you brought over from MLOps pays off: just as you did with the MLflow address, it is best for your `config.py` to read this address from the environment (`os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")`), so that the same code works both locally and inside a container without modifications.

There is a step that many people forget and is worth highlighting: when spinning up the containers for the first time, the Ollama container is **empty of models**. You must download them once inside it, and they will then remain persisted in the volume:

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen3-coder:30b
docker compose exec ollama ollama pull nomic-embed-text
```

And an honest note on performance: Ollama in Docker uses the CPU by default, which works but is slow with large models. For decent performance, an NVIDIA GPU is highly recommended, which requires having the NVIDIA Container Toolkit installed and uncommenting the GPU `deploy` section. Documenting this requirement in the README is important so that anyone spinning it up knows what to expect.

---

## Step 2 — The Architecture Diagram

Before writing the README, create a **clean architecture diagram** using a tool like Excalidraw or draw.io, as it will be your central visual asset. It should show the entire system and, in particular, its two phases: **indexing** (repository → ingestion → chunking → embeddings → vector store) and **querying** (question → hybrid retrieval → reranking → LLM → answer with citations). Showing this two-stage structure, along with the fact that the entire stack runs locally, communicates the system's sophistication at a glance.

A few tips to make it effective: keep a consistent style, group the two phases clearly, mark in some way that everything is local (this is your key differentiator), and do not overload it. A good diagram at the top of the README makes an excellent first impression and is usually the first thing an engineer looks at.

---

## Step 3 — The Definitive README

This is the most important document in your entire project, and it deserves real time and effort. These are the elements it should contain, each with its own rationale:

**A catchy title and one-line description.** The first thing read: a RAG code assistant that runs 100% locally, indexes any repository, and responds with citations.

**The architecture diagram.** The visual piece you just created, placed right at the top.

**A short GIF or video of the assistant in action.** Ideally, a question being answered in the chat with its citations. Visuals capture attention and prove that it works.

**The "Why This Project" section.** The problem (understanding a codebase) and, most importantly, the **privacy angle**: many companies cannot send their code to external APIs, and this assistant solves exactly that. This shows you think in terms of business value.

**The tech stack with badges.** The tools (Ollama, Qwen3-Coder, LlamaIndex, ChromaDB, FastAPI, RAGAS) which communicate the technical scope at a glance.

**Instructions that actually work.** The `docker compose up` command (and the step to download the models), so that anyone can spin it up. Making sure they work without hidden steps is crucial.

**The design decisions section.** Just like in MLOps, this is what differentiates you the most, so give it weight. Explain the reasoning behind your choices: why Qwen3-Coder and not an outdated model like CodeLlama; why code-aware chunking (because code is not prose); why hybrid search (because of exact identifiers); why reranking (the two-stage pattern); why everything runs locally (privacy as a business value); and why evaluate with RAGAS. Remember: code shows you know how to execute, but design decisions show you know how to *think*.

**The RAGAS metrics.** Here, highlight the evaluation report from Phase 7 as objective evidence of your system's quality. This is a massive differentiator, as almost no junior RAG project includes an evaluation. Showing numbers (and, if you did it, a measured improvement) places you far ahead.

**What you learned and what you would do differently.** An honest reflection that demonstrates maturity.

Invest more in this document than you instinctively think you should: this is where the impression your project makes is won or lost.

---

## Step 4 — The Demo Video

Record a short, two-to-three-minute video showing the assistant in action. For this project, a video is especially powerful because the system is visual and interactive: you can see a conversation in progress. The chat interface from Phase 6 is the perfect setting.

A structure that works: start with a hook that makes the value clear, and here, **privacy is the best hook** ("a code assistant that understands your repository without a single line leaving your machine"). Show the assistant answering a question about a real repository and, crucially, **expand the citations** so that it is clear every response is backed by specific code—that transparency is impressive. Also show code generation aware of the project (ask it for a function in the style of the repo). Close by mentioning that everything runs locally and that you have evaluated its quality with metrics.

A few tips: practice so it flows smoothly, keep an engaging pace, narrate what is happening, and make sure the screen is easy to read. This video, showing a private code assistant responding with citations from real code, communicates in two minutes a level of sophistication that very few junior candidates can display. Upload it to YouTube (even if unlisted) and embed it in both the README and your post.

---

## Step 5 — The Post

Write a post sharing the project, as it multiplies its visibility and demonstrates your ability to communicate. The key, just like in MLOps, is to tell the **story and the reasoning**, not just announce that you built something.

A good structure: open with the problem (for example, that the best AI tools for coding require sending your code to the cloud, which is banned in many companies). Tell what you built, focusing on what makes it special: a 100% local and private code RAG. Share one or two design decisions with their reasoning (code-aware chunking or hybrid search demonstrate that you understand the difficult parts of RAG). Mention that you evaluated it with metrics. Close with links to the repository and the video, accompanied by visual material (the diagram or a screenshot of the chat showing citations). The goal is not to brag, but to show how you think and help your work reach a wider audience.

---

## Step 6 — Polish and Review

Before considering the project complete, spend a final effort polishing it. Run the **clean environment test**, which is especially important here: on a clean machine or folder, verify that `docker compose up` (plus downloading the models) spins up the system and works without hidden steps. Verify that **all links work** (the demo, the video, the images). Review the **spelling and grammar** of the README and the post. And, if possible, ask someone to look at the project and tell you if they understand what it does within a couple of minutes.

The question that should guide this review, and which is the success criterion of this phase, is: **could someone understand the entire project in two minutes using only the README and the video, and spin it up with Docker?** If the answer is yes, you are done.

---

## Verification: The "Done" Criterion

The phase, and with it the project, is finished when the following are met:

- [ ] The system is containerized, with a `docker-compose.yml` that spins up the app, the UI, and Ollama.
- [ ] A clean architecture diagram exists, showing both phases (indexing and querying) and its local nature.
- [ ] The definitive README is complete, with special attention to design decisions and RAGAS metrics.
- [ ] There is a 2-3 minute video with the assistant responding with citations, using privacy as a hook.
- [ ] There is a post sharing the story and the reasoning behind the project.
- [ ] The project is polished: it spins up with Docker from scratch, contains correct links, and has careful copywriting.
- [ ] **The key test:** someone can understand the project in two minutes using the README and the video, and spin it up with Docker.

The key test is the culmination of your effort: if someone visiting your project for the first time can, in two minutes, understand what it does, see that it works, grasp its sophistication, and also run it on their own machine with a single command, you have achieved the ultimate goal—which was not just to build a great system, but to build one that impresses and that anyone can use.

---

## Closing: What You Have Built

With this phase, you have finished the entire project, and it is worth pausing to look at what you have achieved. You have built an **end-to-end, local, and private RAG code assistant**: it indexes any repository (with filtered ingestion and code-aware chunking), makes it searchable by meaning (embeddings and vector database), retrieves the correct snippets with advanced techniques (hybrid search and reranking), generates grounded answers with citations (local LLM with grounding), all accessible from a chat interface, rigorously evaluated (RAGAS), and packaged to run on any machine. And none of it sends a single line of code outside.

Beyond the tools, what you have demonstrated is a mastery of the **most in-demand AI architecture today**, along with its challenging aspects: understanding that code is not ordinary text and requires careful chunking, that retrieval is key and combines semantic and lexical approaches, that a good RAG grounds and cites its sources, and that quality is measured, not guessed. This is precisely what is lacking in junior profiles, who often stop at "calling an LLM API."
