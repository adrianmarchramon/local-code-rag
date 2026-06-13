# RAG for Code — Closing the Roadmap: Mistakes, Improvements, and Resources

> With all nine phases complete, you now have the system built, evaluated, and presented. This document covers what surrounds the project and elevates it: the mistakes to avoid (lessons that span across the entire body of work), ways to go further if you want to keep growing it, resources to dive deeper, and a final summary of the journey. The checklist for the final README was already thoroughly developed in Phase 8, so it is not repeated here.

---

## Common Mistakes to Avoid

These mistakes span across the entire project, and each one holds one of the lessons that separate a serious code RAG from a generic tutorial. Internalizing them is, in large part, what this project is all about.

**Blindly chunking code.** This is the most serious mistake and the easiest to make, because it is what almost every RAG tutorial does. Splitting code every N characters cuts functions in half, separates a signature from its body, and produces meaningless fragments whose embeddings capture nothing coherent. As we saw in Phase 2, this sabotages retrieval from the very root, no matter how good everything else is. The solution is syntax-aware chunking, which splits by logical units (functions, classes) while respecting the code's structure. It is the difference between a code RAG that works and one that does not.

**Using only vector search.** Semantic search is excellent for conceptual questions, but it has a blind spot for exact identifiers (function names, variables), which are omnipresent in code. Searching for `validate_token` by its name can fail with purely vector-based search. The solution is hybrid search, which combines semantic with lexical (BM25) search, so that each covers what the other misses. In a code RAG, this is not a luxury, but a necessity.

**Not citing sources.** An answer without sources is a black box: the user has to trust the assistant blindly, and in code, that is dangerous. Always showing which snippets back up each answer (file, symbol, lines) makes the system transparent and verifiable, which is the hallmark of a well-built RAG. Furthermore, it gives purpose to all the care you put into the metadata starting from ingestion.

**Not evaluating, judging everything "by eye".** Without metrics, you do not know if your system works or if your changes make it better or worse; you are fine-tuning blindly. As we saw in Phase 7, evaluating with RAGAS turns guesswork into engineering: you measure, you make changes, you measure again, and you keep what is proven to improve the system. The absence of evaluation is what reveals a lack of rigor in a project; its presence is what distinguishes it most.

**Letting the model make things up.** The main flaw of an LLM is hallucinating: when it does not know something, it invents a plausible answer. In a code assistant, this means describing a function that does not exist. The solution, seen in Phase 5, is a prompt that forces grounding: answering only based on the retrieved context and acknowledging when it cannot answer, instead of inventing. An assistant that says "I cannot answer with the available code" is infinitely more reliable than one that improvises.

**Sending data to an external API.** This is the error that would break the very premise of the project. If at any point (embeddings, the LLM, or even the evaluation judge) you send code to an external service, you lose the complete privacy that is your main differentiator. The solution is to keep everything local: local embeddings, local LLM via Ollama, and, as we saw, even a local RAGAS judge. End-to-end privacy, with no exceptions, is what makes the system valuable.

**Reindexing everything on every execution.** Vectorizing an entire repository is slow; rebuilding it every time you start is absurd and unfeasible for large repos. The solution, seen in Phase 3, is to persist the index and reuse it, and, as an improvement, implement incremental indexing (reindexing only what changes). Properly handling persistence is a sign of good engineering judgment.

The common thread of all these mistakes is the same: each marks the difference between treating RAG as an exercise in "calling an LLM" and treating it as the engineering of a true retrieval and generation system.

---

## Stretch Goals: If You Want to Go Further

Once you have the complete system up and running, these extras take it to another level and demonstrate an even deeper mastery. They are not necessary for the project to impress, but each is a recognizable component of advanced code assistants, and adding one or two (well-built and documented) distinguishes you even further.

**Incremental indexing.** Instead of reindexing the entire repository when it changes, you detect which files have been modified (for example, by comparing hashes or using the git history) and reindex only those: you delete their old fragments from the vector database and re-chunk and vectorize only the changed files. This keeps the index fresh cheaply, which is essential for large repositories or active development where a full reindex every time would be impractical. This connects directly with the ingestion and indexing phases, and takes advantage of ChromaDB's ability to delete and update by ID. It is of medium difficulty and proves that you think of the system as operating continuously, not just as a one-off.

**Agentic assistant.** Instead of a single retrieve-and-generate cycle, you give the model the ability to reason about what to search for, perform multiple chained searches, and combine the findings, bringing it closer to a code agent. For example, when faced with "how does a request flow through the system?", the assistant could search for the entry point, then search for the functions that this entry point calls, and trace the path before synthesizing the response. This is valuable because complex questions often require multi-hop retrieval, which a single search cannot cover. It is of high difficulty and represents one of the most impressive and current directions in the world of applied AI; LlamaIndex offers agent abstractions to build this.

**Multi-repository support.** Indexing and querying multiple repositories simultaneously, with filtering by project. Real assistants often span more than one codebase (a service and its libraries, for example). You would add repository metadata to each fragment and allow filtering the search by project, leveraging the metadata filtering you already built in Phase 4. It is of medium difficulty and expands the system from a "single-repo assistant" to an "organization-wide assistant."

**Code generation with verification.** When the assistant generates new code, you automatically verify it (by trying to run it, or running it through a linter or a type-checker) and, if it fails, you feed the errors back to the model for correction. This closes a quality loop: the code delivered by the assistant does not just "look" correct, but has at least been verified as syntactically valid. This is a self-correction pattern of medium-high difficulty, demonstrating a concern for the reliability of the generated code that goes beyond the ordinary.

**Model comparison with RAGAS.** Evaluate several local LLMs (Qwen3-Coder, DeepSeek-Coder, GPT-OSS) with RAGAS over your evaluation dataset and document which one performs best for your use case. This is a particularly natural extension with a good value-to-effort ratio, as you already have the entire evaluation infrastructure from Phase 7: you just need to swap the model and measure again. It allows you to select the model based on data rather than intuition, and makes for excellent documentation in your README. It is of medium difficulty and demonstrates once again the rigor of measurement.

**Editor integration.** A VS Code extension that queries your assistant directly from the editor, calling the API you built. This transforms the project into a real development tool, used right where developers work. It is of medium-high difficulty (requiring you to learn VS Code extension development), but is enormously impressive, as it takes your system from a "chat screen demo" to a "tool I use daily."

Beyond these six, there are other natural directions: token-by-token **streaming** of responses (which vastly improves the experience with slow local models), **conversation with memory** to allow follow-up questions, or a **cache** for repeated queries. However, with just one or two of the above, well-implemented and documented, you will already have a project that clearly stands out.

---

## Learning Resources

To delve deeper into RAG and local LLMs beyond this project, these are the most valuable resources, along with a note on what each provides.

**LlamaIndex Documentation.** This is the primary reference for building RAG systems and the central tool of your project. It deeply covers indexing, retrieval (including hybrid and reranking), agents, and evaluation, with plenty of examples. Whenever you want to expand any part of the pipeline, this is the first place to look. LangChain's documentation is a valid and also very comprehensive alternative.

**Ollama Documentation.** For everything related to serving local models: how to download them, the available names and sizes (which change frequently, making it the go-to source to choose a model), and execution parameters like the context window. Its model library is where you can see what is currently available in this fast-moving field.

**RAGAS Documentation.** For rigorously evaluating RAG systems. It is highly recommended to keep it handy, especially because RAGAS frequently changes its API; thus, it is the reference for current metric names and usage, as well as for configuring custom judges like the local one you used.

**Qdrant and Chroma Documentations.** Their guides cover vector search, hybrid search, metadata filtering, and reranking with solid examples. These are useful if you want to dive deeper into the vector database or migrate from Chroma to Qdrant for a more robust setup.

**The Foundational Concept.** To understand the foundations, the original paper that introduced RAG (Lewis et al., 2020) explains the core concept of combining retrieval and generation, which remains at the heart of everything you have built, no matter how much the techniques have evolved.

Additionally, there is a set of **concepts** that are well worth mastering, as they underpin the project and are most frequently asked about: why RAG outperforms fine-tuning for proprietary data; why chunking is so decisive (and why code requires syntax-aware chunking); the difference between dense, sparse, and hybrid search; the role of reranking and the distinction between bi-encoders and cross-encoders; how RAG quality is evaluated and what each metric measures; and why the context window matters when injecting retrieved code. If you can explain these fluently, you demonstrate an understanding that goes far beyond simply having followed a recipe.

---

## Milestone Summary

As a recap of the path traveled, this is the complete project map, phase by phase:

| Week | Milestone | What It Demonstrates |
|------|-----------|----------------------|
| 1 | Environment + Ollama + local models | Foundations and mastery of local LLM deployment |
| 1-2 | Repository ingestion and filtering | Knowing what to index and what to discard |
| 2 | Syntax-aware code chunking | The piece that most distinguishes a code RAG |
| 3 | Embeddings + vector database | Making the repository searchable by meaning |
| 4 | Hybrid retrieval + reranking | Retrieval quality, which is the most critical factor |
| 5 | Grounded generation with citations | Grounding and transparency: a reliable assistant |
| 6 | API + chat interface | The system transformed into a usable product |
| 7 | Evaluation with RAGAS | Rigor: measuring quality, not guessing it |
| 8 | Packaging and presentation | The project ready to run and impress |

---

## A Final Reflection

You saw the project's wrap-up itself at the end of Phase 8; here it is enough to capture the underlying idea. What you have built is not "a script that calls an LLM," but a complete RAG system with all its difficult parts resolved: chunking that respects the code's structure, retrieval that combines semantic and lexical search and refines it with reranking, grounded generation that cites its sources, rigorous evaluation that measures quality, and all of this 100% local and private. This understanding of modern AI architectures, with their real-world complexity, is precisely what is lacking in junior profiles.

