# Design decisions: local-code-rag

| Date (UTC) | Phase | Scope |
|---|---|---|
| 2026-11-24 | 0: Setup and fundamentals (Steps 1–7) | Use case, model, privacy argument, corpus, stack. Phases 1+ are planned, not built. |
| 2026-11-25 | 1: Ingestion (Steps 1–7) | Filters, loader, metadata, languages, tests and the key test (sections 7–9). Phases 2+ are planned, not built. |

## 0. How to read this document

Every figure or factual claim carries its provenance:

| Tag | Meaning |
|---|---|
| `[M]` | Measured on this host on 2026-11-24 (UTC) with a command listed in the Appendix |
| `[H]` | From the Phase 0 Steps 2–5 handoff (measured then, not re-measured) |
| `[S]` | From the Phase 0 session record (transcript), not re-measured; a time is shown only where one was recorded |
| `[D: file §]` | Stated by a project document in `project_context/` (attributed, not verified) |
| `[U]` | Stated by the project owner |
| `[E]` | Estimate computed from the figures above; the calculation is shown and assumptions are labelled |

In sections 7–9 (Phase 1), `[M]` means measured on this host on 2026-11-25 (UTC).

`TODO(user)` marks a sentence that needs the owner's own judgment, or a fact nobody recorded. Each one carries the question to answer.

## 1. Use case

| Aspect | Decision | Source |
|---|---|---|
| What | Answer natural-language questions about a codebase ("where are payments handled?", "how is the data layer structured?") and generate code that follows the project's patterns, with citations to files, symbols and lines | [D: rag_generalRoadmap.md §2], [D: rag_fundamentals.md §2] |
| For whom | Developers facing code they did not write, or too large to keep in their head: comprehension and onboarding | [D: rag_fundamentals.md §2] |
| Problem | Commercial code assistants send code to external servers; organizations under strict data-governance policies (banks, healthcare, defense, valuable proprietary code) cannot use them | [D: rag_fundamentals.md §2 "Why Total Privacy…"] |
| Constraint | Fully local at runtime: model, embeddings and vector database run on the host | [D: rag_fundamentals.md "The Differentiating Angle"] |
| Quality lever | Retrieval dominates answer quality, so most effort goes into chunking, embedding, search and reranking | [D: rag_fundamentals.md "Retrieval is Paramount"] |

TODO(user): Which concrete situation motivated this assistant for you? Answer in one or two sentences, in your own words.

## 2. Model choice and rationale

**2.1 Hardware** (no usable GPU, so all inference runs on CPU)

| Item | Value | Source |
|---|---|---|
| OS | Ubuntu 26.04.1 LTS, Linux 7.0, x86_64 | [M] |
| CPU | Intel Core i7-1355U, 10 cores / 12 threads, AVX2 + AVX-VNNI | [M] |
| RAM | 14.81 GiB total; available memory ranged 2.49–7.74 GiB during the work, depending on other applications | [M] |
| GPU | Integrated Iris Xe only; Ollama reports `library=cpu`, `total_vram=0` | [M] |
| Free disk | 311 GB at provisioning; 306 GB after the models | [H], [M] |

**2.2 Code model: `qwen2.5-coder:7b`**

| Field | Value | Source |
|---|---|---|
| Digest | `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364` | [M] |
| Size / format | 4,683,087,561 B; qwen2, 7.6B parameters, Q4_K_M | [M] `ollama show` |
| Maximum context | 32768 tokens | [M] |
| What the docs say | Qwen3-Coder is the 2026 reference for local code models; CodeLlama is outdated | [D: rag_fundamentals.md §3], [D: rag_generalRoadmap.md §3] |
| What Phase 0 prescribes | `qwen3-coder:30b` with about 24–32 GB of memory, or a 7B model (about 5 GB) with about 16 GB; "If in doubt, start with the 7B model" | [D: rag_phase0.md Step 3] |
| What exists today | The Ollama library lists 9 Qwen3-Coder tags. The smallest is `qwen3-coder:30b` / `:latest` / `:30b-a3b-q4_K_M` at 19 GB. None is 7–8B or ≤16 GB, and none is named cloud (read 22:56:41–22:57:29Z). | [M] |
| Why not Qwen3-Coder | 19 GB of weights do not fit in 14.81 GiB of RAM. The 7–8B Qwen3-Coder variant the roadmap mentions does not exist in the library on that date. | [M], [D: rag_generalRoadmap.md §3] |
| Decision | `qwen2.5-coder:7b`, the 7B option Phase 0 documents for hosts with about 16 GB | [U], [D: rag_phase0.md Step 3] |

**2.3 Embedding model: `nomic-embed-text:latest`**

| Field | Value | Source |
|---|---|---|
| Digest | `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` | [M] |
| Size / format | 274,302,450 B; nomic-bert, 137M parameters, F16 | [M] |
| Dimension | 768 reported by `ollama show`; 768 observed by `make verify-local` | [M] |
| Context | Trained context 2048 tokens. Its built-in `num_ctx 8192` is capped: `/api/ps` reports 2048 when the model is loaded. | [M], [H] |
| Implication | Phase 2 chunks must stay under 2048 tokens. Phase 2's `max_chars` 1500 is about 375–500 tokens at 3–4 chars/token, which is within the cap. | [D: rag_phase2.md Step 2], [E] |

**2.4 Context window: the derived model `qwen2.5-coder-7b-rag`** (`e8af2b0938df`, built by `make models` from `ollama/Modelfile`)

| Item | Value | Source |
|---|---|---|
| Modelfile | `FROM qwen2.5-coder:7b` + `PARAMETER num_ctx 16384`. It inherits template, system prompt and weights (store grew by 1,259 B). | [M] |
| Deviation | Phase 0 Step 5 and Phase 5 use 32768; this project uses **16384** | [D: rag_phase0.md Step 5], [D: rag_phase5.md "The Big Picture"], [U] |
| Default without it | 4096: the server log reads `default_num_ctx=4096`, and `/api/ps` shows 4096 for the base model | [M] |
| Resident memory | Base model at 4096: 5.06 GB. Derived model at 16384: 5.94 GB (+0.88 GB). | [M], [H] |
| KV-cache estimate | About 56 KiB per token (2 × 28 layers × 4 KV heads × 128 dims × 2 B, from `ollama show` metadata): 0.875 GiB at 16384, 1.75 GiB at 32768 | [E] |
| Truncation | A 4937-token needle prompt: the derived model read 4936 tokens and recalled the codename. The base model (default 4096) was truncated to 2050 tokens (`limit=2050 keep=4`) and lost it. | [H] |
| Precedence | Per-request `num_ctx` > Modelfile `PARAMETER num_ctx` > server default. `/api/ps` read 16384 (derived), 8192 (derived + request), 8192 (base + request) and 4096 (base). | [H] |
| Why 16384 | Measured basis: it holds the measured needle prompt; it costs 0.88 GB of resident memory over 4096; 32768 would need about 1.75 GiB for the KV cache alone [E], on a host with 2.49–7.74 GiB available [M]. | see sources |

TODO(user): In your own words: why 16384, and not 8192 or 32768?

**2.5 Measured baseline** (figures as measured; not averaged or rounded)

| Metric | Value | Conditions | Source |
|---|---|---|---|
| Generation | 4.29–6.0 tok/s | Derived model, 50 generated tokens, `make verify-local` | [H] |
| Generation, base model | 4.37 tok/s | Base model, same check | [M] |
| Cold load | 5.3–6.6 s | Code model, from disk | [H] |
| Prompt evaluation | About 11.8 tok/s (1833 tokens in 155.47 s) | Base model, 4096 context | [H], [M] |
| Needle prompt | Derived: 478.5 s for 4936 tokens. Base: 181.1 s for 2050 tokens. | `keep_alive=0`, unique nonce per run | [M] |
| Embedding | 326–652 ms per 2-input batch, including 269–535 ms of load | `make verify-local` | [H], [M] |
| Tokenizer ratio | 3.43 chars/token (16,934 chars → 4,936 tokens) | qwen2.5 tokenizer, synthetic English prose | [M] |

**2.6 Trade-offs.** The docs acknowledge that local models on consumer hardware are less capable than frontier cloud models, and argue that for sensitive codebases the privacy guarantee outweighs those last points [D: rag_fundamentals.md "The Differentiating Angle"]. On this host, the larger cost is time. All of the following are `[E]`:

| Quantity | Calculation | Result |
|---|---|---|
| Prompt-eval cost per 1000 tokens of context | 1000 ÷ 11.8 tok/s | 84.7 s |
| Retrieved context per query | 5 chunks (`RERANK_TOP_N`, [D: rag_phase4.md "The Big Picture"]) × 1500 chars (`max_chars`, [D: rag_phase2.md Step 2]) = 7,500 chars, ÷ 3–4 chars/token | 1,875–2,500 tokens |
| Prompt eval per query | (1,875–2,500 + 200 assumed overhead tokens) ÷ 11.8 | 176–229 s |
| Answer generation | 300 assumed tokens ÷ 6.0–4.29 tok/s | 50–70 s |
| Per query | Sum of the two | 226–299 s (3.8–5.0 min), above `REQUEST_TIMEOUT` = 180 s |
| One RAGAS run | 10–30 questions [D: rag_phase7.md Step 3] × (pipeline + about 9 assumed judge calls, each re-reading the context) | About 5.4–7.1 h for 10 questions; about 16.3–21.4 h for 30 |

The assumptions (200 overhead tokens, 300-token answers, 9 judge calls of 100 output tokens each) are not measured. Prompt caching or shorter contexts would lower these figures; neither is verified.

TODO(user): Is a 4–5 minute answer acceptable for your demo? If not, which lever would you pull first: fewer or shorter chunks, a smaller model, or a longer timeout?

**2.7 Alternatives considered**

| Alternative | Source | Status |
|---|---|---|
| `qwen3-coder:30b` (MoE) | [D: rag_phase0.md Step 3] | Rejected: 19 GB > 14.81 GiB RAM [M] |
| GPT-OSS 20B, DeepSeek-Coder, Codestral, Devstral | [D: rag_fundamentals.md §3] | Not benchmarked on this host |
| `qwen2.5-coder:3b` (agreed fallback) | [U] | Not downloaded; not benchmarked on this host |
| BGE-M3, GTE-Qwen2 (embeddings) | [D: rag_fundamentals.md §3] | Not benchmarked on this host |
| CodeLlama | [D: rag_fundamentals.md §3] | Excluded: outdated per the docs |

## 3. The privacy argument

**3.1 Why local.** Many companies are prohibited from sending code to external APIs, for intellectual-property, compliance or data-governance reasons. An assistant whose model, embeddings and database run on the host can be used where cloud tools are banned [D: rag_fundamentals.md "The Differentiating Angle", §2].

**3.2 Scope of the guarantee**

| | Statement |
|---|---|
| Covered | The assistant's runtime (indexing and querying): no repository data, prompt or embedding leaves the host. Verified so far for the Phase 0 runtime (Ollama plus `make verify-local`). |
| Covered | Phase 1 ingestion: the loader CLI, the key test and every test run under the network guard, with 0 violations [M] (sections 8–9) |
| Not covered | The one-off provisioning downloads listed in 3.3 |
| Not covered | The tools used to develop this repository; this document makes no claim about their network behaviour |
| Not covered | The security of the host itself (operating system, other processes, physical access) |

**3.3 Network surface.** Runtime: none.

| Operation | When (UTC) | Destination | What leaves the host | Phase | Evidence |
|---|---|---|---|---|---|
| Ollama installation (official script) | TODO(user) | ollama.com | Download requests | Before Phase 0 | [U] |
| `git ls-remote` on this repo's origin (unplanned, disclosed at the time) | 2026-11-24 | github.com (SSH) | Ref-listing request | 0, before Step 1 | [S] |
| uv installer (run by the owner) | 2026-11-24 | astral.sh | Download requests | 0 / Step 1 | [S] |
| CPython 3.11.18 (29.4 MiB) | 2026-11-24 | python-build-standalone release (GitHub) | Download requests | 0 / Step 1 | [S] |
| PyPI resolve + 33 wheels (39.21 MiB) | 2026-11-24 | pypi.org, files.pythonhosted.org | Package names and versions | 0 / Step 1 | [S] |
| Web searches: "ollama library qwen3-coder tags sizes", "ollama library qwen2.5-coder 7b 3b tags" | 19:03:32–19:03:33Z | Web search service | The two query strings; no repository data | 0 / Block A | [S] |
| Public page reads (Ollama library, up to 5 pages) | 19:03:46Z | ollama.com | HTTP GETs; no repository data | 0 / Block A | [S] |
| Public page reads (library tags, context-length docs, FAQ: 5 pages) | 21:20:44Z | ollama.com, docs.ollama.com | HTTP GETs; no repository data | 0 / Steps 2–5 | [S], [H] |
| `ollama pull qwen2.5-coder:7b` | 21:28:42–21:35:52Z | Ollama registry | Pull requests | 0 / Step 3 | [H] |
| `ollama pull nomic-embed-text` | 21:35:59–21:36:29Z | Ollama registry | Pull requests | 0 / Step 3 | [H] |
| `uv add --offline httpx` | 2026-11-24 | none | Nothing (no download) | 0 / Step 4 | [H] |
| Public tag list read (`qwen3-coder`) | 22:56:41–22:57:29Z | ollama.com | One HTTP GET; no repository data | 0 / Step 7 | [M] |
| Corpus `git clone` (run by the owner) | 23:06:36Z (from the reflog) | github.com (HTTPS) | Clone request | 0 / Step 6 | [M] |
| Public documentation reads (PyPI JSON for `llama-index` and `llama-index-core`, three `llama-index-core` source files, Chroma docs: 6 URLs) | 2026-11-25 13:27:45–13:28:55Z | pypi.org, raw.githubusercontent.com, docs.trychroma.com | HTTP GETs; no repository data | 1 / preparation | [S] |
| `uv add --offline pathspec` (previewed first on scratch copies of `pyproject.toml` and `uv.lock`) | 2026-11-25 15:28:59Z | none | Nothing (no download; 1.1.1 was already in the uv cache) | 1 / Steps 1–3 | [M] |
| `uv add --no-sync llama-index-core` on scratch copies of `pyproject.toml` and `uv.lock` (resolution only) | 2026-11-25 16:37:30–16:37:32Z | pypi.org, files.pythonhosted.org | Package names and versions, uv's User-Agent | 1 / Steps 4–5 | [M] |
| `uv add llama-index-core` (46 new packages, all wheels, 53,381,296 B) | 2026-11-25 16:42:05–16:42:10Z | pypi.org, files.pythonhosted.org | Package names and versions, uv's User-Agent | 1 / Steps 4–5 | [M] |
| NLTK data download attempted by `llama-index-core` on the first use of its stopwords; blocked by `NetworkGuard` before name resolution (see 8.1, venv permissions) | 2026-11-25 about 16:44Z | raw.githubusercontent.com (never reached) | Nothing | 1 / Steps 4–5, test probe | [M] |
| `uv sync --locked --offline --reinstall` in copy mode | 2026-11-25 16:50:13Z | none | Nothing (from the uv cache) | 1 / Steps 4–5 | [M] |
| Phase 1 Steps 6–7 and closure: no download, no installation, no documentation read | 2026-11-25 | none | Nothing | 1 / Steps 6–7 | [M] |

**3.4 Controls matrix**

| Control | What it covers | What it does NOT cover | Evidence |
|---|---|---|---|
| Loopback-only listener | Other hosts cannot reach the Ollama API | Outbound connections made by Ollama | `ss -ltn`: 127.0.0.1:11434 only [M] |
| `OLLAMA_NO_CLOUD=1` | Cloud models and web search in the server | Sign-in (the Ollama FAQ does not say it blocks it) | systemd drop-in `10-no-cloud.conf`; log `cloud disabled: true` [M] |
| No account link | No `ollama signin` was ever run | Not checkable read-only (the CLI offers only `signin`/`signout`) | [U] |
| `src/config.py`, fail-closed | Rejects a non-loopback `OLLAMA_BASE_URL` (unless the host is in `LOCAL_ALLOWED_HOSTS`), cloud tags and malformed values with `ConfigurationError` at import | Code that bypasses `src.config` | `tests/test_config.py` [M] |
| Ollama client wrapper | Explicit host and finite timeout; no redirects; ignores proxy variables and `OLLAMA_HOST` | Other HTTP clients added in later phases | `tests/test_ollama_client.py` [M] |
| Network guard | Blocks and records non-loopback connections and name lookups in the Python process, for every test (autouse) and for `make verify-local` | The Ollama server process, subprocesses, C extensions that bypass Python's `socket` | `make test`: 0 violations; `verify-local`: `blocked_attempts=0` [M] |
| Static URL test | Non-loopback URLs in string literals under `src/` | URLs built at runtime; documentation | `tests/test_no_external_urls.py` [M] |
| `.venv` without hardlinks or group write (`link-mode = "copy"`, `umask 022` in `make install`) | `llama-index-core` uses the NLTK and tiktoken data bundled in its wheel, so their first use needs no download | A `.venv` created another way (plain `uv sync` with a group-writable umask, other tools) | `tests/test_ingestion_privacy.py`: load, tokenizer, stopwords and punkt in a fresh interpreter, 0 violations, 0 files written [M] |
| `models_present` check | Rejects models with no local weights (remote/cloud) | — | `tests/test_health_check.py` [M] |
| Socket sampling | Ollama's TCP/UDP sockets to non-loopback peers | Connections shorter than the 1 s interval; lookups through the local resolver (127.0.0.53) | 579 samples, 0 non-loopback [H] |
| Offline test | Generation and embedding with the network down | — | 22:25Z, exit 0; `nmcli networking connectivity` → `none`; `ip route show default` → empty [U] |

**3.5 Reproducing the audit:** see the Appendix, items A1–A5.

## 4. Corpus selection

**Why this repository.** Indexing the owner's MLOps project demonstrates both projects at once. Because the owner knows that code inside out, it is easy to judge whether answers are right. The system stays uncoupled from any specific repository: Phase 1 receives the path as a parameter [D: rag_fundamentals.md §2 "Which Codebase to Index"], [D: rag_generalRoadmap.md §2].

| Identity | Value | Source |
|---|---|---|
| Remote (public) | `https://github.com/adrianmarchramon/fraud-mlops-pipeline.git` | [U], [M] |
| Local path | `~/repos/project-to-index` (outside this repository; the default `REPO_PATH` in Phase 1) | [M], [D: rag_phase1.md Step 4] |
| Branch / HEAD | `main` / `5c288080c88f0bfec435af0093a5155baaf7f343`, committed 2026-08-30T17:48:36Z | [M] |
| `git describe` | `phase-9-complete-3-g5c28808`; 145 commits, not shallow, clean tree | [M] |
| Freeze policy | HEAD SHA recorded; the clone is never modified. Reproduce with `git checkout <sha>`. | [U] |

**Content, with Phase 1's extension map applied as written** [D: rag_phase1.md Steps 3–4]. There are 161 tracked files (equal to a filesystem walk), 9,208,460 B, 30,785 lines of text and 6 binaries [M].

| Language | Files | Lines | Bytes | Lines per file (median / p90 / max) | Bytes per file (median / p90 / max) |
|---|---|---|---|---|---|
| markdown | 59 | 8,762 | 563,386 | 93 / 346 / 499 | 5,998 / 22,050 / 41,526 |
| python | 50 | 6,812 | 262,545 | 75 / 341.3 / 798 | 3,077 / 14,343 / 30,753 |

| Finding | Value | Source |
|---|---|---|
| Phase 1 would load | 109 files: 59 `.md` (45 `docs/`, 13 `project_context/`, 1 at the root) and 50 `.py` (19 `src/`, 18 `deploy/model/`, 6 `tests/`, 4 `pipelines/`, 2 `scripts/`, 1 at the root). 52 dropped as unmapped. | [M] |
| Not in Phase 1's map | `.yaml`/`.yml` 11, `.json` 6, `.lock` 3, `.toml` 2, `.txt` 2, `MLmodel` 2, `.ipynb` 1 (552,936 B), `Dockerfile` 1, `Makefile` 1; binaries `.mp4`, `.pdf`, `.png`, `.pkl` ×2, `.joblib` | [M] |
| Noise neither filter covers | `deploy/model/` (38 files, 1.48 MB; 18 `.py`, 8 byte-identical to `src/` files by git blob ID), `project_context/` (13 `.md` files), `docs/` binaries, `notebooks/`, `reports/`, `data/`, `.dvc/` | [M] |
| `.gitignore` files | Root present, matching 0 tracked files. 3 nested (`.dvc/`, `data/raw/`, `data/monitoring/`), which Phase 1 does not read. | [M] |
| Pointers | DVC: `dvc.yaml`, `dvc.lock`, `.dvc/config`, `data/raw/creditcard.csv.dvc` (the dataset is not in Git). No submodules, no `.gitattributes`, no LFS pointers. | [M] |
| Secret-like filenames | None (one `.env*` file ends in `.example`; not opened) | [M] |

**Implications** (from the numbers; the decisions belong to their phases)
- Phase 1: whether to index `docs/` and `project_context/` (prose outnumbers code: 59 vs 50 files, 8,762 vs 6,812 lines), whether to exclude `deploy/model/` (duplicates), and whether to extend the map (YAML, TOML, notebooks).
- Phase 2: Python is the only code language, so one tree-sitter grammar plus Markdown handling suffices. Fourteen `.py` files exceed 6 KiB and 8 exceed 8 KiB. At an assumed 3–4 bytes/token [E], 2048 tokens is about 6–8 KiB, so 8–14 Python files cannot be embedded whole and chunking is mandatory.
- Phase 7: 109 loadable files and about 15.6k lines make a small, monolingual corpus. Re-indexing is cheap, but differences between retrieval methods may be hard to see.

TODO(user): Which questions should the assistant answer well on this corpus to convince you? These seed `eval/questions.yaml`.

## 5. Stack decisions

| Component | Choice | Why | Alternative considered | Status |
|---|---|---|---|---|
| Model server | Ollama 0.34.0 | Local API serving generation and embeddings [D: rag_fundamentals.md §3] | — | Decided and verified in Phase 0 [M] |
| Code model | `qwen2.5-coder:7b` → `qwen2.5-coder-7b-rag` (16k) | Fits the measured hardware (§2) | Qwen3-Coder 30b, GPT-OSS 20B, DeepSeek-Coder, Codestral, Devstral | Decided and verified in Phase 0 [M] |
| Embeddings | `nomic-embed-text` (768-d) | Lightweight and local [D: rag_fundamentals.md §3] | BGE-M3, GTE-Qwen2 | Decided and verified in Phase 0 [M] |
| Vector database | ChromaDB → Qdrant | Chroma is embedded with zero config; Qdrant is self-hosted, more robust, with native hybrid search [D: rag_fundamentals.md §3] | Milvus, Weaviate, pgvector; managed services excluded (not local) | Planned: Chroma in Phase 3 [D: rag_phase3.md]; Qdrant optional in Phase 8 ("if you use it") [D: rag_generalRoadmap.md §5] |
| Orchestration | LlamaIndex | Built for indexing and retrieval; ships `CodeSplitter` [D: rag_fundamentals.md §3] | LangChain | Planned, to be validated in Phases 2–5 |
| Chunking | Syntax-aware `CodeSplitter` (tree-sitter) | Cuts at function and class boundaries [D: rag_fundamentals.md §3], [D: rag_phase2.md Step 2] | Fixed-size splitting (rejected by the docs) | Planned, Phase 2 |
| Hybrid search | Vector + BM25, fused with reciprocal rank fusion | Exact identifiers need lexical matching [D: rag_fundamentals.md §3], [D: rag_phase4.md "The Big Picture"] | Vector search only | Planned, Phase 4 |
| Reranking | Cross-encoder `BAAI/bge-reranker-base` | Two-stage retrieval [D: rag_phase4.md "The Big Picture"] | No reranker | Planned, Phase 4; the model download will be a new provisioning operation |
| Evaluation | RAGAS, 4 metrics, local LLM as judge | Measure instead of guessing; a local judge keeps privacy [D: rag_phase7.md Step 4] | RAGAS default (OpenAI) judge, rejected as not local | Planned, Phase 7 |
| API and UI | FastAPI + Streamlit | Serves the assistant; chat with citations [D: rag_phase6.md "The Big Picture"] | Gradio | Planned, Phase 6 |
| Quality tooling | uv, ruff, mypy `--strict`, pytest, pre-commit, Makefile | Reproducible environment [D: rag_phase0.md Step 1] | — | Decided and verified (`make check`) [M] |

Ruff configuration: this repository keeps its own settings (line length 100, py311, rules `E W F I B UP SIM N RUF PT S BLE T20`). The MLOps repository uses line length 88, py312, rules `E F I N UP B D S`, and ignores `D401 N803 N806`. This repository keeps the stricter set on purpose, because `BLE` and `T20` enforce the no-broad-except and no-print guardrails [M], [U].

Update after Phase 1 (2026-11-25): `llama-index-core` 0.14.26 and `pathspec` 1.1.1 are installed and used by ingestion only (7.2, 8.1); the rest of this table is unchanged.

## 6. Known gaps and open questions

| Gap | Status | Procedure to reproduce (not run) |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` vs the Modelfile | Not verified | The owner adds `Environment="OLLAMA_CONTEXT_LENGTH=8192"` in a second drop-in and restarts Ollama. Load the derived model with no per-request `num_ctx` and read `context_length` from `/api/ps` (16384 means the Modelfile wins). Remove the drop-in and restart. |
| Truncation limit at `num_ctx` 16384 | Not verified; only `limit=2050` at 4096 was observed | Run a needle prompt of about 9,000 tokens (about 490 filler lines at 18.3 tokens/line) on the derived model. Compare `prompt_eval_count` with the server log line `truncating input prompt limit=…`. About 13 min or more at 11.8 tok/s [E]. |
| `/api/embed` above 2048 tokens | Not verified | Embed an input of about 10 KB; check whether the server errors or truncates (compare `prompt_eval_count` with the input length). |
| Ollama install date | TODO(user): on what date was Ollama installed? | — |
| Sign-in status | Closed: never signed in [U] | No read-only CLI check exists |
| Offline-test connectivity evidence | Closed [U] | Appendix A5 |
| Corpus limitations | Small (109 loadable files), monolingual, prose-heavy, 8 duplicated files, DVC data absent | Revisit in Phases 1 and 7 |
| Corpus after Phase 1 | 88 files / 491,997 B, deduplicated, without `deploy/model/` and `project_context/` (9.6); still small, and mostly prose and Python | Revisit in Phase 7 |

## 7. Phase 1 — Ingestion filtering (Steps 1–3)

Steps 1–3 decide what may enter the searchable universe: the rules as data, the ignore policy and the per-file classification [D: rag_phase1.md Steps 1–3]. Section 8 (Steps 4–5) adds the directory walker, reading and decoding, the size cap, empty files, deduplication and the `Document` objects. Code: `src/ingestion/{reasons,rules,ignore,classify,binary}.py`; corpus rules: `config/ingest.ignore`.

**7.1 Noise categories**

| Category | Mechanism | Reason code | Test | Status |
|---|---|---|---|---|
| Third-party dependencies | Built-in patterns `node_modules/`, `.venv/`, `venv/`, `site-packages/` (any depth), `/vendor/` (root only) | `builtin_pattern` | `tests/test_ingestion_ignore.py` | Done |
| Builds, caches, VCS metadata | Built-in `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.next/` (any depth), `/build/`, `/dist/`, `/target/` (root only), `.git` (always) | `builtin_pattern` | `tests/test_ingestion_ignore.py` | Done |
| What the project ignores | The corpus's `.gitignore` files, root and nested, with Git semantics | `gitignore` | `tests/test_ingestion_ignore.py`, including a comparison with `git check-ignore` | Done |
| Corpus-specific noise and prose | `config/ingest.ignore`: `/deploy/model/` (Q2) and `/project_context/` (Q1) | `corpus_rules` | `tests/test_ingestion_ignore.py` (the versioned file itself) | Done |
| Lockfiles | Nine exact file names (Q3) | `lockfile` | `tests/test_ingestion_classify.py` | Done |
| Secrets | File-name globs (D4); the file is never opened | `secret_name` | `tests/test_ingestion_classify.py` | Done |
| Data, notebooks, media, serialized models | Not in the language map (`.json`, `.csv`, `.ipynb`, `.pkl`, `.joblib`, `.png`, `.pdf`, `.mp4`, …) | `unmapped_type` | `tests/test_ingestion_classify.py` | Done |
| Disguised binaries | NUL byte in the first 1,024 bytes | `binary` | `tests/test_ingestion_binary.py` | Done |
| Unreadable and special files | Open without following symlinks and without blocking; read failures are a verdict, not a crash | `unreadable`, `not_regular_file` | `tests/test_ingestion_binary.py` | Done |
| Duplicates | sha256 of the content; the lexicographically smallest POSIX path is kept (Q2) | `duplicate` | `tests/test_ingestion_dedup.py` | Done (section 8) |

All of them are exercised together on a synthetic, corpus-shaped tree in `tests/test_ingestion_filters_integration.py`.

**7.2 Decisions**

| ID | Decision | Source |
|---|---|---|
| Q1 | Index `docs/` (44 ADRs and their index) and the root `README.md`; exclude the corpus's `project_context/` (13 `.md` files, 299,816 B) | [U] 2026-11-25; [M] |
| Q2 | Exclude `deploy/model/` (38 files, 1,477,368 B; its 18 `.py` are a stale copy of `src/`) and deduplicate by sha256 in Step 4 | [U] 2026-11-25; [M] |
| Q3 | Language map = the Step 3 map plus `.yaml`/`.yml` → `yaml` and `.toml` → `toml`; `Dockerfile` and `Makefile` by exact file name; nine lockfiles excluded by name; notebooks, JSON and TXT stay out | [U] 2026-11-25 |
| Q4 | Corpus rules live in a versioned gitignore-syntax file; its path is `INGEST_IGNORE_FILE` in `src/config.py` (environment override; a relative value is relative to this repository's root; no disk access at import) | [U] 2026-11-25 |
| Q8 | Root and nested `.gitignore` files with Git semantics: each applies relative to its own directory, the deepest one with a matching pattern wins, and nothing under an excluded directory can be re-included | [U] 2026-11-25 |
| D1 | Rules file `config/ingest.ignore` with exactly `/deploy/model/` and `/project_context/`. Defensive rules for `notebooks/`, `reports/`, `data/` and `.dvc/` were not added: they hold 0 files with a mapped extension | [U] 2026-11-25; [M] |
| D2 | Built-in patterns: unambiguous names at any depth; `build`, `dist`, `target`, `vendor` at the root only, so a legitimate nested `build/` survives. No such directory exists in the corpus | [U] 2026-11-25; [M] |
| D3 | A corpus `.gitignore` that is unreadable, not UTF-8, larger than 1 MiB, or a symlink (or under one) stops ingestion with `IgnoreRulesError` (fail-closed). A single invalid pattern line is skipped with a WARNING (file and line number, no content). The versioned rules file is strict: any invalid line is an error | [U] 2026-11-25 |
| D4 | Exclude and count file names matching `.env`, `.env.*` (except `*.example`, `*.sample`, `*.template`), `*.pem`, `*.key`, `id_rsa*`, `*credentials*`, `*secret*`, on the base name, case-insensitively, before the language check. 0 matches in the corpus | [U] 2026-11-25; [M] |
| Library | `pathspec` 1.1.1 `GitIgnoreSpec`, queried with `check_file()`, whose result tells "excluded", "re-included by `!`" and "no match" apart. Directories are queried with a trailing slash | [M] (installed source) |

Deviations from rag_phase1.md: see the single table in 9.9.

**7.3 Evaluation order.** Cheapest check first; the first filter that rejects a path gives its reason:

1. built-in patterns → `builtin_pattern`
2. corpus rules file → `corpus_rules`
3. the corpus's `.gitignore` chain → `gitignore`
4. lockfile name → `lockfile`
5. secret-like name → `secret_name`
6. language by file name, then by lowercased last extension → `unmapped_type`
7. first-KiB probe → `binary`, `unreadable` or `not_regular_file`

Filters 1–3 also apply to directories, so Step 4 can prune them before descending. They combine by union: a `!` line works inside the rules file and inside the `.gitignore` chain, never across sources. For the loader, 8.3 supersedes this order: symlinks and special files get their own codes from the directory entry, before name classification.

**7.4 Known limitations:** see the single table in 9.10.

**7.5 Result on the corpus (preview of Steps 1–3)** [M]. Run on 2026-11-25 at 15:41:49Z with the corpus at `5c288080…`. Before and after the run, HEAD was the same and `git status` was empty.

| Language | Files | Bytes |
|---|---|---|
| markdown | 46 | 263,570 |
| python | 32 | 192,670 |
| yaml | 7 | 24,349 |
| toml | 1 | 3,797 |
| dockerfile | 1 | 4,964 |
| makefile | 1 | 2,647 |
| **Total** | **88** | **491,997** |

| Finding | Value |
|---|---|
| By top-level directory | `docs/` 45 (238,798 B), `src/` 19 (121,650 B), root 8 (44,348 B), `tests/` 6 (47,488 B), `pipelines/` 4, `scripts/` 2, `.github/` 2, `docker/` 2 |
| Discards (161 tracked files) | `corpus_rules` 51 (38 in `deploy/model/`, 13 in `project_context/`), `unmapped_type` 20, `lockfile` 2 (`uv.lock`, `dvc.lock`). `builtin_pattern`, `gitignore`, `secret_name`, `binary`, `unreadable` and `not_regular_file`: 0 |
| Binaries | The six binaries are dropped earlier, by type or by the rules file, so the probe finds none |
| Against the expected Step 4 result | Equal to the expected 88 files / 491,997 B (decision log). Step 4's deduplication and empty-file rule change nothing here: the eight known duplicates are all inside the excluded `deploy/model/`, the 88 files contain 0 duplicate groups (sha256, diagnostic only), and none is empty |
| Corpus `.gitignore` files | All four were read (root, `.dvc/`, `data/raw/`, `data/monitoring/`) and they exclude none of the tracked files, which agrees with `git ls-files -c -i --exclude-per-directory=.gitignore` (0). This corpus therefore checks the rules and the maps; the `.gitignore` engine is checked by the tests (synthetic trees and the comparison with Git) |

How it was produced:
- **Input:** the universe was `git -C <corpus> ls-files -z` (161 paths).
- **Script:**
  - An ephemeral script, not versioned, kept in a new `mktemp -d` folder.
  - It was run from this repository's root with `uv run --locked --offline python -I`, and imported only `src.config` and `src.ingestion`.
  - It ran under `NetworkGuard`, with 0 violations. `unshare -rn` was blocked by the host's user-namespace policy.
- **Per path:** `IgnorePolicy` (built-ins, `config/ingest.ignore`, the corpus's `.gitignore` files) → `classify_name` → `probe_head` (first KiB only). The first reason was recorded. The included files were also hashed, for the duplicate check only.
- **Superseded by 8.7:** the loader and CLI give the same result. Here the 51 files under the two excluded directories are counted one by one; 8.7 counts the pruned directories instead. Both add up: 161 tracked files = 88 included + 22 discarded + 51 under `deploy/model/` and `project_context/`.

## 8. Phase 1 — Loading and metadata (Steps 4–5)

Steps 4–5 turn the filters of section 7 into the loader that later phases call: traversal, reading, deduplication, LlamaIndex `Document` objects with their metadata, a load report and a CLI [D: rag_phase1.md Steps 4–5]. Code: `src/ingestion/{walker,content,documents,dedup,report,loader}.py`; only `documents.py` and `loader.py` import LlamaIndex, and only `loader.py` imports `src.config`.

**8.1 Decisions**

| ID | Decision | Source |
|---|---|---|
| Q2 | Deduplicate by sha256 of the bytes on disk, last and only among files that passed every other filter; keep the lexicographically smallest POSIX path, discard the rest as `duplicate` with their canonical path | [U] 2026-11-25 |
| Q5 | `llama-index-core` 0.14.26 only (no metapackage, no integrations): 46 new packages, all wheels, 53,381,296 B; no version already locked changed | [U] 2026-11-25; [M] |
| Q6 | Six metadata keys; `doc_id` derived from `file_path`; only `file_path` and `language` visible to the embedding model and the LLM (table 8.5) | [U] 2026-11-25 |
| D5 | Strict UTF-8; one leading BOM stripped; `\r\n` and lone `\r` become `\n` in `Document.text`; `num_lines` = number of `\n` plus 1 if the text does not end with one; `size_bytes` and `content_hash` over the bytes on disk | [U] 2026-11-25; [M] the 88 corpus files have no CR, BOM or NUL |
| D6 | `doc_id` (`id_`) = `file_path`: deterministic, unique within a repository, independent of content and of the root's location | [U] 2026-11-25 |
| D7 | `max_file_bytes` = 1 MiB (1,048,576 B); larger files are `too_large`. The largest corpus file has 30,753 B, so the cap drops nothing here | [U] 2026-11-25; [M] |
| D8 | `load_repository` raises `NoDocumentsError` when nothing passes the filters; `load_repository_with_report` returns the empty result; the CLI exits 21 | [U] 2026-11-25 |
| `REPO_PATH` | New `Final[Path]` in `src/config.py`, default `~/repos/project-to-index`. It must be absolute or start with `~/` (relative values are rejected); `~` expands from `HOME`; no disk access at import; without `REPO_PATH` and without a usable `HOME`, import fails with `ConfigurationError` | [D: rag_phase1.md Step 4], prompt contract |
| Venv permissions | With the owner's `umask 0002` and uv's default hardlinks, nltk 3.10.4 refused the NLTK data bundled in `llama-index-core` (group-writable, then multiply-linked files), and `llama-index-core` tried to download it instead (blocked by the guard). Fixed with `chmod -R go-w .venv`, `[tool.uv] link-mode = "copy"`, `umask 022` in `make install`, and an offline reinstall. Never set `NLTK_DATA` or `TIKTOKEN_CACHE_DIR`: either one makes `llama-index-core` skip its bundled data | [U] 2026-11-25; [M] |
| Walker order | Symlinks and special files are rejected by the walker from the directory entry, right after the path filters and before name classification: every symlink is reported as `symlink`, whatever its name | Implementation decision; prompt test requirement |

Deviations from rag_phase1.md: see the single table in 9.9.

**8.2 Discard reasons.**

| Code | Mechanism | Test |
|---|---|---|
| `invalid_name` | Name not valid UTF-8, or containing a backslash or a control character | `tests/test_ingestion_walker.py` |
| `builtin_pattern`, `corpus_rules`, `gitignore` | Ignore policy (section 7); directories pruned | `tests/test_ingestion_ignore.py`, `tests/test_ingestion_walker.py` |
| `symlink` | Directory entry, or `lstat` of a file swapped after the walk | `tests/test_ingestion_walker.py`, `tests/test_ingestion_content.py` |
| `not_regular_file` | FIFO, socket or device; never opened | `tests/test_ingestion_walker.py`, `tests/test_ingestion_binary.py` |
| `lockfile`, `secret_name`, `unmapped_type` | File name (section 7) | `tests/test_ingestion_classify.py` |
| `too_large` | `lstat` size above the cap, or more than the cap read | `tests/test_ingestion_content.py` |
| `binary` | NUL in the first KiB (probe) or anywhere in the decoded text | `tests/test_ingestion_binary.py`, `tests/test_ingestion_content.py` |
| `unreadable` | Open, read or directory listing failed | `tests/test_ingestion_content.py`, `tests/test_ingestion_walker.py` |
| `undecodable` | Not valid UTF-8 | `tests/test_ingestion_content.py` |
| `empty` | No bytes, or only whitespace after decoding | `tests/test_ingestion_content.py` |
| `duplicate` | Same sha256 as a kept file with a smaller path | `tests/test_ingestion_dedup.py` |

All of them are exercised together in `tests/test_ingestion_loader_integration.py`, and `tests/test_ingestion_discard_coverage.py` produces each one through the loader (9.5).

**8.3 Evaluation order** (the first rejection gives the reason):
1. Directory entry: `invalid_name` → the three ignore sources (directories are pruned) → `symlink` → `not_regular_file`.
2. File name: `lockfile` → `secret_name` → `unmapped_type`.
3. `lstat`: `symlink` / `not_regular_file` / `too_large`.
4. First-KiB probe: `binary` / `unreadable` / `not_regular_file`.
5. Bounded read: `too_large` / `unreadable` / `undecodable` / `binary` (NUL) / `empty`.
6. `duplicate`, among the survivors only.

**8.4 API and CLI.**
- **Functions:** `load_repository(repo_path, *, rules=None, rules_file=None) -> list[Document]` and `load_repository_with_report(...) -> LoadResult(documents, report, discarded, pruned)`. Both are deterministic and write nothing.
- **Report:** included files and bytes, by language and by top-level directory; discarded files and pruned directories by reason; duplicate pairs; `.gitignore` files read. No timings and no absolute paths.
- **CLI:** `python -m src.ingestion.loader [--repo P] [--rules-file F] [--list | --json]`, or `make ingest INGEST_FLAGS=...`.
- **Exit codes:** 0 OK, 1 unexpected, 2 arguments, 10 configuration, 14 network, 20 ingestion error, 21 no documents.

**8.5 Metadata contract**

| Key | Type | Definition | Seen by the models |
|---|---|---|---|
| `file_path` | str | POSIX path relative to the root (also the `doc_id`) | Yes |
| `language` | str | One of the 16 labels of section 7 | Yes |
| `size_bytes` | int | Bytes on disk | No |
| `num_lines` | int | Editor-style line count of `text` (D5) | No |
| `content_hash` | str | sha256 hex of the bytes on disk | No |
| `directory` | str | POSIX parent of `file_path`, `""` at the root | No |

What the embedding model and the LLM see is `file_path: <path>\nlanguage: <label>\n\n<text>`. For the longest corpus path (65 characters) the visible metadata takes 95 characters, which Phase 2's splitter subtracts, in tokens, from every chunk. Each document has its own excluded-key lists. Vector stores add `doc_id`, `document_id`, `ref_doc_id`, `_node_content` and `_node_type`, none of which collides with these keys. LlamaIndex's `Document.hash` is a different value from `content_hash`.

**8.6 Known limitations:** see the single table in 9.10.

**8.7 Result on the corpus** [M], 2026-11-25 at 17:04–17:05Z. The production CLI ran under `NetworkGuard` (0 violations). The corpus was at `5c288080…`, with the same HEAD, an empty `git status` and 0 untracked files before and after the run.

| Language | Files | Bytes |
|---|---|---|
| markdown | 46 | 263,570 |
| python | 32 | 192,670 |
| yaml | 7 | 24,349 |
| toml | 1 | 3,797 |
| dockerfile | 1 | 4,964 |
| makefile | 1 | 2,647 |
| **Total** | **88** | **491,997** |

| Finding | Value |
|---|---|
| By top-level directory | `docs` 45, `src` 19, root 8, `tests` 6, `pipelines` 4, `scripts` 2, `.github` 2, `docker` 2 |
| Discarded files | 22: `unmapped_type` 20, `lockfile` 2 |
| Pruned directories | 3: `.git` (`builtin_pattern`), `deploy/model` and `project_context` (`corpus_rules`, which held the 51 files of 7.5) |
| Duplicates, `.gitignore` files read | 0, 4 |
| Against the expected result | Equal to 88 files / 491,997 B and to every per-language figure. The `--list` output is the same set of 88 paths as the Steps 1–3 preview (`git ls-files` universe), sorted. `size_bytes` sums to 491,997; `num_lines` sums to 10,194 |
| Spot checks | `content_hash` and `size_bytes` match `sha256sum` and `stat` for `src/models/register.py` and `README.md` |
| Time | 9.3 ms for the load (from the log); 0.86 s for the whole process, including `uv run` and importing LlamaIndex |

**8.8 Notes for Phase 2:** moved to the risk register in 9.11.

## 9. Phase 1 — Languages, tests and closure (Steps 6–7)

Step 6 asks ingestion to handle several languages and leaves parsers to Phase 2, starting with the repository's primary language [D: rag_phase1.md Step 6]. The map already holds 18 extensions, 2 file names and 16 labels (Q3), so this step is a review, contract tests and documentation; nothing was added (D9).

**9.1 Language labels** [M] corpus figures from `make ingest INGEST_FLAGS=--json`, 2026-11-25. Phase 2 status per [D: rag_phase2.md Steps 3–4]: "syntax-aware" = in `SYNTAX_AWARE_LANGUAGES`; "+ symbol" = also in `DEF_NODE_TYPES`.

| Label | Extensions / file names | Corpus files | Corpus bytes | Phase 2 status |
|---|---|---|---|---|
| `python` | `.py` | 32 | 192,670 | syntax-aware + symbol |
| `javascript` | `.js`, `.jsx` | 0 | 0 | syntax-aware + symbol |
| `typescript` | `.ts`, `.tsx` | 0 | 0 | syntax-aware + symbol |
| `java` | `.java` | 0 | 0 | syntax-aware |
| `go` | `.go` | 0 | 0 | syntax-aware |
| `rust` | `.rs` | 0 | 0 | syntax-aware |
| `c` | `.c`, `.h` | 0 | 0 | syntax-aware |
| `cpp` | `.cpp` | 0 | 0 | syntax-aware |
| `ruby` | `.rb` | 0 | 0 | fallback (labelled, not in the set) |
| `csharp` | `.cs` | 0 | 0 | fallback (labelled, not in the set) |
| `php` | `.php` | 0 | 0 | fallback (labelled, not in the set) |
| `markdown` | `.md` | 46 | 263,570 | fallback (prose) |
| `yaml` | `.yaml`, `.yml` | 7 | 24,349 | fallback |
| `toml` | `.toml` | 1 | 3,797 | fallback |
| `dockerfile` | `Dockerfile` (exact name) | 1 | 4,964 | fallback |
| `makefile` | `Makefile` (exact name) | 1 | 2,647 | fallback |

Every row is pinned by `tests/test_ingestion_rules.py` (exact map and exact label set) and reaches a `Document` in `tests/test_ingestion_languages.py` (synthetic polyglot repository, with each ecosystem's noise discarded by reason). Hypotheses for Phase 2, not verified here [H]: a grammar name may differ from the label (`makefile`, `csharp`); `.tsx` may need its own grammar rather than `typescript`'s; a `.h` file may be C++.

**9.2 Primary language** [M]

| Ranking | By files | By bytes |
|---|---|---|
| 1 | `markdown` 46 (52.3%) | `markdown` 263,570 (53.6%) |
| 2 | `python` 32 (36.4%) | `python` 192,670 (39.2%) |
| 3 | `yaml` 7 | `yaml` 24,349 |
| 4–6 | `dockerfile`, `toml`, `makefile`: 1 each | `dockerfile` 4,964, `toml` 3,797, `makefile` 2,647 |

Definition used here: the primary language is the code language with the most files and bytes among the labels Phase 2 can parse. That is `python`, the only code language in the corpus. Markdown leads both rankings, but it is prose (the ADRs and the README, Q1).

Share that would take Phase 2's fallback splitter on this corpus: every label except `python`, that is 56 of 88 files (63.6%) and 299,327 of 491,997 B (60.8%) [M] (label counts). Syntax-aware: 32 files (36.4%), 192,670 B (39.2%). How many chunks that means depends on Phase 2's splitters [H].

**9.3 How to add a language**
1. Get the owner's decision: labels are a contract with Phases 2 and 4.
2. Data only, in `src/ingestion/rules.py`: add a lowercase single suffix to `DEFAULT_LANGUAGE_BY_EXTENSION`, or an exact file name to `DEFAULT_LANGUAGE_BY_FILENAME`. The label must be lowercase (`[a-z][a-z0-9_+-]*`).
3. Tests: update `EXPECTED_EXTENSIONS` (or the file-name map) and `EXPECTED_LANGUAGES` in `tests/test_ingestion_rules.py`, and add a sample file to the polyglot repository in `tests/test_ingestion_languages.py`.
4. Add a row to table 9.1.
5. Phase 2: add the label to its syntax-aware set (and to its symbol node types) only if a parser exists for it; otherwise it falls back.
6. If the corpus has files with that extension: the expected totals (88 files / 491,997 B) and the manifest change. Regenerate the manifest (Appendix P3), validate it independently, and update the key test's totals.

**9.4 Files the map leaves out (D9)** [M] All 22 discarded corpus files are tracked.

| Reason | Files | Paths (bytes) |
|---|---|---|
| `lockfile` | 2 | `uv.lock` (1,012,957), `dvc.lock` (3,846) |
| `unmapped_type`, media and data | 7 | `docs/images/architecture.png`, `docs/videos/intro.mp4`, `docs/whitepaper.pdf`, `notebooks/01_exploration.ipynb`, `reports/metrics.json`, `reports/validation.json`, `data/raw/creditcard.csv.dvc` |
| `unmapped_type`, text with no label | 5 | `LICENSE` (1,077), `docker/.env.example` (608), `.dvc/config` (111), `.dockerignore` (2,001), `.python-version` (5) |
| `unmapped_type`, VCS plumbing | 8 | four `.gitignore`, `.dvcignore`, three `.gitkeep` |

D9 [U] 2026-11-25: the map stays as it is. The five text files in the third row cannot answer questions (for example "which license?" or "which variables does the Docker setup need?"); mapping them would need new labels.

**9.5 Definition of Done → tests** [M] `pytest --collect-only -q`; test counts per file in brackets. The checkboxes are those of [D: rag_phase1.md "Verification: The Definition of Done"].

| # | Checkbox | Tests |
|---|---|---|
| 1 | `loader.py` traverses a repository and loads its code files | `test_ingestion_walker.py` [23], `test_ingestion_loader.py` [20] (API, CLI, determinism, no writes), `test_ingestion.py` [2] |
| 2 | Dependencies, builds, binaries and `.gitignore` are filtered out | `test_ingestion_ignore.py` [102] (incl. a comparison with `git check-ignore`), `test_ingestion_binary.py` [22], `test_ingestion_content.py` [32], `test_ingestion_loader_integration.py` [2], `test_ingestion_filters_integration.py` [4], `test_ingestion_dedup.py` [7] |
| 3 | Each document carries path, language, size and lines | `test_ingestion_documents.py` [79] (the six keys, visibility, `num_lines`), `test_ingestion_loader_integration.py`, `test_ingestion.py` |
| 4 | Multiple languages through the extension map | `test_ingestion_rules.py` [48] (exact map and labels), `test_ingestion_classify.py` [96], `test_ingestion_languages.py` [5] (all 16 labels end to end) |
| 5 | Tests verify the discards, not only the loads | Every discard is asserted with its reason; `test_ingestion_discard_coverage.py` [16] produces each of the 15 reasons through the loader and fails if a reason has no scenario |
| 6 | Key test: the real repository gives only the relevant files | See 9.6 |

Gaps closed in this step: no test loaded all 16 labels through the loader (6 did, all from the corpus shape); `unreadable` and `invalid_name` were asserted only at module level; the doc's two tests did not exist in an adapted form; there was no key test. Privacy is covered across all checkboxes by the autouse network guard and by `test_ingestion_privacy.py` [2].

**9.6 Key test and manifest (Q7)**
- **Manifest:** `tests/fixtures/corpus_manifest.tsv`, UTF-8, LF line endings, no BOM. A `#` header gives the public corpus URL, the frozen commit, the generation time and command, the columns and the totals. Then there is one line per included file, `path<TAB>language<TAB>size_bytes<TAB>sha256`, sorted by the path string (code points, like the loader). It holds no file contents. The sha256 column detects a changed tree even when HEAD matches, and it gives Phase 7 a fixed anchor.
- **Generation** [M], 2026-11-25 at 19:42:37Z: `python -m tests.corpus_manifest` (Appendix P3) used the production loader under `NetworkGuard` (0 violations). It refuses to write unless the corpus HEAD is the frozen commit.
- **Independent validation** [M], so that the manifest is not a copy of the code under test:
  - All 88 paths are in `git ls-files`, sorted and unique.
  - `sha256sum` and `stat` agree for 88 of 88 files.
  - A literal extension map in `awk` agrees on 88 of 88 labels.
  - The Steps 1–3 test helper (`os.walk` → `IgnorePolicy` → `classify_name` → `probe_head`) keeps the same 88 paths with the same labels.
  - There are 0 duplicate hashes.
  - 161 tracked files = 88 included + 22 discarded + 51 under the two pruned rule directories.
- **Key test:** `tests/test_ingestion_corpus_integration.py`, marker `integration`, 9 tests, run with `make test-corpus`.
  - **Skips** only when `REPO_PATH` is missing or its HEAD (read from `.git` in pure Python, without running `git`) is not `5c288080c88f0bfec435af0093a5155baaf7f343`. The skip states the path and both SHAs.
  - **Asserts:**
    - the exact included set and order against the manifest;
    - `language`, `size_bytes` and `content_hash` for every file;
    - the totals of 88 files / 491,997 B, by language and by top-level directory;
    - that nothing comes from noise directories, lockfiles, secret-like names or unmapped extensions;
    - the 22 tracked discards (`unmapped_type` 20, `lockfile` 2), and that the pruned directories include `.git`, `deploy/model` and `project_context`;
    - 0 duplicates and 4 `.gitignore` files read;
    - the six-key contract, with two visible keys, and no NUL, BOM or CR in the text;
    - that a second load through `load_repository` is identical;
    - that a `(mode, size, mtime_ns)` snapshot of the whole tree is unchanged, with 0 network violations.
- **Evidence** [M]:
  - 9 passed on this host in 0.81 s.
  - With `REPO_PATH=/nonexistent/corpus`: 9 skipped, each with the reason.
  - On a modified scratch copy of the corpus (one edited file and one new file, same HEAD): 4 failed, with a diff naming the path, the field and both values. Not skipped.
  - `make test` deselects it.
  - The test's logic is covered without the corpus by `tests/test_corpus_manifest.py` [60]: HEAD resolution (loose, packed and detached refs, worktree `.git` file, symlinks, unsafe ref names, corrupt data), the strict parser and the comparator.
- **Not automatable:** whether the 88 files are "only the relevant code, with no junk" is the owner's judgment (section 9.8).

**9.7 Contract with later phases** (read-only audit of [D: rag_phase2.md]–[D: rag_phase8.md] and [D: rag_generalRoadmap.md]; nothing was changed in them)

| Phase | What it reads from Phase 1 | What Phase 1 delivers | Status |
|---|---|---|---|
| 2 | `{doc.doc_id: doc.text}`, `node.ref_doc_id`, `metadata["language"]` against `SYNTAX_AWARE_LANGUAGES`, `count("\n")` for line numbers, `REPO_PATH` in `__main__` (Steps 3, 4, 6) | `doc_id` = `file_path` (so `ref_doc_id` = path); text with `\n` only; 16 labels; `REPO_PATH: Final[Path]` | ✔, with risks R1–R8 |
| 3 | `build_index(repo_path)` calls `load_repository(repo_path)`; persistent Chroma (Steps 4–5) | Unchanged signature; `NoDocumentsError` on an empty load | ✔; R6 |
| 4 | `MetadataFilter(key="language", operator=EQ)`; the prose also mentions a directory (Step 4) | Exact labels; `directory` and `file_path` stored | ✔; R10 |
| 5 | Citations from `metadata["file_path"]`; `__main__` with `REPO_PATH` | Relative POSIX path, visible to the LLM | ✔ |
| 6 | `lifespan` reloads `chunk_documents(load_repository(REPO_PATH))` at every start-up; `/index` takes `repo_path` from the request (Steps 2–4) | Deterministic, writes nothing, 6.0–9.6 ms per load [M] | ✔; R12 |
| 7 | An evaluation set of questions and references (Step 3); no file anchors | Manifest with the commit and a sha256 per file (9.6) | ⚠ R13 (informational) |
| 8 | The README's design-decisions section | This document | ✔ |

**9.8 Closure** (2026-11-25). Phase 1 Definition of Done [D: rag_phase1.md "Verification: The Definition of Done"]:

| # | Checkbox | Status | Evidence |
|---|---|---|---|
| 1 | `loader.py` traverses a repository and loads its code files | ✔ | API and CLI tests (9.5); `make ingest` on the corpus: 88 files |
| 2 | Dependencies, builds, binaries and `.gitignore` are filtered | ✔ | Reasons asserted (9.5), 15/15 coverage, the key test's negative checks |
| 3 | Rich metadata per document | ✔ | The six-key contract (8.5), checked on all 88 corpus documents |
| 4 | Multiple languages through the map | ✔ | Exact map, 16 labels end to end, inventory (9.1–9.4) |
| 5 | Tests verify the filtering | ✔ | 9.5; `make check` green |
| 6 | Key test: only the relevant files of a real repository | ✔ | 9.6 passes (9 of 9). The judgment was delegated by the owner ([U] 2026-11-25: "check it yourself") and made by the agent's review below |

Key-test review [M], 2026-11-25, paths and structural signals only (no content read into the record):
- **Junk among the 88:** none in the Definition of Done's sense (third-party dependencies, builds, binaries). 0 private-key headers, 0 AWS-shaped keys, 0 base64 runs of 100+ characters, 0 quoted secret assignments; no minified file (longest line 490 characters, `README.md`).
- **Low value, but the project's own code:** `main.py` (325 B, one function, no imports; its first docstring line says it was generated, by the corpus's `uv init` commit), and six `__init__.py` files under 100 B. Kept: excluding them would need a new rule and an owner decision.
- **Omissions:** none beyond the owner's decisions. All 18 `.py` files under `deploy/model/` have a counterpart in `src/` (8 byte-identical, 10 stale versions), so no unique code is lost (Q2). `project_context/` holds 13 `.md` files (Q1). The 22 discards are covered by Q3 and D9, and no code-like extension is left unmapped.

Phase 1 is closed. Final results on the corpus are those of 8.7 and 9.6: 88 files / 491,997 B, unchanged since the Steps 1–3 preview. Nothing in `src/`, `pyproject.toml` or `uv.lock` changed in Steps 6–7.

**9.9 Deviations from rag_phase1.md** (single table, Steps 1–7)

| Step | Doc | This project | Why |
|---|---|---|---|
| 1 | `IGNORED_DIRS` matched by name at any depth | D2: unambiguous names at any depth; `build`, `dist`, `target`, `vendor` at the root only | Avoid dropping a legitimate nested `build/` or `vendor/` |
| 1 | Every `.md` file is indexed | `project_context/` excluded | Q1 |
| 2 | Only the root `.gitignore` | Root and nested files, plus built-in patterns and the corpus rules file | Q8, Q4; the corpus has three nested `.gitignore` files [M] |
| 2 | `PathSpec.from_lines("gitwildmatch", …)` | `GitIgnoreSpec`, evaluated directory level by directory level | `gitwildmatch` is deprecated since `pathspec` 1.0.0; `GitIgnoreSpec` alone re-includes `build/keep.py` under `build/` + `!build/keep.py`, which Git does not [M] |
| 3 | Extension-only map, 15 entries | 18 extensions, 2 file names, lockfiles | Q3 |
| 3 | `is_probably_binary` reads the whole file and returns True on `OSError` | At most 1,024 bytes, never follows symlinks, does not block on FIFOs; `unreadable` is its own verdict | Bounded I/O; an I/O error says nothing about content |
| 4 | `rglob` and `path.is_file()` (follows symlinks) | Iterative `os.scandir`, ignored directories pruned, symlinks never followed (`symlink`) | Speed, Q8, D2; nothing outside the root can be read |
| 4 | `read_text` (universal newlines; errors skipped without a count) | Bounded read, strict UTF-8, normalization (D5), every failure counted with a reason | Determinism and traceability |
| 4 | No size cap, no empty-file rule, no deduplication, filesystem order | `too_large`, `empty`, `duplicate`; output sorted by POSIX path | D7, Q2; the same tree gives the same output |
| 4 | `num_lines = count("\n") + 1` | Editor-style count (D5) | The doc's formula adds a line to every file that ends with a newline (88 of 88 here) |
| 4 | Random `Document` ID | `id_` = `file_path` | D6; Phase 2 indexes text by `doc_id` |
| 4 | `__main__` with `print`; `uv add llama-index pathspec` | CLI with exit codes and JSON logs; `llama-index-core` and `pathspec` only | Ruff `T20`/`BLE`; Q5 |
| 5 | All metadata visible to the models | Four keys hidden | Q6 |
| 6 | "Expanding it is as simple as adding more extensions" | Review only: contract tests and documentation; the map is unchanged | Q3 already expanded it; labels are a contract (D9) |
| 7 | Two tests that assert absence (`logo.png` and `secret.py` each have two possible causes) | Single-cause noise with the reason asserted, a 15-reason meta-test, the polyglot repository | A passing absence test cannot tell which filter worked |
| DoD | Key test by inspection | Versioned manifest, `integration` test and the owner's confirmation | Q7 |

**9.10 Known limitations** (single table; 7.4's "NUL after the first KiB" is resolved by Step 4's decoded-text check)

| Limitation | Effect |
|---|---|
| UTF-16 and UTF-32 | Rejected as `binary` or `undecodable` |
| Case-sensitive matching | Like Git with `core.ignorecase=false`; may differ on case-insensitive filesystems |
| Repository-external ignore rules | `.git/info/exclude`, the global excludes file and `.gitattributes` are not read, so results do not depend on the machine |
| Exact build-file names | `Dockerfile.dev`, `dockerfile`, `makefile` get no label (tested in `test_ingestion_languages.py`) |
| Broad secret globs | `*secret*` and `*credentials*` also drop files such as `secrets_manager.py`; visible in the reason counts |
| Invalid ignore patterns | `pathspec` 1.1.1 raises on `[z-a]`, a lone `!` or `\`; Git ignores such lines; counted as invalid lines (D3) |
| Counts by directory | A pruned directory counts once; the report is not file-for-file comparable with `git ls-files` |
| Line-ending normalization | `len(text)` differs from `size_bytes`; copies differing only in CRLF/LF are not deduplicated |
| No line-length heuristic | Minified or generated files under 1 MiB load whole (a 32 KiB one-line file is a single line); the longest corpus line has 490 characters |
| Files swapped during a load | A file replaced by a symlink between `lstat` and opening is reported as `not_regular_file` |
| Walking cost | The ignore policy's cost per path grows with depth²: negligible at the corpus's depth of 6; a synthetic 1,100-level tree took about 11 s |
| `doc_id` scope | Unique within one repository only |
| Unlabelled text (D9) | `LICENSE`, `docker/.env.example`, `.dvc/config`, `.dockerignore`, `.python-version` cannot answer questions |
| One real codebase (D11) | Only this monolingual corpus was loaded for real; multi-language support rests on the synthetic polyglot repository |
| Key test portability | Needs the corpus; skips elsewhere (its logic is tested without it), and skips for a worktree or submodule `.git` file, which is not followed |
| `integration` marker | `make test-integration` also runs the two Phase 0 tests that need Ollama; the marker's description in `pyproject.toml` mentions Ollama only (owner's item) |

**9.11 Risk register for Phase 2 and later** (not fixed in Phase 1; [H] = hypothesis, not verified)

| # | Risk | Evidence | Recommendation |
|---|---|---|---|
| R1 | Most of the corpus takes the fallback splitter: 56/88 files, 299,327 B; Markdown alone is 53.6% of the bytes | 9.2 | Decide in Phase 2 whether prose deserves its own splitter |
| R2 | `ruby`, `csharp`, `php` are labelled but not in `SYNTAX_AWARE_LANGUAGES`; `DEF_NODE_TYPES` covers only Python, JavaScript and TypeScript | [D: rag_phase2.md Steps 3–4] | Extend per label only when a corpus needs it |
| R3 | Grammar names may differ from labels (`makefile`, `csharp`); `.tsx` and `.h` may need other grammars [H] | 9.1 | Map label → grammar explicitly in Phase 2 |
| R4 | `end_line = count("\n", 0, end_char_idx) + 1` points one line too far when a chunk ends with its newline | [D: rag_phase2.md Step 4] | Test start and end lines against the source |
| R5 | `nomic-embed-text` accepts 2048 tokens, visible metadata included (95 characters for the longest path) | 2.3, 8.5 | Keep chunk plus metadata under the cap |
| R6 | Default node IDs are `uuid4`: re-indexing into a persistent store would duplicate chunks; Qdrant needs UUID or integer point IDs [H] | [D: rag_phase3.md Step 5], [D: rag_generalRoadmap.md §5] | A deterministic `id_func` (for example a UUID5 of path and position) |
| R7 | The doc's chunker uses `except Exception` and `print` | [D: rag_phase2.md Step 6] | Ruff `BLE`/`T20`: narrow the exception; reuse the loader's CLI pattern |
| R8 | No line-length heuristic: a one-line file reaches the splitter whole | 9.10 | Check how `CodeSplitter` handles a line longer than `max_chars` [H] |
| R9 | The bundled NLTK and tiktoken data work offline only in a `.venv` without hardlinks or group write | 8.1 | Add dependencies with `umask 022`, or run `make install` afterwards; never set `NLTK_DATA` or `TIKTOKEN_CACHE_DIR` |
| R10 | An `EQ` filter on `directory` does not include subdirectories [H] | [D: rag_phase4.md Step 4] | Decide prefix semantics in Phase 4 |
| R11 | `doc_id` is unique per repository only | 9.10 | Prefix IDs if several repositories share a collection (Phase 3) |
| R12 | `/index` loads a path chosen by the API client; empty or missing roots raise typed errors; the endpoint loads twice | [D: rag_phase6.md Steps 3–4] | Allow-list paths; map `NoDocumentsError` and `RepositoryRootError` to 4xx |
| R13 | Evaluation answers are not tied to a corpus version | [D: rag_phase7.md Step 3] | Record the corpus commit and the manifest with each RAGAS run |

## Appendix: Evidence and reproduction

| # | Command | What it proves |
|---|---|---|
| A1 | `ss -ltnH 'sport = :11434'` | Ollama listens on 127.0.0.1 only |
| A2 | `systemctl cat ollama`; `journalctl -u ollama -o cat \| grep 'cloud disabled'` | The drop-in exists and the server logs `cloud disabled: true` |
| A3 | `make test` | The whole suite runs under the network guard with 0 violations |
| A4 | `make verify-local` (exit 0) | Local generation and embedding; `network_guard: blocked_attempts=0` |
| A5 | Offline: `nmcli networking connectivity` → `none`; `ip route show default` → empty; then `make verify-local` | Generation and embedding without internet (key DoD test) |
| A6 | `sudo ss -Htunap` once per second for 300 s, keeping only Ollama sockets with a non-loopback peer | Ollama opens no outbound sockets (sampled) |
| A7 | `ollama list`; `ollama show <model>`; `curl -s localhost:11434/api/ps` | Digests, sizes, dimension, context lengths |
| A8 | `make models`, twice; `ollama show qwen2.5-coder-7b-rag --modelfile` | Same digest both times; `num_ctx 16384` persisted |
| C1 | `git -C ~/repos/project-to-index remote -v`, `rev-parse HEAD`, `show -s --format=%cd`, `describe --tags --always`, `rev-parse --is-shallow-repository`, `rev-list --count HEAD`, `status --porcelain`, `log -g -1` | Corpus identity and clone time |
| C2 | `git -C <corpus> ls-files -z` and `git -C <corpus> ls-files -z -c -i --exclude-from=.gitignore` | The file universe, and the tracked files the root `.gitignore` matches |
| C3 | `python3 -I characterize.py <corpus> tracked.z root_gitignored.z`, from a new `mktemp -d` folder | All Step 6 aggregates |
| C4 | `git -C <corpus> ls-files -s -z`, grouped by blob ID | Byte-identical duplicates (8 `deploy/model` ↔ `src`) |
| P1 | `git -C <corpus> ls-files -z`, then the Steps 1–3 preview script via `uv run --locked --offline python -I`, from a new `mktemp -d` folder | Section 7.5: 88 files / 491,997 B and the discard reasons |
| P2 | `make ingest INGEST_FLAGS="--repo ~/repos/project-to-index"` (also with `--list` and `--json`) | Section 8.7: 88 files / 491,997 B, discard reasons and pruned directories |
| P3 | Only after an owner decision that changes the expected files. First check that `git -C <corpus> rev-parse HEAD` is the frozen SHA and `GIT_OPTIONAL_LOCKS=0 git -C <corpus> status --porcelain` is empty. Then run `uv run --locked --offline python -m tests.corpus_manifest --out tests/fixtures/corpus_manifest.tsv` from this repository's root. Validate independently: the paths against `git -C <corpus> ls-files -z`, every row against `sha256sum` and `stat`, and the set against a second traversal (the Steps 1–3 helper `_filter_tree`) | The manifest of section 9.6 |
| P4 | `make test-corpus` (`pytest -m integration -rs tests/test_ingestion_corpus_integration.py`) | The key test (9.6). `make test-integration` also runs it, together with the two Phase 0 tests that need Ollama |

C3 is a stdlib-only script, not versioned in this repository, and imports no networking modules. It does not follow symlinks and handles paths as bytes. Per file it records:
- a binary flag (NUL byte in the first KiB), lines as `count("\n") + 1`, and UTF-8 validity;
- Phase 1's map and `IGNORED_DIRS`, applied in Phase 1's filter order.

It prints counts and relative paths only. `unshare -rn` was attempted, and blocked by the host's user-namespace policy.
