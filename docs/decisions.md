# Design decisions: local-code-rag

| Date (UTC) | Phase | Scope |
|---|---|---|
| 2026-11-24 | 0: Setup and fundamentals (Steps 1–7) | Use case, model, privacy argument, corpus, stack. Phases 1+ are planned, not built. |

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

## 7. Phase 1 — Ingestion filtering (Steps 1–3)

Steps 1–3 decide what may enter the searchable universe: the rules as data, the ignore policy and the per-file classification [D: rag_phase1.md Steps 1–3]. Step 4 adds the directory walker, reading and decoding, the size cap, empty files, deduplication and the `Document` objects. Code: `src/ingestion/{reasons,rules,ignore,classify,binary}.py`; corpus rules: `config/ingest.ignore`.

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
| Duplicates | sha256 of the content; the lexicographically smallest POSIX path is kept (Q2) | `duplicate` | — | Planned, Step 4 |

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

**Deviations from rag_phase1.md**

| Doc | This project | Why |
|---|---|---|
| Step 2 reads the root `.gitignore` only | Root and nested files, plus built-in patterns and the corpus rules file | Q8, Q4; the corpus has three nested `.gitignore` files [M] |
| `PathSpec.from_lines("gitwildmatch", …)` | `GitIgnoreSpec`, evaluated directory level by directory level | `gitwildmatch` is deprecated since `pathspec` 1.0.0 (it emits `DeprecationWarning`); `GitIgnoreSpec` alone re-includes `build/keep.py` under `build/` + `!build/keep.py`, which Git 2.53 does not [M] |
| `IGNORED_DIRS` matched by name at any depth | D2 | Avoid dropping a legitimate nested `build/` or `vendor/` |
| Extension-only map, 15 entries | 18 extensions, 2 file names, lockfiles | Q3 |
| `is_probably_binary` reads the whole file and returns True on `OSError` | Reads at most 1,024 bytes, never follows symlinks, does not block on FIFOs; "unreadable" is its own verdict | Bounded I/O; an I/O error says nothing about binary content |
| Every `.md` file is indexed | `project_context/` excluded | Q1 |

**7.3 Evaluation order.** Cheapest check first; the first filter that rejects a path gives its reason:

1. built-in patterns → `builtin_pattern`
2. corpus rules file → `corpus_rules`
3. the corpus's `.gitignore` chain → `gitignore`
4. lockfile name → `lockfile`
5. secret-like name → `secret_name`
6. language by file name, then by lowercased last extension → `unmapped_type`
7. first-KiB probe → `binary`, `unreadable` or `not_regular_file`

Filters 1–3 also apply to directories, so Step 4 can prune them before descending. They combine by union: a `!` line works inside the rules file and inside the `.gitignore` chain, never across sources.

**7.4 Known limitations**
- UTF-16 and UTF-32 text contains NUL bytes and is classified as binary.
- A NUL byte after the first KiB is not detected. Step 4's strict UTF-8 decoding rejects most other binary content, but NUL is valid UTF-8.
- Matching is case-sensitive, like Git with `core.ignorecase=false`. On case-insensitive filesystems the results may differ from Git's.
- `.git/info/exclude`, the global `core.excludesFile` and `.gitattributes` are not read: they are not part of the repository, so reading them would make results depend on the machine.
- `Dockerfile.dev` or a lowercase `makefile` get no language (exact names only).
- `*secret*` and `*credentials*` also exclude legitimate files such as `secrets_manager.py`. The skip is visible in the reason counts.
- `pathspec` 1.1.1 raises `re.error` on an invalid range such as `[z-a]` and `GitIgnorePatternError` on a lone `!` or `\`; Git ignores such lines. Both count as invalid lines (D3).

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
- **Next:** Step 4's loader and CLI will replace the script.

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

C3 is a stdlib-only script, not versioned in this repository, and imports no networking modules. It does not follow symlinks and handles paths as bytes. Per file it records:
- a binary flag (NUL byte in the first KiB), lines as `count("\n") + 1`, and UTF-8 validity;
- Phase 1's map and `IGNORED_DIRS`, applied in Phase 1's filter order.

It prints counts and relative paths only. `unshare -rn` was attempted, and blocked by the host's user-namespace policy.
