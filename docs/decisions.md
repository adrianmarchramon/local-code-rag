# Design decisions: local-code-rag

| Date (UTC) | Phase | Scope |
|---|---|---|
| 2026-11-24 | 0: Setup and fundamentals (Steps 1–7) | Use case, model, privacy argument, corpus, stack. Phases 1+ are planned, not built. |
| 2026-11-25 | 1: Ingestion (Steps 1–7) | Filters, loader, metadata, languages, tests and the key test (sections 7–9). Phases 2+ are planned, not built. |
| 2026-11-25 | 2: Chunking (Steps 1–2) | Coherence invariants, parser registry and segmenter (section 10). Steps 3–7 and Phases 3+ are planned, not built. |
| 2026-11-26 | 2: Chunking (Steps 3–6) | Dispatch, nodes, fallbacks, chunker, report and CLI (section 11). Step 7 and Phases 3+ are planned, not built. |
| 2026-11-26 | 2: Chunking (Step 7 and closure) | Tests, key test, review and closure (section 12). Phase 2 is closed; Phases 3+ are planned, not built. |
| 2026-11-27 | 3: Embeddings and indexing (Steps 1–4) | Recipe, adapter, hardened Chroma store, manifest and pipeline (section 13). Steps 5–7 and Phases 4+ are planned, not built. |
| 2026-11-27 | 3: Embeddings and indexing (Steps 5–7) | Verified loading, search, evaluation and tests (section 14); a Step 2 defect fixed (exhaustive HNSW search, 14.11). Phase 3 closed: all six Definition of Done items ✔, item 6 with the owner's review. Phases 4+ are planned, not built. |

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
| Public documentation reads (PyPI JSON for `tree-sitter-language-pack`, `tree-sitter`, `tree-sitter-python`; the language pack's README and docs; py-tree-sitter docs) | 2026-11-25 22:19:46Z–22:22:22Z | pypi.org, github.com, docs.tree-sitter-language-pack.xberg.io, tree-sitter.github.io | HTTP GETs; no repository data | 2 / preparation | [S] |
| Public documentation re-read (PyPI JSON for `tree-sitter` and `tree-sitter-python`: releases, dates, sizes, hashes, yanked flags) | 2026-11-25 23:15:25Z | pypi.org | HTTP GETs; no repository data | 2 / Steps 1–2 | [S] |
| `uv add "tree-sitter==0.26.1" "tree-sitter-python==0.25.0"` under `umask 022` (2 wheels, 782,192 B; hashes match PyPI) | 2026-11-25 23:18:10–23:18:11Z | pypi.org, files.pythonhosted.org | Package names and versions, uv's User-Agent | 2 / Steps 1–2 | [M] |
| Phase 2 Steps 3–6: no download, no installation, no documentation read | 2026-11-26 | none | Nothing | 2 / Steps 3–6 | [M] |
| Phase 2 Step 7: no download, no installation, no documentation read | 2026-11-26 | none | Nothing | 2 / Step 7 | [M] |
| Public documentation reads (PyPI JSON for `chromadb`, `llama-index-vector-stores-chroma`, `llama-index-embeddings-ollama`; their GitHub sources; Chroma 1.5.9 sources; Ollama `docs/api.md`; the nomic-embed-text v1.5 model card and Ollama library page) | 2026-11-26 17:03:32–17:04:59Z | pypi.org, raw.githubusercontent.com, huggingface.co, ollama.com | HTTP GETs; no repository data | 3 / preparation | [S] |
| Unintended: an unquoted `--noproxy *` made `curl` treat this repository's root file names as hosts: about 42 DNS lookups, and 3 `GET /` to `README.md` (a real domain, which answered 301) | 2026-11-26, just after 17:05:52Z | DNS resolver; the `README.md` host | Those file names and curl's User-Agent; no content, corpus or personal data | 3 / preparation (disclosed, owner chose to continue) | [S] |
| PyPI JSON for 11 `chromadb` dependencies (size estimate) | 2026-11-26 17:34:34Z | pypi.org | HTTP GETs; no repository data | 3 / preparation | [S] |
| Throwaway spike: `uv add --no-sync` on scratch copies of the lock, then install into a temporary venv with its own cache, deleted afterwards (about 84 MB) | 2026-11-26 17:36:34–17:37:25Z | pypi.org, files.pythonhosted.org | Package names and versions, uv's User-Agent | 3 / preparation (owner's OK) | [M] |
| Blocked, nothing sent: Chroma's default embedding function tried to resolve its model host when invoked on purpose; `NetworkGuard` stopped it before DNS and `strace` showed no connect | 2026-11-26 17:38:36Z | chroma-onnx-models.s3.amazonaws.com (never reached) | Nothing | 3 / preparation (disclosed) | [M] |
| PyPI JSON for `chromadb` 1.5.9 and `llama-index-vector-stores-chroma` 0.6.0 (dates, yanked flags) | 2026-11-27 10:37:49–10:37:50Z | pypi.org | HTTP GETs; no repository data | 3 / Steps 1–4 | [M] |
| `uv add "chromadb==1.5.9" "llama-index-vector-stores-chroma==0.6.0"` under `umask 022` (55 packages, 83,866,105 B; no locked version moved) | 2026-11-27 10:52:56–10:53:05Z | pypi.org, files.pythonhosted.org | Package names and versions, uv's User-Agent | 3 / Steps 1–4 (owner's OK) | [M] |
| Phase 3 Steps 1–4 at run time (`make index`, tests): loopback to Ollama only | 2026-11-27 | none | Nothing | 3 / Steps 1–4 | [M] |
| Phase 3 Steps 5–7: no download, no installation, no documentation read; at run time loopback to Ollama only | 2026-11-27 | none | Nothing | 3 / Steps 5–7 | [M] |
| Notes pasted into the assistant conversation by the owner (written locally): fragments of the golden queries' text, the names of the 4 placeholders and a description of one corpus chunk. They were not repeated or used | 2026-11-27 about 18:10Z | The assistant's provider | That text | 3 / Gate 3 | [S] |
| Notes pasted into the assistant conversation by the owner (written locally, the Gate 3 review): headings of corpus sections, terms from the golden queries' text and descriptions of what some chunks hold. They were not repeated or used; 14.12 records the review by ID and category only | 2026-11-27 about 19:05Z | The assistant's provider | That text | 3 / Gate 3 | [S] |

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
| Guard-independent network check (`strace -f -e trace=network`, `bwrap --unshare-net`) | Native code, such as tree-sitter's C core, that the Python guard cannot see | Hosts without `strace` or `bwrap`; it is run by hand (Appendix S1), not by the suite | 0 network system calls; segmenting works with no network [M] (10.4) |
| Guard-independent check of the full chunking pipeline (Appendix S3) | Native code anywhere in import, segmentation and node building | Same as above. Every process that imports `llama_index.core` makes urllib3's loopback IPv6 probe (`socket` + `bind` to `::1`, no connect), classified as benign | `bwrap --unshare-net`: works; `strace`: no connect or send to any address [M] (11.7) |
| `.venv` without hardlinks or group write (`link-mode = "copy"`, `umask 022` in `make install`) | `llama-index-core` uses the NLTK and tiktoken data bundled in its wheel, so their first use needs no download | A `.venv` created another way (plain `uv sync` with a group-writable umask, other tools) | `tests/test_ingestion_privacy.py`: load, tokenizer, stopwords and punkt in a fresh interpreter, 0 violations, 0 files written [M] |
| `models_present` check | Rejects models with no local weights (remote/cloud) | — | `tests/test_health_check.py` [M] |
| Socket sampling | Ollama's TCP/UDP sockets to non-loopback peers | Connections shorter than the 1 s interval; lookups through the local resolver (127.0.0.53) | 579 samples, 0 non-loopback [H] |
| Offline test | Generation and embedding with the network down | — | 22:25Z, exit 0; `nmcli networking connectivity` → `none`; `ip route show default` → empty [U] |
| Chroma client factory (`src/indexing/vector_store.py`) | Every setting that decides where Chroma connects is pinned (no `.env`, no `CHROMA_*` overrides), and the effective settings are verified; no default embedding function; telemetry off | Chroma's native core, which the guard cannot see (covered by `strace` and `bwrap`, 13.7) | `tests/test_vector_store.py` (hostile `.env` and environment, in-process and fresh interpreter); static audit [M] |
| Content-logger containment | `llama_index` and `chromadb` never log below WARNING, so retrieved code never reaches the logs | Other libraries added later | Sentinel tests with the root logger at DEBUG, with a negative control [M] |

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

Update after Phase 2 Steps 1–2 (2026-11-25): chunking uses `tree-sitter` 0.26.1 and `tree-sitter-python` 0.25.0 through an own segmenter, not `CodeSplitter` (section 10).

## 6. Known gaps and open questions

| Gap | Status | Procedure to reproduce (not run) |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` vs the Modelfile | Not verified | The owner adds `Environment="OLLAMA_CONTEXT_LENGTH=8192"` in a second drop-in and restarts Ollama. Load the derived model with no per-request `num_ctx` and read `context_length` from `/api/ps` (16384 means the Modelfile wins). Remove the drop-in and restart. |
| Truncation limit at `num_ctx` 16384 | Not verified; only `limit=2050` at 4096 was observed | Run a needle prompt of about 9,000 tokens (about 490 filler lines at 18.3 tokens/line) on the derived model. Compare `prompt_eval_count` with the server log line `truncating input prompt limit=…`. About 13 min or more at 11.8 tok/s [E]. |
| `/api/embed` above 2048 tokens | Closed (section 13.3): Ollama 0.34.0 truncates silently to 2048 tokens by default and returns HTTP 400 with `truncate=false`; the project always sends `truncate=false` [M] | Embed an input of about 10 KB; check whether the server errors or truncates (compare `prompt_eval_count` with the input length). |
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

## 10. Phase 2 — Syntax-aware chunking (Steps 1–2)

Step 1 explains why blind chunking destroys code; Step 2 asks for chunking along the code's structure with a parser per language [D: rag_phase2.md Steps 1–2]. Here Step 1 becomes executable invariants and Step 2 a segmenter that turns the text of a `.py` file into spans. No LlamaIndex node is built yet: `chunk_documents`, `symbol`, line numbers, IDs and the fallback splitters belong to Steps 3–6. Code: `src/ingestion/{spans,parser_registry,segmenter}.py`, `ChunkingRules` in `src/ingestion/chunker.py`; checkers in `tests/chunking_invariants.py`.

**10.1 Decisions**

| ID | Decision | Source |
|---|---|---|
| P1 | `tree-sitter` 0.26.1 and `tree-sitter-python` 0.25.0 only, hash-pinned in `uv.lock`; grammars compiled into the wheels, nothing downloaded at run time. No `tree-sitter-language-pack` (its 1.x line downloads parsers on first use, from native code) and no `llama-index` metapackage. `tree-sitter` is the newest release at least 14 days old (0.27.0 was 2 days old) | [U] 2026-11-25 |
| P2 | Own segmenter on `tree_sitter.Parser`, not `CodeSplitter`: the whole file parsed once; one span per top-level definition; a class over the cap becomes a header, one span per method and class blocks; a definition over the cap is split at statement boundaries and marked forced; module code in contiguous blocks; cap 1500 characters; no overlap | [U] 2026-11-25 |
| P3 | Own fallback splitters per file type (Step 5) | [U] 2026-11-25 |
| P4 | Node metadata, `symbol`, line numbers and UUID5 IDs (Step 4) | [U] 2026-11-25 |
| P5 | Python only, with a data-driven label → grammar registry; every other label has no parser | [U] 2026-11-25 |
| P6 | Key test = invariants on the frozen corpus + a stats manifest + the owner's review (Step 7) | [U] 2026-11-25 |
| G1-Q1 | Leading comments: comment-only runs before a definition join it, even across blank lines, when only comments separate it from the previous code; a shebang or encoding line never does | [U] 2026-11-25 |
| G1-Q2 | Class over the cap: header = decorators, comments, `class` line, docstring and statements up to the first method; then one span per method; statements between or after methods form class blocks; nested classes recurse; a class without methods is a forced header split | [U] 2026-11-25 |
| G1-Q3 | A statement or line over the cap on its own steps down: inner statements and clauses, then lines, then characters (breaking after a word), always forced | [U] 2026-11-25 |
| G1-Q4 | Labels without a grammar and without a P3 splitter get a generic text splitter in Step 5: blank-line blocks merged up to the cap, then lines, then characters | [U] 2026-11-25 |
| G1-policies | Boundaries, grouping, `forced`, comments and cap, name paths, 1500 characters, `Language` cached and one `Parser` per call (10.3) | [U] 2026-11-25 |
| G2-Q1 | Syntax errors are a status, not an exception: top-level nodes without errors are segmented normally; a top-level node with errors is treated as module code, split at whole lines if needed, and counted. This makes the syntax-error branch of G1-Q4 unnecessary | [U] 2026-11-25 |
| G2-Q2 | Definitions inside a module-level `if`/`try`/`with` are module code (0 in the corpus) | [U] 2026-11-25; [M] |

**Deviations from rag_phase2.md (Steps 1–2)**

| Doc | This project | Why |
|---|---|---|
| `uv add llama-index tree-sitter tree-sitter-language-pack` | `uv add "tree-sitter==0.26.1" "tree-sitter-python==0.25.0"` under `umask 022` | Q5, P1 |
| `tree-sitter-language-pack`, "pre-built parsers, hassle-free" | Not used | 1.x fetches parsers on first use from GitHub Releases through its Rust core, invisible to `NetworkGuard`, into `~/.cache`; 0.9.1 is a 19.5 MB wheel pinned to `tree-sitter` 0.23.2 [public docs, 10.6] |
| `CodeSplitter.from_defaults(chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)` | Own segmenter; 1500 characters; no line target; no overlap | In the installed 0.14.26 `code.py`, `chunk_lines` and the overlap are never used; adjacent definitions are packed greedily into one chunk; chunks are stripped and located with `str.find` |
| `get_parser(language)` per splitter, and per chunk in Step 4 | Registry: label → grammar module; `Language` cached per process; the whole file parsed once | No downloads; a label is not a grammar name; a fragment is not a valid module |
| 8 syntax-aware languages | `python` only | P5 |
| `max_chars` as a safeguard | The cap is an invariant (I6); what does not fit is split and marked forced | P2 |

**10.2 Invariants (Step 1).** A coherent unit, as checkers that return violations (`tests/chunking_invariants.py`); each has a negative test proving it can fail. The oracle for I5 and I7 is the standard library's `ast` (line numbers only), not the tree that produced the spans; it does not apply to text that is not valid Python.

| ID | Invariant |
|---|---|
| I1 | `0 <= start < end <= len(text)`; the span is the literal slice `text[start:end]` and does not end on whitespace |
| I2 | Spans are sorted and do not overlap |
| I3 | Every non-whitespace character belongs to exactly one span |
| I4 | A span starts at the beginning of a line and ends at the end of its content, except forced pieces of a single line longer than the cap |
| I5 | A function, method or class that fits within the cap is never split |
| I6 | No span exceeds the cap |
| I7 | A function or method split across spans has all those spans forced; a forced span always overlaps something that does not fit |
| I8 | Two runs give identical results |

Blind splitting violates them: `naive_line_split` every 4 lines (the doc's Step 1 example) cuts `validar_token` in two (I5), and a split every 120 characters also cuts it mid-line (I4); the segmenter keeps it in one span. The doc's `test_function_stays_whole` is vacuous: its assertions also pass for a "chunker" that returns the whole file (`tests/test_chunking.py`).

**10.3 Span model and policies (Step 2)**
- **Result:** `segment(text, spec, *, max_chars) -> Segmentation(spans, status, error_regions)`. `Span(start, end, kind, forced, name_path)` holds **character** offsets. The kind is one of `function`, `class`, `class_header`, `method`, `class_block`, `module_block`. The name path is `("f",)`, `("C",)` or `("C", "m")`, and empty for module code. The status is `clean` or `has_errors`. No line numbers or IDs: Step 4 derives them from the offsets without re-parsing.
- **Boundaries:**
  - A span starts at column 0 of its first non-blank line and ends after its last non-whitespace character; blank lines between spans belong to none.
  - Rows come from the tree and are mapped to characters through a line-start index of the `str`, split on `\n` only. Byte offsets are never used to index the `str`, and `str.find` is never used to locate text.
  - An end-of-line comment stays with its line.
- **Grouping:**
  - Every top-level definition that fits is its own span, even when tiny (definitions are never merged).
  - Module code is the complement of the definitions, so coverage holds by construction.
- **Over the cap:**
  - Whole units are packed greedily.
  - A header (decorators, `def`/`class`/`if`… line) keeps its first part.
  - What still does not fit steps down to statements and clauses, then lines, then characters.
  - `forced` marks the split of a function, method, class header, statement or line. It lives on the span, never in node metadata (P4: no extra keys).
- **Comments:** leading comments that would push a fitting definition over the cap get their own span; the definition is not split. Since Step 7, the same holds when they cannot share a span with the first part of a split definition (section 12.1).
- **Parsers:**
  - `GrammarSpec` holds node types and field names as data; every declared type is checked against the installed grammar.
  - A label without an entry returns no parser as a value; a registered grammar that cannot load raises `ParserUnavailableError` (run `make install`).
  - Importing the registry loads no grammar; `Language` is cached per process; each call gets a new `Parser`.
- **Errors:** `ChunkingError`, `InvalidChunkingRulesError` and `ParserUnavailableError` under `IngestionError` (`src/errors.py`). A file with syntax errors is a result with a status, not an exception.

**10.4 Spike results** [M], 2026-11-25, on synthetic snippets only.

| Topic | Result |
|---|---|
| API and ABI | `Language(tree_sitter_python.language())`, ABI 15 = `tree_sitter.LANGUAGE_VERSION` (minimum 13); `Parser(language).parse(bytes)`; `has_error`, `is_error`, `is_missing`; both packages ship `py.typed` and `.pyi` stubs, so strict mypy needs no override |
| Positions | Byte offsets and byte columns; rows split on `\n` only (`\x0c` and ` ` do not start a row); the root ends at column 0 of the next row |
| Tree shape | `decorated_definition` = decorators (a comment between them included) + definition; `async def` is a `function_definition`; methods and nested classes sit in the class `block`; clauses are `elif_clause`, `else_clause`, `except_clause`, `finally_clause`, `case_clause` (no `except_group_clause` type in this grammar); `a = 1; b = 2` gives two siblings on one row |
| Comments | An end-of-line comment is a sibling on its row; a comment indented at the end of a block stays inside the definition; a column-0 comment after a block is module-level, but between two methods it stays inside the class |
| Errors | A broken definition keeps its type and span (`has_error`) and its neighbours stay clean; an unclosed parenthesis yields a top-level `ERROR` node that can swallow the rest of the file; a stray indent is not flagged although Python rejects it |
| Privacy | Fresh interpreter under `NetworkGuard`: 0 violations; `strace -f -e trace=network`: 0 network system calls (a positive control with one socket is detected); `bwrap --unshare-net`: works without a network; `HOME`, `XDG_CACHE_HOME` and `TMPDIR` in an empty directory stayed empty; nothing under `~/.cache` |

**10.5 Result on the corpus** [M], 2026-11-25 at 23:38Z, under `NetworkGuard` (0 violations); the corpus at `5c288080…` with the same HEAD and an empty `git status` before and after.

| Finding | Value |
|---|---|
| Input | 32 `.py` documents, 192,670 B, from the loader |
| Spans | 279: `function` 179, `module_block` 71, `class` 22, `method` 6, `class_header` 1 |
| Forced spans | 77: `function` 64, `method` 5, `module_block` 8 |
| Span size (characters) | p50 515, p90 1,454, max 1,498 |
| Syntax errors | 0 files |
| Invariant violations (I1–I8) | 0 |
| Time | 14.1 ms for the 32 files (grammar already loaded) |
| Step 1 evidence | Of 188 definitions (`ast`, any depth), 33 exceed the cap. The segmenter splits exactly those 33. A blind split every 40 lines also splits 39 definitions that fit; one every 1,500 characters splits 46 |

**10.6 Known limitations and open questions**

| Item | Effect or question |
|---|---|
| tree-sitter is lenient | A stray indent parses cleanly; an `ERROR` node can swallow the code that follows it, which is then segmented as module code |
| Recursion | Stepping down follows statement nesting; extremely deep nesting could reach Python's recursion limit |
| `Parser` per call | Chosen because sharing a parser across threads is not documented as safe [H] |
| Input precondition | Text normalized by the loader (`\n` only, no BOM, no NUL); a lone `\r` is not a line break for the line index |
| Step 5 (was open) | Built: the generic text splitter for labels without a grammar or a dedicated splitter, and for files whose strategy fails (section 11.4) |
| Step 4 (was open) | Resolved: only `file_path`, `language` and `symbol` are visible to the embedding model and the LLM (G3-Q1, section 11.3) |

## 11. Phase 2 — Metadata, fallbacks and the chunker (Steps 3–6)

Steps 3–6 turn the spans of section 10 into the nodes that later phases embed, retrieve and cite, for the whole repository. Step 3 dispatches each document by language, Step 4 turns spans into `TextNode` objects with their metadata, Step 5 adds the fallbacks, and Step 6 builds the chunker, its report and its CLI [D: rag_phase2.md Steps 3–6]. In the corpus, 56 of 88 documents have no grammar, so the fallbacks are the common case. Code:
- `src/ingestion/strategies.py`: dispatch;
- `src/ingestion/nodes.py`: nodes;
- `src/ingestion/{markdown_splitter,config_splitter,text_splitter}.py`: fallbacks;
- `src/ingestion/lines.py`: line arithmetic shared with the segmenter;
- `src/ingestion/chunk_report.py`: the report;
- `src/ingestion/chunker.py`: API and CLI.

**11.1 Decisions**

| ID | Decision | Source |
|---|---|---|
| G3-Q1 | Visible to the embedding model and the LLM: `file_path`, `language` and `symbol` only. `start_line`, `end_line` and the four hidden document keys are hidden from both | [U] 2026-11-26 |
| G3-Q2 | `symbol`: the qualified name in Python (`Class.method`). In Markdown, the heading path joined with ` > ` (empty headings skipped), bounded to 120 characters and keeping the most specific end. No symbol for configuration or generic text | [U] 2026-11-26 |
| G3-Q3 | A Markdown heading with no body joins the following section, which keeps its own heading path; nothing else is merged | [U] 2026-11-26 |
| G3-Q4 | A YAML block over the cap is split at the keys or list items of the next indentation level, recursively, each kept whole when it fits; then blank lines, lines, characters | [U] 2026-11-26 |
| G3-Q5 | A file whose strategy raises `RecursionError` or `ValueError` is segmented as generic text, with reason `segmentation_failed`, a WARNING (path at DEBUG only) and a counter. `ParserUnavailableError` and invalid input stay fatal | [U] 2026-11-26 |
| G3-Q6 | The line index and the character pieces moved from the segmenter to `lines.py` (non-additive). The spans of the 32 corpus `.py` files are identical before and after (digest `263c2a4d…`, 279 spans) | [U] 2026-11-26; [M] |
| G3-policies | Approved as listed:<br>• Dispatch per document, in input order, then by position.<br>• Label → strategy map (11.2).<br>• Every strategy returns a `Segmentation`; `SpanKind` grew additively (`section`, `config_block`, `text_block`).<br>• Cap 1500, no overlap, whole lines.<br>• Markdown fences and config blocks that fit are never split.<br>• `forced` on spans and the report only.<br>• Input validation and UUID5 IDs (11.3).<br>• Key order: the six document keys, `symbol`, `start_line`, `end_line` | [U] 2026-11-26 |
| Corpus fix | Column-0 comments that would push a configuration block that fits over the cap get their own chunk, and the block is not split: the same rule as for Python definitions. Found by invariant F1 on the corpus and reproduced by a synthetic test | [M] 2026-11-26 |
| IPv6 probe | `strace` shows one `socket(AF_INET6)` + `bind` to `::1` in every process that imports `llama_index.core`: urllib3's import-time IPv6 capability probe (`urllib3/util/connection.py:137`). It is present since Phase 1. There is no `connect`, no send and no DNS. Classified as benign loopback; the guard-independent criterion is refined (11.7) | [U] 2026-11-26; [M] |

**Deviations from rag_phase2.md (Steps 3–6)**

| Doc | This project | Why |
|---|---|---|
| Group documents by `language`, then split each group | Per-document dispatch; output in input order, then by position | Determinism: ordering must not depend on grouping |
| 8 syntax-aware languages; everything else `SentenceSplitter(512, 50)` | `python` syntactic; Markdown, configuration and generic text splitters of our own | P5, P3, G1-Q4. On this corpus the doc's splitter started 80 of its 195 fallback chunks mid-line and cut 7 code blocks and 5 tables ([S], Phase 2 preparation) |
| `CodeSplitter` with `get_parser(language)` | The section 10 segmenter, through the registry | P1, P2 |
| "CodeSplitter inherits the document metadata automatically" | A fresh metadata dict per node, in a fixed key order | P4: the node hash depends on order |
| `end_line = count("\n", 0, end_char_idx) + 1`, omitted when the index is missing | `1 + newlines before the offset` of the last character, with `bisect`; never omitted | R4; Phase 5 prints `?` for a missing line |
| `extract_symbol_name` re-parses each chunk, sees only direct children (no decorated definitions, no methods) and takes the first | `symbol` comes from the span's name path; nothing is re-parsed | Section 10.3 |
| `try: … except Exception:` degrades a whole language | Per-file degradation, for `RecursionError` and `ValueError` only | Ruff `BLE`; G3-Q5 |
| Default IDs (`uuid4`) | UUID5 of a fixed namespace, `doc_id` and the offsets | P4: idempotent re-indexing; Phase 6 start-up matches Phase 3 |
| `print` in `__main__` | CLI with JSON logs and the loader's exit codes | Ruff `T20` |
| Step 7 tests build `Document` objects without `id_` | Input validation rejects them (`InvalidDocumentError`) | A random `doc_id` makes node IDs random |

**11.2 Strategy map**

| Labels | Strategy | Segmenter | `symbol` | Reason |
|---|---|---|---|---|
| `python` (from the parser registry) | `syntactic` | `segment` (section 10) | `Class.method` | — |
| `markdown` | `markdown` | `split_markdown` | heading path, ≤ 120 chars | `no_parser` |
| `yaml`, `toml`, `dockerfile`, `makefile` | `config` | `split_yaml`, `split_toml`, `split_dockerfile`, `split_makefile` | none | `no_parser` |
| `javascript`, `typescript`, `java`, `go`, `rust`, `c`, `cpp`, `ruby`, `csharp`, `php`, any unknown label | `text` | `split_text` | none | `no_parser` |
| any label whose segmenter failed on a file | `text` | `split_text` | none | `segmentation_failed` |

A test fails when the loader emits a label that is in neither the registry nor the map.

**11.3 Node contract**

| Field | Value |
|---|---|
| `text` | The literal slice `document.text[start:end]`, never stripped; `start_char_idx` and `end_char_idx` are the span offsets |
| `metadata` | In this order: `file_path` (str), `language` (str), `size_bytes` (int), `num_lines` (int), `content_hash` (str), `directory` (str), `symbol` (str, only when the span has a name path; never `None` or `""`), `start_line` (int), `end_line` (int). Lines are 1-based and inclusive; `end_line` is the line of the last character |
| Visible to the embedder and the LLM | `file_path`, `language`, `symbol`. Every other key is in fresh `excluded_embed_metadata_keys` and `excluded_llm_metadata_keys` lists per node |
| `id_` | `str(uuid5(NODE_ID_NAMESPACE, doc_id + "\0" + start + "\0" + end))`, with `NODE_ID_NAMESPACE = 41a05dbb-07dd-4b83-840e-8c25648f281a` (a fixed literal; a golden test pins one ID) |
| Relationships | `SOURCE` only (the document); `ref_doc_id` = `doc_id` = `file_path`; no previous or next links |
| Hash | LlamaIndex's `TextNode.hash` = sha256 of text + `str(metadata)`: hidden keys and key order included. Phase 4's reciprocal-rank fusion deduplicates on it. The vector-store round trip (`node_to_metadata_dict` with `flat_metadata=True` and back) keeps hash, ID, offsets and key order |

**11.4 Fallbacks (Step 5)** — custom, iterative and linear; no new library, no `SentenceSplitter` and no `MarkdownNodeParser`. Every strategy keeps I1–I4, I6 and I8.
- **Markdown:**
  - ATX and setext headings outside fences. Fences open with ``` or `~~~` at any indentation, close with a run at least as long, and an unclosed fence runs to the end of the file. `---` after a blank line is a thematic break; a leading `---` block is front matter.
  - The preamble is its own section, and a heading-only section joins the next (G3-Q3).
  - A section over the cap splits at blank lines outside fences, then lines, then characters; its heading stays with its first part and every piece is forced. A fence that fits is never split.
- **Configuration:**
  - Top-level blocks are column-0 YAML keys, list items and `---`; TOML tables (outside multi-line strings and arrays); Dockerfile instructions with continuations and heredocs (a `FROM` always starts a chunk); Makefile rules with their recipes, `define … endef`, conditionals and continuations.
  - Column-0 comments stay with the block below them, unless they would push a fitting block over the cap (Corpus fix).
  - Whole blocks merge greedily up to the cap. A block over the cap steps down per G3-Q4 (YAML) or at blank lines (TOML: between statements only), then lines, then characters.
- **Generic text:** blank-line blocks merged up to the cap, then lines, then characters.
- **New invariants:**
  - F1: a fence or a top-level configuration block that fits is never split.
  - F2: a forced span overlaps a natural unit or a line that does not fit.

  Their oracles are simple scanners in `tests/chunking_invariants.py`, independent of the splitters, and each has a negative test.

**11.5 API, report and CLI (Step 6)**
- **API:** `chunk_documents(documents, *, rules=None) -> list[BaseNode]` and `chunk_documents_with_report(documents, *, rules=None) -> ChunkResult(nodes, spans, report)`. Both validate the input (all six loader keys with their types, `doc_id == file_path`, no duplicates; otherwise `InvalidDocumentError`) and dispatch, segment, build nodes and report. No I/O, no network, no mutable state; no module in `src/` imports `chunker.py`.
- **`ChunkReport`:** frozen, no contents, no timings:
  - documents and nodes, by language and by strategy; nodes by kind;
  - `forced`, in total and by kind; fallback reasons;
  - syntax-error files and regions; degraded files; documents without nodes;
  - node size p50/p90/max, using nearest-rank percentiles: rank `ceil(p/100·n)`;
  - the longest node as the embedder sees it.
- **Logs:** one INFO `chunking.done` with counts and `duration_ms`; WARNING `chunking.degraded` without the path; paths at DEBUG only.
- **CLI:** `python -m src.ingestion.chunker [--repo P] [--rules-file F] [--json | --list | --show FILE_PATH]` (`make chunk CHUNK_FLAGS=…`).
  - Runs under `NetworkGuard` and uses the loader's exit codes (0, 1, 10, 14, 20, 21), plus 2 when `--show` names a path that is not a loaded document; it never opens that path.
  - `--show` escapes control characters except newline and tab.
  - `--list` and `--show` print symbols and content and are run only by the owner.

**11.6 Tests.** All in `tests/test_chunking.py` (275 tests at the end of this step). They cover:
- **Dispatch:** every label's strategy, unknown labels and the adapter.
- **Nodes:** keys, types and order; `symbol` in every situation; lines against an independent oracle over all section 10 cases; EMBED, LLM and NONE views; a golden ID; the vector-store round trip; flat-metadata behaviour (`None` accepted, lists rejected); the shared-hash edge case.
- **Fallbacks:** F1, F2 and the shared invariants over 18 cases (fences with `#`, unclosed fence, no headings, empty heading, YAML block scalar with blank lines, `---` documents, TOML multi-line string with `[`, Dockerfile heredoc and continuation, Makefile recipe with blank lines, non-ASCII); exact cap and cap + 1; pathological inputs (100,000-character line, thousands of `#`, backticks, dashes and brackets) under 2 s.
- **Chunker:** all four strategies; ordering with interleaved languages; report consistency; empty and invalid input; the degradation policy; exit codes; `--show` escaping and unknown paths; logs without content; identical nodes across processes with different `PYTHONHASHSEED`; two corpus regressions reproduced synthetically.

**11.7 Privacy** [M], 2026-11-26
- **Fresh interpreter:** `tests/test_ingestion_privacy.py` runs `chunk_documents_with_report` on seven labels (all four strategies) in a fresh interpreter with `NetworkGuard` active before any import. It reads every `MetadataMode` view and does the vector-store round trip, with `HOME`, `XDG_CACHE_HOME` and `TMPDIR` in an empty directory: 0 violations, 0 files written.
- **Guard-independent check (Appendix S3):** under `bwrap --unshare-net` the full run works (99 nodes, all strategies). Under `strace -f -e trace=network` the only calls are urllib3's IPv6 probe (`socket(AF_INET6)` + `bind` to `::1`).
- **Refined criterion for the full pipeline:** no `connect`, `sendto` or `sendmsg` to a non-loopback address, and a successful run with no network namespace. The segmenter alone still makes 0 network calls.

**11.8 Result on the corpus** [M], 2026-11-26 at 10:20–10:23Z, under `NetworkGuard` (0 violations); the corpus at `5c288080…` with the same HEAD and an empty `git status` before and after.

| Finding | Value |
|---|---|
| Documents → nodes | 88 → 658. By strategy: syntactic 32 → 279, markdown 46 → 340, config 10 → 39. By language: python 279, markdown 340, yaml 29, toml 4, dockerfile 4, makefile 2 |
| Nodes by kind | `section` 340, `function` 179, `module_block` 71, `config_block` 39, `class` 22, `method` 6, `class_header` 1 |
| Forced | 176: `section` 77, `function` 64, `config_block` 22, `module_block` 8, `method` 5 |
| Fallback reasons, syntax errors, degraded files, documents without nodes | `no_parser` 56; 0; 0; 0 |
| Node size (characters) | p50 683, p90 1,439, max 1,500 (nearest-rank); longest embedding input 1,683 characters; longest `symbol` 120 |
| Invariants | 0 violations (I1–I8 for Python; I1–I4, I6, I8, F1, F2 for the fallbacks). Every node is the exact slice with exact lines; IDs unique; two runs identical; round trip keeps every hash; 0 nodes share a hash |
| No regression on Python | 279 nodes (`function` 179, `module_block` 71, `class` 22, `method` 6, `class_header` 1), 77 forced, p50/p90/max 515/1,454/1,498 with the section 10.5 method |
| Time | `chunk_documents` on the 88 documents: 62–130 ms over 5 runs (grammar loaded); Phase 6 pays this at every start-up, on top of the 6–16 ms load |
| Embedder headroom [E] | 1,683 characters ≈ 842 tokens at a conservative 2 characters per token, about 41% of the 2,048-token limit |

**11.9 Known limitations and risks for Phases 3–4**

| Item | Effect |
|---|---|
| IDs follow offsets | Any edit above a span changes its ID. Re-indexing a file must delete its nodes by `ref_doc_id` first (Phase 3) |
| Equal hashes | Two forced pieces with the same text on the same line share a hash (different IDs), and fusion would collapse them (0 in the corpus; P4 adds no key) |
| `None` in metadata | LlamaIndex's flat check accepts `None`; this contract never emits it (keys are omitted) |
| BM25 [H] | What `BM25Retriever` indexes (text only, or visible metadata too) is not verified: the package is not installed |
| Markdown | Setext headings under a list item, HTML blocks and indented code are approximated; a heading path over 120 characters is trimmed from the left |
| Configuration | No YAML, TOML or Make parser: anchors, flow collections spanning lines and Dockerfile `escape` directives are handled as plain lines |
| Start-up cost | 62–130 ms per start-up on this corpus, linear in repository size |

## 12. Phase 2 — Step 7 and closure

Step 7 asks for tests showing that a function is not split and that the metadata is correct. The Definition of Done's key test asks whether the chunks of a real repository are meaningful units, not cuts in the middle of a line [D: rag_phase2.md Step 7, Verification]. This project answers with five pieces of work:
- the doc's three tests rewritten at node level, each with a negative control;
- an automated key test against a versioned manifest (P6);
- a smoke test on a second codebase;
- a structural review of every corpus chunk;
- one fix in `src/`, found by the smoke test.

Code: `tests/chunk_validation.py` (content-free validation), `tests/chunk_manifest.py` (manifest), `tests/fixtures/chunk_manifest.json`, `tests/test_chunking_corpus_integration.py` (key test) and `src/ingestion/segmenter.py` (the fix).

**12.1 Decisions**

| ID | Decision | Source |
|---|---|---|
| G4-Q1 | Manifest = the `ChunkReport` + one row per document: path, strategy, nodes, forced spans, and a digest of its (ID, hash) pairs | [U] 2026-11-26 |
| G4-Q2 | Review sample: stratified and deterministic (10–14 files planned), plus `--list` of every node. The result has 9 files, because two criteria fell on files already chosen | [U] 2026-11-26 |
| G4-Q3 | Fix only demonstrated defects; anything merely improvable becomes a Phase 7 experiment | [U] 2026-11-26 |
| G4-Q4 | Second codebase = this repository: invariants and `--json` aggregates only, no downloads | [U] 2026-11-26 |
| G4-policies | Approved as listed:<br>• The manifest is compared by exact equality: UTF-8, LF, canonical JSON, versions and regeneration command included.<br>• The key test carries the `integration` marker. It skips only when `REPO_PATH` is missing or its HEAD is not the frozen commit; a modified tree fails.<br>• Failures are content-free.<br>• `make test-corpus` runs both key tests.<br>• Tokens are not measured with Ollama (11.8 [E]) | [U] 2026-11-26 |
| Step 7 fix | Leading comments that cannot share a span with the first part of a split definition (a function, or a class header) get their own unforced block span, with no `symbol`. This extends the 10.3 comment rule. Before the fix, the smoke test found a comment-only chunk that was forced and carried the function's `symbol` (I7). The case was reproduced synthetically (2 tests) and the fix approved. Corpus: 0 such cases; the manifest is unchanged | [U] 2026-11-26; [M] |
| DoD #5 | The owner asked the agent to do the review ("try it yourself"). After the warning that reading chunks would send corpus content off the machine and change P6, the owner chose a content-free review by the agent. Asked to confirm, the owner answered "check it yourself". Recorded as judged by the agent, on the structural review of 12.5, delegated by the owner. No corpus content left the machine. P6 is satisfied as: invariants, manifest and the delegated structural review | [U] 2026-11-26 |

**12.2 Tests (Step 7).** In `tests/test_chunking.py`: 317 tests (275 at the end of Steps 3–6).

| Doc test | Here | Why it is not vacuous |
|---|---|---|
| `test_function_stays_whole` (substring checks) | `test_each_function_is_one_node_with_its_whole_source`: a decorated, an `async` and an over-the-cap function. Each function is exactly one node whose text is its whole source, or, when over the cap, consecutive forced pieces that cover it | The same assertions fail for a whole-file "chunker" and for `naive_line_split` every 4 lines (a negative test) |
| `test_extracts_symbol_name` (`extract_symbol_name` does not exist here) | `test_symbols_on_nodes_for_every_kind_of_unit` (function, class, method, nested method, Markdown heading paths) and `test_nodes_without_a_name_have_no_symbol_key` (module code, YAML, Go) | It checks the exact `symbol` on the nodes Phase 5 cites, and checks that the key is absent rather than empty |
| `test_metadata_inherited_and_enriched` (key presence) | `test_metadata_is_inherited_exactly_and_enriched_with_exact_lines` | Exact values of the six inherited keys and exact `start_line`/`end_line`, checked against the lines' text |

The key-test helpers have their own synthetic tests, which need no corpus:
- **Manifest reader:** strict, with 12 malformed cases (BOM, CRLF, no final newline, invalid JSON, keys, format, counts, strategy, digest, row keys, empty list, corpus digest); each message says how to regenerate. A non-canonical file is rejected.
- **Comparison:** by field, by path (missing or extra) and by order.
- **Digests:** the written digest definition, and a golden digest of a synthetic repository.
- **Validator:** detects 9 kinds of tampering, duplicates and degraded files, and its findings never carry text or symbols.
- **Helpers:** the review sample is deterministic; the tree snapshot sees a change.
- **Sentinel test:** a validation failure is forced in a `pytest` subprocess, and the sentinel planted in the chunked text does not appear in its output.

`make check`: 1,110 passed, 19 deselected (1,066 before Step 7).

**12.3 Key test (P6).** `tests/test_chunking_corpus_integration.py`, marker `integration`, 8 tests. `make test-corpus` now runs both key tests (17 tests, offline, no Ollama, about 2 s); `make test` does not collect them. The module chunks the corpus once, under its own `NetworkGuard`, and snapshots the tree before and after. Its tests check:
- **Manifest:** the run equals the manifest: strict parse, comparison by field and by path, and byte-identical file text.
- **Documents:** the documents are the Phase 1 manifest's files, with their labels.
- **Validation:** 0 findings over the 658 nodes:
  - strategy invariants, exact slices and offsets;
  - key order and types, visibility, relationship, recomputed UUID5 IDs;
  - lines against an independent oracle, the vector-store round trip;
  - unique IDs and hashes, order.
- **Splits:** of 188 Python definitions, 33 exceed the cap, and exactly those 33 are split; 0 definitions that fit are split.
- **Report:** 0 degraded files, 0 syntax errors; 88 → 658 nodes, 176 forced; by strategy, syntactic 279, markdown 340, config 39.
- **Determinism:** two runs are identical, and another process with `PYTHONHASHSEED=12345` gives the same corpus digest.
- **Privacy:** the corpus tree is unchanged and the guard saw 0 violations.

Failures are content-free:
- Assertions compare records (code, path, offsets, lines, length, kind), counts and digests, never corpus strings.
- I5 and I7 are re-implemented without definition names.
- Never run the key test with `-vv`, `-l` or `--showlocals`.

| Manifest field | Value |
|---|---|
| File | `tests/fixtures/chunk_manifest.json`, 21,008 B, UTF-8, LF, canonical 2-space JSON, format `local-code-rag/chunk-manifest/1` |
| Corpus | `5c288080c88f0bfec435af0093a5155baaf7f343` |
| Rules | `max_chunk_chars` 1500 |
| Versions | `tree-sitter` 0.26.1, `tree-sitter-python` 0.25.0, `llama-index-core` 0.14.26 (the node hash is LlamaIndex's) |
| Digest | Document: sha256 of the lines `<node id>:<node hash>`, in order. Corpus: sha256 of the lines `<path>:<document digest>`. Corpus digest `0ddc4d1ae608b2e92a3e6840a2d5fdd6d6fd0b3e9961b4c4d284c82f982444cd` |
| Rows | 88 (path, strategy, nodes, forced, digest), in loader order; the row sums equal the report |
| Regeneration | Only after an owner decision that changes the chunks (Appendix S5) |

Before it was versioned, the manifest was accepted by an independent script, under `NetworkGuard`, with 0 violations. The script checked that:
- HEAD is the frozen commit;
- the manifest equals a fresh run;
- paths and languages equal `tests/fixtures/corpus_manifest.tsv`;
- the rows cover every document, in order, and their sums equal the report;
- the content-free validation gives 0 findings;
- the split counts are as above.

**12.4 Second codebase** [M], 2026-11-26. This repository's working tree, Step 7 work included, under `NetworkGuard` (0 violations). These figures move with the repository; they are not a regression baseline.

| Finding | Value |
|---|---|
| Documents → nodes | 70 → 1,201. By strategy: syntactic 65 → 1,097, markdown 2 → 97, config 3 → 7 |
| Forced; reasons | 182; `no_parser` 5 |
| Syntax errors, degraded files, documents without nodes | 0; 0; 0 |
| Node size (characters) | p50 390, p90 1,265, max 1,498; longest embedding input 1,632 |
| Definitions | 993 (`ast`, any depth); 43 over the cap; exactly those 43 split |
| Content-free validation | 1 finding (I7) before the Step 7 fix, 0 after |
| Time | 87 ms |

**12.5 Review of the chunks (DoD #5)** [M], 2026-11-26.
- **Sample.** `review_sample` picked 9 files: 96 nodes, 51 of them forced.
  - Python with the most forced spans: `src/models/register.py`. It also holds the only split class.
  - Python with the most module blocks: `src/config.py`.
  - Markdown with the most nodes: `README.md`, the root README.
  - Markdown with the most forced spans: `docs/decisions/README.md`.
  - Markdown with the most code fences: `docs/decisions/0029-cd-workflow-and-ghcr-publication.md`.
  - One file per configuration format: `docker/Dockerfile`, `Makefile` and `pyproject.toml`.
  - YAML with the most forced spans: `.github/workflows/cd.yml`.
- **Owner's procedure.** Appendix S6. The commands were tested on a synthetic repository; the owner did not run them (12.1, DoD #5).
- **Structural review.** All 658 nodes were checked against independent oracles: `ast` for Python, and headings, blank lines and column 0 for the other strategies. The script printed counts and records only.

| Strategy | Whole chunks at a boundary | Forced pieces starting at a boundary | Forced pieces starting inside a unit |
|---|---|---|---|
| Python | 202 of 202. Each starts at a definition, decorator, statement or comment and ends at a statement's end | 69 of 77 | 8. Each is a line-level split of a docstring longer than the cap |
| Markdown | 263 of 263 start at a heading or are the preamble | 59 of 77 start at a heading or after a blank line | 18. Each starts at a text line (11), a list item (4) or a table row (3) inside a blank-line block longer than the cap |
| Configuration | 17 of 17 start at column 0, a nested key or list item, or a comment | 22 of 22 | 0 |

Between consecutive chunks there are 0 non-whitespace gaps. No Python definition lacks its `symbol`, and no configuration chunk has one. Every cut inside a unit falls on a unit that does not fit, as G1-Q3 requires.
- **Verdict:** ✔, judged by the agent under the owner's delegation. Not covered: whether each chunk is semantically useful for retrieval. Phase 7 measures that.

**12.6 Definition of Done → evidence**

| # | DoD item [D: rag_phase2.md, Verification] | Evidence | Status |
|---|---|---|---|
| 1 | `chunker.py` chunks code in a syntax-aware manner, by language, using tree-sitter (via `tree-sitter-language-pack`) | `tree-sitter` with the Python grammar wheel, no language pack (P1, P5). Segmenter (10.3), dispatch (11.2), 279 syntactic nodes, I1–I8 with 0 violations (12.3) | ✔ (deviation: Python only, no language pack) |
| 2 | Each chunk inherits the source file and is enriched with the symbol and line range | Node contract (11.3); 12.2; 0 findings for keys, inheritance, lines, line text and the round trip over 658 nodes | ✔ |
| 3 | A fallback guarantees coverage for languages without a parser and for large chunks | Fallbacks (11.4): 56 `no_parser` documents → 379 nodes. 0 gaps (I3), forced splits over the cap (I6, F2), 0 degraded files; `segmentation_failed` is tested | ✔ |
| 4 | The tests verify that functions are not split and that the metadata is correct | 12.2: the doc's three tests at node level, with negative controls; `make check` 1,110 passed | ✔ |
| 5 | Key test: the chunks are meaningful units (a complete function, a complete class), not cuts in the middle of a line | Automated (12.3); structural review of all 658 nodes (12.5) | ✔ (delegated, structural) |

**12.7 Deviations from rag_phase2.md (Steps 1–7, consolidated)**

| Step | Doc | This project | Where |
|---|---|---|---|
| 1 | Blind splitting shown by example | Executable invariants I1–I8, each with a negative test | 10.2 |
| 2 | `tree-sitter-language-pack`; `CodeSplitter(chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)` | `tree-sitter` + `tree-sitter-python` wheels; own segmenter; cap 1500; no overlap | 10.1 (P1, P2) |
| 2–3 | 8 syntax-aware languages; documents grouped by language | `python` only; per-document dispatch in input order | P5; 11 |
| 4 | `extract_symbol_name` re-parses; `end_line` omitted when missing; default `uuid4` IDs; the inherited metadata dict | `symbol` from the name path; lines always present; UUID5; a fresh dict per node, in a fixed key order | 11.3 |
| 5 | `SentenceSplitter(512, 50)` for everything else | Own Markdown, configuration and generic text splitters | 11.4 |
| 6 | `except Exception` degrades a whole language; `print` | Per-file degradation (`RecursionError`, `ValueError`); a CLI with JSON logs | 11.1 (G3-Q5) |
| 7 | The three tests (substring checks, a nonexistent function, key presence) | Node-level tests with negative controls | 12.2 |
| 7 | `Document` without `id_` | Rejected with `InvalidDocumentError`; tests set `id_` | 11 |
| 7 | All tests run with `make test` | Synthetic tests in `make test`; the corpus key test in `make test-corpus` | 12.3 |
| DoD | "Inspect the chunks" by eye | Exact manifest, content-free validation and a structural review | 12.3, 12.5 |

**12.8 Known limitations (Phase 2, consolidated)**

| Item | Effect |
|---|---|
| Python only | Other code languages get the generic text splitter (P5) |
| tree-sitter is lenient | A stray indent parses cleanly; an `ERROR` node can swallow the code after it, which then becomes module code (10.6) |
| Units over the cap | They are cut at lines (26 corpus pieces start inside a docstring, a paragraph, a list item or a table) or, beyond that, at characters |
| Markdown and configuration | Approximated without parsers: setext headings under list items, HTML blocks, indented code, YAML anchors, flow collections, Dockerfile `escape` (11.9) |
| IDs follow offsets | Any edit above a span changes its ID (11.9) |
| Equal hashes | Two identical forced pieces on one line share a hash (0 in the corpus) |
| Manifest | Tied to `tree-sitter`, `tree-sitter-python` and `llama-index-core` (whose node hash it uses). Upgrading any of them, or changing a chunking rule, needs an owner decision and a regeneration (S5) |
| Review | Structural, not semantic (12.5) |
| Recursion and cost | Very deep nesting could reach the recursion limit; 62–130 ms per chunking of the corpus, linear in size |

**12.9 Risks for Phases 3–8.** A read-only audit of the contract against the phase documents.

| Phase | What it consumes | Risk | Recommendation |
|---|---|---|---|
| 3 | `VectorStoreIndex(nodes)` on a persistent Chroma collection [D: rag_phase3.md Steps 4–5] | IDs change when offsets move (R11: `doc_id` is unique per repository only) | Delete a file's nodes by `ref_doc_id` before re-indexing it; prefix IDs if several repositories share a collection. Qdrant accepts UUID IDs [H] |
| 4 | `BM25Retriever.from_defaults(nodes=...)`; `QueryFusionRetriever(mode="reciprocal_rerank")`; a `MetadataFilter` on the vector retriever only [D: rag_phase4.md Steps 2, 4] | The `language` filter does not reach BM25; fusion deduplicates by `hash` (0 shared in the corpus); what BM25 indexes is [H]; R10 (`directory` `EQ` excludes subdirectories) | Filter BM25 results by the same metadata; verify BM25's indexed text once the package is installed |
| 5 | `format_context` wraps `node.get_content()` in a triple-backtick fence and cites `symbol` and lines [D: rag_phase5.md Step 2] | 35 corpus nodes (all Markdown) contain a triple-backtick fence and would close the prompt's fence early | Use a fence longer than any backtick run in the chunk, or `~~~` |
| 6 | `lifespan` re-chunks the repository for BM25 and reloads the persisted index [D: rag_phase6.md Step 2] | If the repository changed since indexing, BM25 nodes and stored vectors diverge; R12 (`/index` takes a client path) | Store the corpus digest with the index, compare it at start-up, and refuse or re-index on a mismatch |
| 7 | `retrieved_contexts` = `get_content()` per node [D: rag_phase7.md Step 5] | R13: answers not tied to a corpus version | Record the corpus commit and the chunk digest with every RAGAS run |
| 8 | `uv sync --frozen` with `UV_LINK_MODE=copy` [D: rag_phase8.md Step 1] | None found: the grammars are hash-pinned wheels in `uv.lock`, with nothing downloaded at run time | Keep `umask 022` for `uv` operations; never set `NLTK_DATA` or `TIKTOKEN_CACHE_DIR` |

R10–R13 (section 9) still apply.

**12.10 Privacy and network** [M], 2026-11-26
- **Network in Step 7: none.** No download, installation or request to an external host. The only socket activity is the loopback probe of 11.1.
- **Under `strace -f -e trace=network`,** the key test and `--json` (Appendix S1 method; `HOME`, `XDG_CACHE_HOME` and `TMPDIR` in an empty directory) make:
  - only urllib3's `socket(AF_INET6)` + `bind` to `::1`, and no IP `connect`, `sendto` or `sendmsg`;
  - under `env -i`, two `connect` calls to the local `nscd` Unix socket (absent: `ENOENT`). They come from pytest's user lookup for `tmp_path` and disappear when `USER` is set.
- **Under `bwrap --unshare-net`:** the key test passes (8 of 8), and `--json` exits 0.
- **Determinism and side effects:** `--json` gives byte-identical output under `strace`, under `bwrap` and through `make chunk`. Only pytest's empty temporary directories were written, in `TMPDIR`.
- **Corpus:** same HEAD and an empty `git status` before and after every run.
- **Content:** no corpus content, symbol or heading entered the conversation. `--list` and `--show` were run only on a synthetic repository.

## 13. Phase 3 — Embeddings and indexing (Steps 1–4)

Steps 1–4 turn the Phase 2 nodes into vectors with the local model and persist them in embedded ChromaDB [D: rag_phase3.md Steps 1–4]. The failures they guard against are silent ones: truncated text, a different recipe at index and query time, stale vectors, the wrong metric, an empty index. Step 5 (`load_index`), Step 6 (search) and Step 7 (key test) are not built yet. Code:
- `src/indexing/embeddings.py`: recipe and adapter;
- `src/indexing/vector_store.py`: store and generations;
- `src/indexing/manifest.py`: the active-index pointer and manifest;
- `src/indexing/build_index.py`: the pipeline and its CLI;
- `src/ingestion/digest.py` and `src/ingestion/repo_head.py`: moved from `tests/` (decision P6).

**13.1 Decisions**

| ID | Decision | Source |
|---|---|---|
| P1 | Own `BaseEmbedding` adapter over the hardened `OllamaHttpClient`; `embed` gains `truncate` and the token count (additive). `llama-index-embeddings-ollama` is not installed: its client follows redirects, trusts proxy variables, has no timeout and no `truncate`, strips text, and pulls `pytest-asyncio` at run time | [U] 2026-11-26 |
| P2 | Recipe: prefixes `search_document: ` and `search_query: `, `truncate=False` (typed error), cosine space set at creation; score = exp(cos − 1), so cos = 1 + ln(score) | [U] 2026-11-26 |
| P3 | Keep `nomic-embed-text` v1.5 and accept Spanish; golden queries in Spanish and English (Step 6) | [U] 2026-11-26 |
| P4 | Build a staging generation, activate it atomically, drop the old one; reuse a stored vector only for the same node ID, EMBED-string sha256, recipe fingerprint and model digest | [U] 2026-11-26 |
| P5 | The manifest records everything Step 5 must compare (model, recipe, digests, repository identity); `load_index` and `StaleIndexError` arrive with Step 5 | [U] 2026-11-26 |
| P6 | `document_digest`, `corpus_digest` and `read_head` live in `src/`; the Phase 2 helpers import them, and the expected values stay hardcoded (fixture, golden digest, frozen SHA) | [U] 2026-11-26 |
| P7 | One index = one repository; the manifest records its resolved path and HEAD SHA, "not a git repo" or "HEAD unresolvable" with the reason | [U] 2026-11-26 |
| P8 | Today: tests for Steps 1–4 and the real corpus run; golden queries, the key test and the owner's review belong to Steps 6–7 | [U] 2026-11-26 |
| Q-A | `get_vector_store(persist_dir, collection)` returns a location handle (`IndexStore`) that `build_index` drives; Phases 4–5 call it unchanged | [U] 2026-11-27 |
| Q-D | An unchanged repository is still rebuilt in staging, with every vector reused: one code path, always verified | [U] 2026-11-27 |
| Q-E | The recipe fingerprint covers only what changes vectors (model, prefixes, `truncate`, dimension, context); batch size is operational (batching measured to change nothing) | [U] 2026-11-27 |
| Q-F | The recipe lists the model digests it was verified for; an unknown digest behind the same name fails (`UnverifiedModelError`) | [U] 2026-11-27 |
| Q-B | One Chroma directory per generation: `delete_collection` leaves deleted code in `chroma.sqlite3` (V2), deleting a directory leaves nothing | [U] 2026-11-27 |
| Policies | Approved with four additions:<br>• store root `0700` and an `fsync` of the directory after `os.replace`;<br>• `EmbeddingRecipeError` (exit 10) and `UnverifiedModelError` (exit 13);<br>• index exit codes 30–33;<br>• the store factory also caps the `llama_index` and `chromadb` loggers at WARNING | [U] 2026-11-27 |
| Chunker rule | Phase 2's "no module imports the chunker" now covers `src/ingestion/` only (its intent: no cycles); orchestrators such as `src.indexing` call its public API | [U] 2026-11-27 |
| Chroma reads | Accepted: Chroma has no read-only mode, so opening a generation rewrites the content of its own sqlite and HNSW files (no file is created or deleted). Constructing the handle and reading the manifest write nothing | [U] 2026-11-27; [M] |

**13.2 What an embedding is (Step 1).**
- **Vectors.** An embedding model maps text to a vector, here 768 numbers of unit length. Texts that mean similar things get vectors that point in similar directions.
- **Similarity.** Cosine similarity measures the angle between two vectors. For unit vectors it is their dot product: 1 means the same direction, values near 0 mean unrelated.
- **Geometry as meaning.** Search embeds the question and returns the nearest code. "How do I authenticate?" shares no word with a function that verifies an access token, yet their vectors are close. The demo test proves it with the real model, `tests/test_embeddings_semantic_integration.py` [M]:
  - English: the token check ranks first, cosine 0.704 against 0.556 for the best of five distractors;
  - Spanish (`¿cómo me autentico?`): it also ranks first, 0.484 against 0.430, with a thinner margin (P3).
- **Asymmetric prefixes.** nomic-embed-text v1.5 was trained to embed indexed text as `search_document: <text>` and questions as `search_query: <text>`. Ollama adds neither: its template is `{{ .Prompt }}`. The same text with and without its prefix gives vectors with cosine 0.94, so a prefix forgotten on one side moves everything slightly and silently.
- **The same recipe.** Model, prefixes, truncation, dimension and context must be identical at index and query time. One `EmbeddingRecipe` serves both sides, and its fingerprint goes into the manifest.
- **What breaks silently:**
  - an input over 2048 tokens is cut without an error unless `truncate=False`;
  - a new model behind `:latest` changes the space;
  - Chroma's default metric is l2;
  - `get_or_create_collection` turns a missing index into an empty one;
  - an `add` with an existing ID is ignored, keeping the old vector.

**13.3 The recipe** (`src/indexing/embeddings.py`, [M] 2026-11-26/27)

| Field | Value |
|---|---|
| Model | `nomic-embed-text:latest` (v1.5, 137M, F16), verified digest `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |
| Prefixes | `search_document: ` (indexed text), `search_query: ` (questions); applied exactly once, no `.strip()` or normalization |
| Truncation | `truncate=False`. Ollama 0.34.0 answers HTTP 400 `the input length exceeds the context length`. Without it, any input over 2048 tokens (CLS and SEP included) is cut silently. A batch with one long input fails as a whole, so the adapter re-embeds one by one and reports the position and length, never the text |
| Dimension, context, batch | 768; 2048 tokens; 10 inputs per request. Batching changes no vector (max abs diff 0), and batch size gives no speed-up (458–494 ms per node, CPU-bound) |
| Fingerprint | `546cb922dd10be43bb41dfb6ef3818579b7f0f3e12cb8498609dd87cd453ca81` (sha256 of the canonical JSON of the vector-affecting fields; pinned by a test) |
| Vectors | Unit norm, deterministic, case-insensitive (upper and lower case give identical vectors: identifier case is invisible to the embedder); each one is checked for count, dimension, finiteness and non-zero |
| Tokens on the corpus | EMBED string plus prefix: p50/p90/max 237/403/915 tokens, none above 75% of the context; 3.68 characters per token at the median, 1.73 at the minimum |
| Identity | `identity()` checks with `ollama list` and `show` that the model is local (weights on disk, not cloud), that its digest is a verified one, and that context and dimension match. A missing model raises `ModelNotFoundError` naming `make pull-models`. Nothing is ever pulled |

**13.4 The vector store** (`src/indexing/vector_store.py`)

- **One client factory.** `chromadb.config.Settings` reads `.env` from the working directory and `CHROMA_*` environment variables. With `CHROMA_API_IMPL=…FastAPI` and a server host there, `PersistentClient` became an HTTP client and tried to reach that host (V1, measured inside `bwrap --unshare-net`; `NetworkGuard` blocked it, and Chroma re-raised the block as a `ValueError`). The factory:
  - pins every field that decides the destination or the imported module (`_env_file=None`, the Rust API, persistence, telemetry off, `allow_reset` off, no server host or headers, an empty OpenTelemetry endpoint);
  - checks the *effective* settings and closes and refuses any difference;
  - refuses a directory that does not exist (V5: Chroma would create it).
- **Collections.** Writing creates with `embedding_function=None` and `{"hnsw": {"space": "cosine"}}`; reading only opens with `embedding_function=None`, and the metric is verified on both paths. Never `get_or_create_collection`, `from_params`, a `ChromaVectorStore` without a collection, or `query_texts`: a static audit of `src/` enforces this.
- **Layout** (Q-B):
  - `<VECTOR_STORE_DIR>/` (`0700`, git-ignored; default `data/vector_store` under the repository root, never the working directory);
  - `manifest.json`: the pointer and the manifest (`0600`);
  - `.lock`: `fcntl` lock held while building;
  - `gen-<utc>-<hex>/`: one complete Chroma store per generation (`0700`), with a `.local-code-rag-generation` marker. Only directories with that name pattern and marker are ever deleted; symlinks are never followed.
- **Self-ingestion.** None of these file types is one the loader ingests (`.json`, `.sqlite3`, `.bin`, `.lock`, no extension), and `build_index` fails if any loaded document comes from inside the store.
- **Logs.** `ChromaVectorStore` and `llama_index.core` log the first 100 characters of every retrieved node at DEBUG (V6). `configure_logging` and the factory keep `llama_index` and `chromadb` at WARNING, so a host that sets the root logger to DEBUG cannot leak code. A test with a sentinel in the code checks this, with a negative control.
- **Releasing a generation.** After `close()` two sqlite file descriptors stay open until every reference to the collection is released. `discard` closes the client and collects garbage, then deletes the directory.

**13.5 The manifest** (`manifest.json`, `format` `local-code-rag/index-manifest`, `schema_version` 1). Keys, in order:
- `generation`, `collection`, `created_utc`;
- `repo` (`path`, `state`, `head`, `reason`);
- `documents`, `nodes`, `node_digest` (the Phase 2 digest), `embed_digest` (sha256 over `<node id>:<sha256 of its EMBED string>` lines);
- `embedder` (`model`, `digest`, `recipe_fingerprint`, `dimension`, `verified`), `recipe` (`to_dict()`, or null when unverified);
- `ollama_version`, `metric`, `max_chunk_chars`, `rules_file_sha256`, `libraries` (chromadb, llama-index-core, llama-index-vector-stores-chroma, ollama, tree-sitter, tree-sitter-python).

The reader is strict: unknown format or schema, extra or missing keys, or wrong types raise `VectorStoreError`, and the manifest is never repaired in place. The writer uses a temporary file, `fsync`, `os.replace` and an `fsync` of the directory.

**13.6 The pipeline** (`build_index`; the order and its reasons are in the module docstring):
1. lock;
2. load and chunk;
3. repository identity and digests;
4. verify the embedder;
5. reuse;
6. staging and embedding (progress logged per batch, without content);
7. verify count, IDs, metric, dimension, vectors and every reconstructed node hash;
8. check the model digest again;
9. activate;
10. discard the previous generation.

Any exception before activation, `KeyboardInterrupt` included, discards the staging generation, and the previous index keeps serving. Nothing is ever added to `node.metadata` (it would change the hash). `make index` (`python -m src.indexing.build_index [--repo] [--rules-file] [--json]`) runs it under `NetworkGuard`; exit codes 30–33 are listed in the module docstring.

**13.7 Evidence** [M], 2026-11-27

| Check | Result |
|---|---|
| V1–V7 on this host | All confirmed (spike in a `mktemp -d`, synthetic data, `.venv/bin/python`). V4: an `add` with an existing ID is ignored; NaN and a wrong dimension are rejected; the zero vector is accepted (the adapter rejects it); vectors come back at float32 precision; the maximum write batch is 5461 |
| Ollama `/api/embed` | Status and message as in 13.3; `prompt_eval_count` of a batch is the sum of its inputs' tokens |
| Corpus, first run | 88 documents → 658 nodes, all embedded; node digest `0ddc4d1ae608b2e92a3e6840a2d5fdd6d6fd0b3e9961b4c4d284c82f982444cd` (Phase 2's); 343.9 s (5 min 45 s wall); 14,481,604 B on disk; peak RSS of the indexing process 266 MB; `nomic-embed-text` resident in Ollama 376,449,269 B |
| Corpus, second run | 658 reused, 0 embedded, 0 removed, same node and EMBED digests, one generation left; 1.4 s |
| Corpus integrity | Same HEAD and empty `git status` before and after; `make chunk --json` still 88 → 658 (279/340/39) |
| Tests | `make check` 1,267 passed (1,110 before). Key files: `test_build_index.py` 40, `test_vector_store.py` 36, `test_embeddings.py` 45, `test_config.py` 104. `make test-corpus` 18 (one P6 test added). Integration: the Step 1 demo, 2 tests |
| Privacy | Every test under `NetworkGuard` with 0 violations. A fresh interpreter indexes and queries with a hostile `.env`, `CHROMA_*` and proxy variables: 0 violations, no `onnxruntime`, no writes outside the store. `strace -f -e trace=network` on `make index` of a synthetic repository: connects only to `127.0.0.1:11434` and the absent local `nscd` socket, plus urllib3's `::1` probe; no non-loopback send. `bwrap --unshare-net` with the fake embedder: 126 tests pass. A temporary `HOME` and `XDG_CACHE_HOME` stay empty |

**13.8 Deviations from rag_phase3.md (Steps 1–4)**

| Doc | This project | Why |
|---|---|---|
| `uv add llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb` without versions | `chromadb==1.5.9`, `llama-index-vector-stores-chroma==0.6.0`; no Ollama integration | P1; V1–V6 depend on the versions; newest releases at least 14 days old |
| `OllamaEmbedding(model_name, base_url)` | `OllamaEmbeddingAdapter` over the hardened client | P1 |
| No prefixes, default truncation | One recipe with prefixes, `truncate=False`, fingerprint and verified digests | P2, Q-E, Q-F |
| `get_or_create_collection`, default metric and embedding function | Writes create, reads open; cosine; `embedding_function=None` | V3 |
| `Path("data/vector_store")` | Resolved against the repository root; overridable by `VECTOR_STORE_DIR` | Working-directory independence |
| `vector_store or get_vector_store()` | `is None` checks | A falsy object is never replaced silently |
| `get_vector_store` returns a `ChromaVectorStore` | It returns an `IndexStore` handle | Q-A |
| `VectorStoreIndex(nodes, storage_context=…)` adds into the live collection | Staging generation, verification, atomic activation, manifest | P4, P5, P7, Q-B |
| `EMBEDDING_MODEL`/`OLLAMA_BASE_URL` constants in config | Already defined and validated in Phase 0; only `VECTOR_STORE_DIR` and `COLLECTION_NAME` were added | — |
| `__main__` with a search and `print` | A CLI that only indexes; search is Step 6 | Scope; Ruff `T20` |

**13.9 Definition of Done** [D: rag_phase3.md, Verification]

| # | Item | Status |
|---|---|---|
| 1 | `embeddings.py` configures the local model, used for indexing and querying | ✔ (13.3; `get_embedding_model()` serves both sides with one recipe) |
| 2 | `vector_store.py` configures ChromaDB with disk persistence | ✔ (13.4) |
| 3 | The pipeline repo → ingestion → chunking → embeddings → storage works end to end | ✔ (13.7: the corpus, 658 vectors) |
| 4 | The index is persisted and reloaded without re-indexing | ⏳ at Steps 1–4; ✔ since Step 5 (section 14.7) |
| 5 | Tests verify that search returns relevant code | ⏳ at Steps 1–4; ✔ since Step 7 (section 14.7) |
| 6 | Key test | ✔ since the owner's review (section 14.7) |

**13.10 Risks and limitations**

| Item | Effect |
|---|---|
| Generations and live readers | A `VectorStoreIndex` returned by `build_index` (or held by Phase 6) reads a generation that the next activation deletes. Phase 6 must swap its state after `/index` and never serve while a build in another process activates. Step 5 should open a generation once per process |
| Chroma rewrites on open | Opening a generation changes its internal files (13.1); two processes opening the same generation at once is unsupported by Chroma [H] |
| Uncased embeddings | `getUserID` and `getuserid` embed identically; Phase 4's BM25 has to carry identifier case |
| Spanish queries | Thinner margins (P3); measured on golden queries in Step 6 |
| Embedding time | About 5–6 minutes for the corpus on this CPU, linear in the number of changed nodes |
| Phase 4 | Its `src/settings.py` must use `get_embedding_model()`, not a second `OllamaEmbedding` (not installed) |
| Phase 6 | `/index` takes a client path (R12), and one index holds one repository (P7) |
| Phase 7 | RAGAS's own embeddings client would use another recipe |
| Phase 8 | No `.dockerignore` yet (`COPY . .` would copy the index), Ollama published on all interfaces, `LOCAL_ALLOWED_HOSTS=ollama` needed |
| Unverified embedders | A custom `embed_model` without `identity()` is indexed as "unverified" and never reused |

## 14. Phase 3 — Steps 5–7 and closure

Steps 5–7 reload the index without re-indexing while refusing an incompatible or stale one, search it and measure the search, and test it [D: rag_phase3.md Steps 5–7]. Code:
- `src/indexing/loading.py`: `load_index`, re-exported from `src.indexing.build_index`, where Phase 6 imports it;
- `src/indexing/search.py`: search and its CLI;
- `src/evaluation/retrieval.py`: golden queries, metrics, the `Retriever` protocol and the evaluator;
- tests: `tests/test_loading.py`, `tests/test_search.py`, `tests/test_retrieval_eval.py`, `tests/test_indexing.py`, `tests/test_indexing_corpus_integration.py`, plus the helper `tests/retrieval_oracle.py`.

**Status: Phase 3 is closed** (2026-11-27). All six Definition of Done items are ✔ (14.7). The owner's review answered "Yes, confirmed" (14.12), after the golden queries, the baseline (14.4) and the Step 2 fix (14.11).

**14.1 Decisions**

| ID | Decision | Source |
|---|---|---|
| Q-E | HEAD moved with identical nodes → `StaleIndexError`: P5 to the letter; the cheap re-run also refreshes the manifest | [U] 2026-11-27 |
| Q-F | `load_index_with_report` exposes the loaded generation and `is_current()`; nothing reloads automatically (Phase 6 decides) | [U] 2026-11-27 |
| Q-G | `eval/retrieval_golden.json` is git-ignored; a synthetic example, the queries' sha256 and the metrics are versioned | [U] 2026-11-27 |
| Q-H | The corpus key test builds into a temporary store (never the owner's index) and has its own target | [U] 2026-11-27 |
| Golden format | `{"version": 1, "queries": [{"id", "intent", "lang", "kind", "query", "expected_files"}]}`; it replaces the Steps 1–4 proposal, which was never used | [U] 2026-11-27 |
| Corpus count | `make test-corpus` counts 18 (the P6 check added in Steps 1–4), not 17 | [U] 2026-11-27 |
| Key-test cost | `make test-integration` ignores `tests/test_indexing_corpus_integration.py`; the key test runs via `make test-index-corpus` | [U] 2026-11-27 |
| Policies | Approved with three additions:<br>• exit 2 for search usage errors; 34 incompatible, 35 stale;<br>• `load_index` lives in `loading.py` and is re-exported; the evaluator lives in `src/evaluation/retrieval.py`;<br>• evaluation depth 20, with code vs docs decided by the expected files' language | [U] 2026-11-27 |
| Golden authorship | The owner asked the agent to write the queries; after the warning (corpus content would leave the machine, and queries by the implementer are not blind) the owner kept P8 |
| Placeholders | The golden loader refuses a query or intent whose whole text is a template marker | [U] 2026-11-27 |
| Exhaustive search | Fixes a Step 2 defect (14.11):<br>• a build sets each generation's `ef_search` to max(1024, vectors);<br>• loading refuses less (exit 34);<br>• the owner's index was rebuilt (658 reused, 0 embedded);<br>• the void baseline was overwritten | [U] 2026-11-27 | [U] 2026-11-27 |

**14.2 Loading (Step 5).** `load_index(nodes=None, *, vector_store=None, embed_model=None, repo_path=None) -> VectorStoreIndex` and `load_index_with_report(...) -> LoadResult`. The report carries `index`, `manifest`, `generation`, `drift_checked`, `attempts`, `duration_ms`, `is_current()` and `close()`. In order:
1. **Identity first.** The embedder must prove its identity: the real adapter asks Ollama, so loading needs Ollama on loopback, and the check is never skipped. An embedder without `identity()` is refused.
2. **Open.** `open_active` never creates. A missing index raises `IndexNotFoundError`; a corrupt manifest, an unknown schema or a missing generation raises `VectorStoreError`.
3. **Re-read the pointer.** The manifest is read again after opening, and the load retries a bounded number of times if it moved (3 by default).
4. **Compatibility, always** (`compatibility_problems`, pure). Model, model digest, recipe fingerprint, dimension, metric (cosine), stored count against the manifest, and both identities verified. Any difference raises `IncompatibleIndexError` (exit 34), listing them all and naming `make index`.
5. **Drift, only with `nodes`** (`drift_problems`, pure). The Phase 2 digest of the given nodes, and the HEAD of `repo_path` (or of the manifest's path) when the index recorded a SHA: a moved or unreadable HEAD is drift. A repository without git is checked by digest only. Any difference raises `StaleIndexError` (exit 35); the message notes that other ingestion or chunking rules also change the digest. `nodes=[]` is checked; `nodes=None` logs `index.drift_unchecked`.
6. **Wrap** with the explicit embedder: zero document embeddings, and no file created or deleted (section 13.1 covers Chroma's rewrite of its own files on open).

Loading never takes the build lock, and the `index.loaded` log carries the generation, counts, checks and duration, never content.

- **A replaced generation is served silently.** Measured with synthetic data: when another process activates a new generation and deletes the old directory, an already-loaded index keeps returning the old results, with no error and nothing empty, because its open file handles keep the deleted files alive. `is_current()` detects it with one manifest read; loading again serves the new generation (Q-F).
- **The reader/writer race.** Forced interleavings prove the retry: a generation deleted between reading and opening, and one replaced right after opening, both load the new generation at attempt 2; an index that keeps changing fails after the bounded attempts. A narrower window remains between the factory's existence check and `PersistentClient`: Chroma creates a missing directory with an empty `chroma.sqlite3` (measured), which would leave an empty, unmarked `gen-…` directory that no cleanup ever deletes. It holds no code; the load itself retries.
- **One open per process.** Chroma keeps one `System` per persist directory and process, shared by reference count, and it rewrites its own files on open. Two consecutive loads in one process share it and close cleanly (tested); separate processes opening one generation at once are not supported by Chroma [H].

**14.3 Search (Step 6).** `search(index, query, *, top_k=5, language=None) -> tuple[SearchHit, ...]` is pure over a loaded index. Each `SearchHit` carries `rank`, `file_path`, `symbol`, `start_line`, `end_line`, `language`, `score`, `cosine`, `node_id` and `text` (the Phase 6 `Source` fields, plus rank and cosine).
- **Score.** `cosine = 1 + ln(score)`; it is `None` when the score is missing or non-positive.
- **Filter.** `language` uses `MetadataFilters` with `FilterOperator.EQ`, as Phase 4 does. An unknown label is a usage error; a known label without chunks returns no hits.
- **Recipe.** The query carries `search_query: ` exactly once, and no returned text carries a prefix: tested through the real adapter over `httpx.MockTransport`.
- **Input.** An empty query is refused, since it would embed the prefix alone. An over-long query raises `EmbeddingInputTooLongError` and is never truncated. The query text never reaches logs or exceptions; only its length is logged.
- **CLI.** `python -m src.indexing.search QUERY|- [--top-k] [--language] [--repo] [--rules-file] [--json | --show]`, through `make search SEARCH_FLAGS=…` (the recipe is not echoed). It chunks the repository without embedding and loads with those nodes, so a stale index is refused. It prints rank, path, lines, symbol, cosine and score. Code only with `--show`, with control characters and ANSI sequences escaped. Exit codes are those of `build_index`, plus 2 for usage errors.

**14.4 Measuring it (Step 6)**
- **Golden queries.** `eval/retrieval_golden.json`, git-ignored, written by the owner before seeing any result. `lang` is `es` or `en`, ES/EN pairs share an `intent`, and `kind` is `concept` or `identifier`. The loader is strict: unknown or missing keys, repeated IDs, empty or repeated `expected_files`, and files the index does not hold are errors that name the ID and the field, never the text. So is a query or intent whose whole text is a template marker: `REPLACE_WITH_…`, `<…>`, or text starting with TODO, FIXME, TBD or XXX. This rule was added after Gate 2 (see the baseline status). A synthetic example is `eval/retrieval_golden.example.json`.
- **Metrics** (pure; hand-computed tests). A query's rank is the position of the first chunk from an expected file. hit@1/3/5/10/20 and MRR@20 are broken down by language, `kind` and code vs docs. No threshold: with 8–15 queries they are indicative.
- **Evaluator.** `evaluate(queries, retriever, file_languages)` talks to a `Retriever` protocol (`name`, `rank(query, depth)`), so Phase 4 can plug its hybrid retriever and reranker into the same metrics; `VectorRetriever` is Phase 3's. `make eval-retrieval` (`python -m src.evaluation.retrieval`) loads the verified index with the repository's nodes.
- **Baseline.** `--write-baseline eval/retrieval_baseline.json` (versioned) records metrics, breakdown, rank per ID, the queries' sha256 and count, the index identity (generation, digests, model and digest, recipe fingerprint), the corpus HEAD, depth, date and versions. It holds no query text and is never overwritten without `--overwrite-baseline`.
- **Owner's review dump.** `--review-dir` dumps the queries and the retrieved code for the owner's review (`0700` directory, `0600` file); it is refused inside this repository or the indexed one.
- **Baseline status:** ✔ recorded [M], 2026-11-27T18:46:27Z, after two discarded runs.
  - **Discarded run 1** (18:05Z): 4 of the 14 queries (`q06_id`–`q09_id`, all `identifier`) still held template placeholders. Their ranks measure nothing, and neither does any aggregate that includes them. Validation missed it because the loader accepted any non-empty text; it now refuses whole-text template markers (owner's OK, 2026-11-27).
  - **Discarded run 2** (between 18:17Z and 18:46Z): after the owner completed those 4 queries, the ranks changed from one process to the next (`q09_id` 1st or absent, `q04_es` 18th or 19th). The cause was a Step 2 defect: the HNSW search was approximate and different in every process (14.11). The run is void.
  - **Golden file:** 14 queries; sha256 `09803093092e299932babaa725739d068f52b5cdcb08388f5ffd7fe2f193be67`.
    - 9 are EN and 5 ES; 10 are `concept` and 4 `identifier`; 10 target code and 4 docs.
    - They cover 9 intents, 5 of them as ES/EN pairs, with 13 distinct expected files.
    - Only the 4 `identifier` queries changed: the file shrank by 39 bytes, exactly the difference between their old and new lengths.
    - Each identifier is a symbol of a chunk of its expected file.
  - **Baseline file:** `eval/retrieval_baseline.json`, sha256 `218fc62de1fd1a14686eee806423a610b4e571f1c3ee6225288e269519d78a89`, over generation `gen-20261127T184611601007Z-1fa485b5`: 658 nodes, digest `0ddc4d1a…`, corpus `5c288080…`, `ef_search` 1024.

  | Group | n | hit@1 | hit@5 | MRR@20 |
  |---|---|---|---|---|
  | All | 14 | 0.7857 | 0.8571 | 0.8341 |
  | EN | 9 | 0.8889 | 1.0 | 0.9444 |
  | ES | 5 | 0.6 | 0.6 | 0.6355 |
  | EN, the 5 paired queries only | 5 | 0.8 | 1.0 | 0.9 |
  | concept | 10 | 0.7 | 0.8 | 0.7678 |
  | identifier | 4 | 1.0 | 1.0 | 1.0 |
  | code | 10 | 0.8 | 0.9 | 0.8553 |
  | docs | 4 | 0.75 | 0.75 | 0.7812 |

  - **Ranks and coverage:** every rank is 1 except `q03_es` 8, `q04_en` 2 and `q04_es` 19. hit@3 is 0.8571, hit@10 0.9286 and hit@20 1.0. The ranks equal those of an exact cosine kNN.
  - **Reproducible:** three runs gave byte-identical output.
  - **Spanish gap (P3):** on the 5 pairs, MRR@20 is 0.6355 in ES vs 0.9000 in EN. Two ES variants rank 8 and 19, where their EN pairs rank 1 and 2. With n = 5 this is input for Phase 7, not a decision.
  - **Privacy of the run:**
    - `strace` saw only `127.0.0.1:11434`, plus two failed `connect` calls to the absent nscd socket;
    - `NetworkGuard` recorded 0 violations, and the logs carry no content;
    - the baseline holds none of the 28 query and intent strings.

**14.5 Definition of Done → tests → commands**

| # | DoD item | Tests | Command |
|---|---|---|---|
| 1 | One local embedding model for indexing and querying | `test_embeddings.py`; the query-prefix test in `test_search.py` | `make test` |
| 2 | ChromaDB with disk persistence | `test_vector_store.py` | `make test` |
| 3 | The pipeline works end to end | `test_build_index.py`; the corpus key test | `make test`, `make test-index-corpus` |
| 4 | Persisted and reloaded without re-indexing | `test_loading.py` (new process, 0 document embeddings); the corpus key test (new process, 0 embedded) | `make test`, `make test-index-corpus` |
| 5 | Tests verify that search returns relevant code | `test_indexing.py` (hermetic, plus a broken embedder that makes it fail); the real-model test; HNSW against exact kNN | `make test`, `make test-integration` |
| 6 | Key test: index a repository and search returns relevant snippets | The corpus key test; the golden-query baseline and the owner's review | `make test-index-corpus`, `make eval-retrieval`, owner's review |

**14.6 Evidence** [M], 2026-11-27

| Check | Result |
|---|---|
| Real model (Step 7) | "how are input data validated?" ranks the input check first among four distractors, although the check never uses "validate" in its path, symbol or body. The Spanish variant also ranks first |
| Corpus key test | 5 passed in 379 s:<br>• 88 documents → 658 vectors, node digest `0ddc4d1a…`;<br>• second build: 658 reused, 0 embedded, identical IDs;<br>• a new process loads with 0 document embeddings;<br>• HNSW top-20 equals exact kNN for 5 generic queries;<br>• corpus unchanged, 0 guard violations;<br>• the owner's index untouched.<br>That oracle check passed by luck: before the fix it could fail (14.11). After the fix, 5 passed in 308 s |
| Real index | `load_index(nodes=…)` on `data/vector_store`: verified with drift, 1 attempt, current, 486 ms |
| Suite | `make check` 1,355 passed (1,269 before); `make test-corpus` 18; `make test-integration` 23 in 40 s; `make verify-local` OK; `uv lock --check` OK |
| Privacy | A fresh interpreter builds, loads and searches with a hostile `.env`, `CHROMA_*`, proxies and the root logger at DEBUG: 0 violations, neither query nor code in the logs, no `onnxruntime`, `HOME` untouched. `strace -f -e trace=network` on a real `search` and `eval-retrieval` over a synthetic index: one connect to `127.0.0.1:11434`, every `sendto` on that socket, the local `nscd` socket and urllib3's `::1` probe; the query sentinel is absent from their output and logs. `bwrap --unshare-net`: the 88 tests of the new files pass |

**14.7 Definition of Done** [D: rag_phase3.md, Verification]

| # | Item | Status |
|---|---|---|
| 1 | `embeddings.py` configures the local model, used for indexing and querying | ✔ (13.3; one recipe, query prefix tested in 14.3) |
| 2 | `vector_store.py` configures ChromaDB with disk persistence | ✔ (13.4) |
| 3 | The complete pipeline works end to end | ✔ (13.7; the key test, 14.6) |
| 4 | The index is persisted and can be reloaded without re-indexing | ✔ (14.2; 0 document embeddings in a new process, on synthetic data and the corpus) |
| 5 | The tests verify that the search returns relevant code | ✔ (14.5; real model, first place) |
| 6 | Key test: a similarity search returns relevant snippets | ✔ The corpus key test passes, the baseline is recorded (14.4), and the owner's review answered "Yes, confirmed" on 2026-11-27 (14.12) |

**14.8 Deviations from rag_phase3.md (Steps 5–7)**

| Doc | This project | Why |
|---|---|---|
| `load_index()` = `get_vector_store()` + `from_vector_store` | Open the active generation, verify compatibility and drift, then `from_vector_store` with an explicit embedder | P5 |
| "Persistence is automatic" | Manifest verified on every load | P5 |
| Search in the `__main__` of `build_index.py` | Separate search CLI over `load_index` | Searching never re-indexes |
| `print` of content and raw scores | No `print`; content only with `--show`, sanitized; cosine = 1 + ln(score) | Privacy; P2 |
| One query, eyeballed | Owner-written golden queries, metrics, owner review | P8 |
| `except Exception` + `skipif` calling Ollama at collection time | `integration` marker, no `skip` | A skip is not evidence |
| The hit is `validation.py` | Neutral names, distractors, no "validate" in the target | Path and symbol are embedded |
| "Run them with `make test`" | Item 5 needs `make test-integration`; the key test has `make test-index-corpus` | `make test` excludes integration |

**14.9 Contract audit with Phases 4–8** (read-only; ✔ confirmed, ✖ refuted, [H] not verifiable without those phases)

| Phase | Point | Verdict and action |
|---|---|---|
| 4 | `src/settings.py` sets `Settings.embed_model = OllamaEmbedding(…)` | ✔ Clashes with P1 (the package is not installed): reuse `get_embedding_model()`, and never rely on `Settings` |
| 4 | BM25 over the nodes | ✔ Build it from the same `nodes` passed to `load_index(nodes=…)`, so both see one digest |
| 4 | Its `__main__` calls `build_index(REPO_PATH)` on every run | ✔ Cheap with reuse (1.4 s), but it activates a new generation each time; prefer `load_index` |
| 4 | `SentenceTransformerRerank("BAAI/bge-reranker-base")` | ✔ in the doc; [H] it downloads the model on first use: needs explicit provisioning and an offline check |
| 5 | The context uses `node.get_content()` | ✔ Raw text, no prefix (tested: stored text carries no prefix). No `as_query_engine` in Phase 5 |
| 6 | Start-up `state["index"] = load_index()` | ✔ Pass the start-up nodes: `load_index(nodes=state["nodes"], repo_path=REPO_PATH)` |
| 6 | `/index` assigns `state["nodes"]` before `build_index` | ✔ A failed build leaves nodes and index mismatched: assign both after success; handle `IndexLockedError` (exit 30); allow-list paths (R12); then load the new generation (Q-F) |
| 6 | `API_HOST = "0.0.0.0"` | ✔ Use `127.0.0.1`; Streamlit usage statistics [H]: disable them explicitly |
| 7 | RAGAS through LangChain's `ChatOllama` and `OllamaEmbeddings` | ✔ A third embeddings client without the recipe's prefixes; record corpus HEAD, node digest and baseline sha256 per run (R13); RAGAS telemetry [H] |
| 8 | `COPY . .` without a `.dockerignore` | ✔ The image would contain `data/vector_store` (the code): add a `.dockerignore` |
| 8 | `OLLAMA_BASE_URL=http://ollama:11434` | ✖ Not a clash: the configuration accepts it with `LOCAL_ALLOWED_HOSTS=ollama` (Phase 0) |
| 8 | Ports `11434`, `8000`, `8501` on all interfaces | ✔ Bind them to `127.0.0.1` |
| 8 | A `0700` volume against `appuser` | [H] The host directory belongs to the host user; a different UID inside the container cannot read it |
| 8 | The repository path changes inside the container | ✔ Drift compares digest and HEAD, not the path; pass `repo_path` explicitly |

**14.10 Limitations and risks**

| Item | Effect |
|---|---|
| A replaced generation is served silently | Callers must check `is_current()` (Phase 6) |
| Empty unmarked directory in a narrow race window | Harmless (no code); never deleted by the cleanup |
| Uncased embeddings | Identifier case is invisible to the vector search; Phase 4's BM25 has to carry it |
| Spanish margins | Thinner in Step 1 (13.2); first place in Step 7's real-model test; on the golden pairs, MRR@20 is 0.6355 in ES vs 0.9000 in EN, with n = 5 (14.4). This is input for Phase 7 (P3) |
| Small n | 8–15 golden queries are indicative; no threshold |
| File-level relevance | A hit is the first chunk from an expected file, even a heading-only chunk without the answer. Point 1 of the owner's review checks the content; chunk-level labels would be a Phase 4 choice |
| Loading needs Ollama | By design (the digest check is never skipped) |
| Exhaustive search cost | Every query visits every vector. That is trivial at 658; it grows linearly with the index, so revisit it if a repository reaches hundreds of thousands of chunks |

**14.11 Step 2 defect: approximate, per-process HNSW search** [M], 2026-11-27, fixed with the owner's OK (stop 11)
- **Symptom.** With identical inputs, the golden ranks changed between processes:
  - in 2 of 5 CLI runs, `q09_id` was absent from the top 20 although it ranks 1st exactly;
  - `q04_es` came 18th or 19th;
  - in 3 of 5 processes, HNSW top-20 ≠ exact top-20 for some query.
- **Ruled out.** Embeddings are deterministic and independent of the previous request: on a synthetic text, every repeat gave an identical vector, in 3 processes and after 5 different prior requests.
- **Cause.** The generation created with Chroma's defaults (`ef_search` 100, `num_threads` = CPUs, `sync_threshold` 1000) never wrote its graph to disk: `link_lists.bin` 0 B, `data_level0.bin` sized for 100 elements, not 658. Every process rebuilds the graph from the log, so each gets a different graph and different approximate misses.
- **Minimal synthetic case.** 658 clustered random vectors, 30 queries, 4 processes:

  | Configuration | Queries that differ from exact | Across processes |
  |---|---|---|
  | Chroma's defaults | 1–5 | a different result in every process |
  | `num_threads` 1 | 3 | still different |
  | `ef_search` ≥ 200, set at creation or later with `modify` (Chroma keeps it) | 0 | identical |

- **Fix.**
  - `ChromaGeneration.make_search_exhaustive()` sets `ef_search` = max(`SEARCH_EF_FLOOR` 1024, vectors). The build calls it before verification, which then requires `ef_search` ≥ vectors.
  - `compatibility_problems` refuses less (exit 34, naming `make index`).
  - `modify` builds no embedding function: read in the source, then measured (none built, no `onnxruntime`, 0 violations). Its "legacy embedding function config" `DeprecationWarning` is silenced only around that call.
- **Tests.**
  - The synthetic case through the store: exact and identical in 2 new processes. Without the fix, the same scenario differs from exact in each of 4 processes (1–3 of 20 queries).
  - The build sets the breadth.
  - An approximate generation is refused with exit 34.
  - The corpus key test's oracle, which had passed by luck, is now deterministic.
- **Effect on the owner's index.**
  - The old generation `…df25baa7` was refused with exit 34.
  - `make index` built `gen-20261127T184611601007Z-1fa485b5`: 658 reused, 0 embedded, the same node digest and embed digest, 2.6 s, loopback only.

**14.12 Owner's review (Gate 3)** [U], 2026-11-27. Done in a clean terminal, outside the assistant, and the temporary directory was deleted. Verdict: approved as Phase 3's baseline, then "Yes, confirmed". Recorded here by ID, rank and category only: no query terms, headings, symbols or content.

| Point | Finding |
|---|---|
| 1. First chunk | `identifier` 4/4 at rank 1 with the exact symbol; 3 return the whole definition. In `q06_id` the body continues in the next chunk (a forced cut). Useful answer at rank 1: `q02_en`, `q02_es`, `q05_en`, `q05_es` (in `q05` the values live in the second expected file). File hit but the chunk does not answer (risk 14.10): `q01_es` (the module's header chunk), `q01_en` (another part of the same file), `q03_en` (another section of the same document) |
| 2. Path, lines, symbol | Correct for all 14; `-` for module-level and YAML blocks, as the chunker emits |
| 3. Above the expected file | `q04_en` (2): a reasonable distractor, a design document that edges out the code. `q03_es` (8): a failure, code that only shares generic words. `q04_es` (19): a reasonable distractor at rank 1, but the code sinks to 19, a severe ranking failure |
| 4. ES vs EN | ES cosines are about 0.13–0.22 lower in every pair. ES holds when the query shares literal technical vocabulary and degrades with translated domain terms |
| 5. Free questions (3, the owner's) | Identifier: excellent (the definition at rank 1, its uses at 2–5). EN concept: the right document in the top 5, but a less relevant section outranks the best one and prose pushes code out of the top 5. ES concept: the right document in the top 5, a heading-only chunk at rank 1, the useful chunks at 2–4 |

**Inputs for Phase 4** (observations, not decisions):
- **Chunk-level labels.** File-level hit@k overstates relevance: 3 of the 10 `concept` hits are file hits whose chunk does not answer. Chunk- or section-level labels, or a reviewed top-k, would measure it.
- **Reranker.** Its targets are the order of sections inside the right document, heading-only chunks at rank 1, and prose outranking code.
- **BM25.** It will not close the ES gap: translated terms share no tokens with the code. Query translation or the Phase 7 model decision is the lever (P3).
- **Chunk shape.** Heading-only chunks and forced cuts are Phase 2 output (closed; node IDs are fixed). Phase 4 can mitigate them at retrieval time, for example by returning the neighbouring chunk, without changing nodes.

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
| P4 | `make test-corpus` (`pytest -m integration -rs tests/test_ingestion_corpus_integration.py`; since Phase 2 Step 7 also `tests/test_chunking_corpus_integration.py`, section 12.3) | The key test (9.6). `make test-integration` also runs it, together with the two Phase 0 tests that need Ollama |
| S1 | From a new `mktemp -d` folder, with `HOME`, `XDG_CACHE_HOME` and `TMPDIR` set to an empty directory: a script that imports `src.ingestion.segmenter`, loads the grammar and segments a synthetic text, run with `strace -f -qq -e trace=network -o <log> .venv/bin/python -I -B <script> <repo>` and with `bwrap --ro-bind / / --dev /dev --proc /proc --bind <dir> <dir> --unshare-net .venv/bin/python -I -B <script> <repo>` | Section 10.4: no network system calls, works with no network, writes nothing |
| S2 | From a new `mktemp -d` folder: a script that loads the corpus with the loader, segments the `python` documents and runs `tests/chunking_invariants.py`, under `NetworkGuard`, via `PYTHONPATH=<repo>:<repo>/tests uv run --locked --offline python <script> <corpus>` | Section 10.5 |
| S3 | As S1, with a script that runs `chunk_documents_with_report` on synthetic documents of every strategy, reads every `MetadataMode` view and round-trips each node through `node_to_metadata_dict` | Section 11.7: works with no network namespace; the only network calls are urllib3's loopback IPv6 probe |
| S4 | `make chunk CHUNK_FLAGS="--repo ~/repos/project-to-index --json"`, then a `mktemp -d` script that runs every strategy's checkers on the 88 documents, under `NetworkGuard`, via `PYTHONPATH=<repo>:<repo>/tests uv run --locked --offline python <script> <corpus>` | Section 11.8 |
| S5 | Only after an owner decision that changes the chunks. First check that `git -C <corpus> rev-parse HEAD` is the frozen SHA and `GIT_OPTIONAL_LOCKS=0 git -C <corpus> status --porcelain` is empty. Then run `uv run --locked --offline python -m tests.chunk_manifest --out tests/fixtures/chunk_manifest.json` from this repository's root (it refuses a corpus that fails those checks and writes atomically). Validate independently with a `mktemp -d` script, under `NetworkGuard`, via `PYTHONPATH=<repo>:<repo>/tests uv run --locked --offline python <script> <corpus>`: the manifest equals a fresh run; paths and languages equal `tests/fixtures/corpus_manifest.tsv`; row sums equal the report; `validate` gives 0 findings; only the definitions over the cap are split. Review the diff of the JSON (counts and digests only) before committing | The manifest of section 12.3 |
| S6 | The owner, in a separate terminal (never through an agent or an IDE connected to one): `umask 077`; `D=$(mktemp -d)`; `make -s chunk CHUNK_FLAGS="--repo <corpus> --list" > "$D/list.tsv"`; for each sampled path `make -s chunk CHUNK_FLAGS="--repo <corpus> --show <path>" > "$D/<path with / as _>.txt"`; read the files next to the sources; `rm -rf -- "$D"`. Report findings as path + lines + symptom, never content. The content-free alternative is a `mktemp -d` script that checks every node's first and last line against `ast`, headings, blank lines and column 0 and prints counts only | Section 12.5 |
| E1 | `make index` (twice) on the frozen corpus; on a synthetic repository with `VECTOR_STORE_DIR` set to a temporary directory: `strace -f -qq -e trace=network -o <log> make -s index INDEX_FLAGS="--repo <repo> --rules-file <empty file> --json"`, and the same CLI via `.venv/bin/python -m src.indexing.build_index` with `HOME`, `XDG_CACHE_HOME` and `TMPDIR` in an empty directory | Section 13.7: 658 vectors with the Phase 2 digest, a second run that embeds nothing; only loopback to Ollama |
| E2 | From a new `mktemp -d` folder, synthetic data only: scripts run with `.venv/bin/python -B` and an empty `HOME`, under `strace -f -e trace=network`; the hostile-`.env` case (V1) inside `bwrap --ro-bind / / --dev /dev --proc /proc --bind <dir> <dir> --unshare-net` | Section 13.7: V1–V7 |
| E3 | `make test-index-corpus` (pytest `-m integration -rs tests/test_indexing_corpus_integration.py`): the corpus built twice into a temporary store, reloaded in a new process, HNSW against exact kNN | Section 14.6: the key test |
| E4 | On a synthetic repository indexed into a temporary `VECTOR_STORE_DIR`: `strace -f -qq -e trace=network -o <log> .venv/bin/python -B -m src.indexing.search "<question>" --repo <repo> --rules-file <empty file>` and the same for `-m src.evaluation.retrieval --golden <synthetic golden file>`, with `HOME`, `XDG_CACHE_HOME` and `TMPDIR` in an empty directory | Section 14.6: only loopback to Ollama |

C3 is a stdlib-only script, not versioned in this repository, and imports no networking modules. It does not follow symlinks and handles paths as bytes. Per file it records:
- a binary flag (NUL byte in the first KiB), lines as `count("\n") + 1`, and UTF-8 validity;
- Phase 1's map and `IGNORED_DIRS`, applied in Phase 1's filter order.

It prints counts and relative paths only. `unshare -rn` was attempted, and blocked by the host's user-namespace policy.
