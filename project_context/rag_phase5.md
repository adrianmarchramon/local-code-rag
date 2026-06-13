# Phase 5 — Generation with the Local LLM

> This phase finally closes the complete cycle from question to answer. You have an excellent retrieval system that, for any question, brings back the correct code snippets; now you pass those snippets to the local LLM along with the question so that it generates an answer grounded in them. And here we see the reward of all the care put into retrieval: because you provide the model with the correct context, its answer will be accurate. It is worth keeping the core idea in mind: in RAG, generation is the "small" part. You have already done the hard work (retrieving well); here, the quality lever lies primarily in the **prompt**: in how you ask the model to ground itself in the context and cite its sources. Those two things, grounding and citations, are what turn an unreliable generator into a trusted assistant.

**Phase objective:** Generate answers grounded in the retrieved code.  
**Duration:** Week 5.  
**Upon completion you will have:** The complete RAG pipeline working end-to-end (question → retrieval → generation → answer with citations), with a local LLM that answers by grounding itself in the actual repository code and citing the files it is based on.

---

## The Big Picture

Generation is the final stage of the query path, and by adding it you complete the entire RAG pipeline:

```
   User Question
        │
        ▼  (Phase 4)
   Hybrid retrieval + reranking  ──►  the 5 correct snippets
        │
        ▼  (Phase 5, new)
   Build the prompt  ──►  instructions + snippets (context) + question
        │
        ▼  (Phase 5, new)
   Local LLM (Qwen3-Coder via Ollama)  ──►  generate the answer
        │
        ▼
   Grounded answer + citations (file, symbol, lines)
```

We will use LlamaIndex's Ollama client to communicate with the model (you already installed `llama-index-llms-ollama` in Phase 4). Add the generation configuration to `src/config.py`:

```python
REQUEST_TIMEOUT = 180.0       # local models take time; the 30s default is very low
LLM_CONTEXT_WINDOW = 32768    # wide window: we are going to inject code into the prompt
```

---

## Step 1 — The Local LLM Client

First up is the client that communicates with Qwen3-Coder via Ollama. Complete `src/generation/llm.py`:

```python
from llama_index.llms.ollama import Ollama

from src.config import CODE_MODEL, LLM_CONTEXT_WINDOW, OLLAMA_BASE_URL, REQUEST_TIMEOUT


def get_llm() -> Ollama:
    """Local code LLM client (Qwen3-Coder via Ollama)."""
    return Ollama(
        model=CODE_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=REQUEST_TIMEOUT,
        context_window=LLM_CONTEXT_WINDOW,
    )
```

Two parameters deserve attention, as they correspond to two real problems that catch many people off guard. The first is the `request_timeout`. The Ollama client has a default timeout of only 30 seconds, and local models—especially on hardware without a powerful GPU—can take significantly longer to generate a response. If you don't increase it, you will see timeout failures that look like mysterious errors but simply mean "the model did not finish in time." Increasing it to 180 seconds (or more, on slow hardware) resolves this.

The second is the `context_window`, and this is where the detail we anticipated in Phase 0 materializes. We are going to inject multiple code snippets into the prompt, and that can consume many tokens. If the context window is small (remember that Ollama uses 4096 by default), the context gets truncated, the model cannot see part of the code you provided, and the quality of the response degrades without you understanding why. That is why we set a wide window (32,768), which is sufficient to fit all the retrieved code. This is exactly why we warned you about this limit in Phase 0: now it shows its face, and because you anticipated it, you solve it with a single line. (Alternatively, you can use the custom model with the built-in window that you created in Phase 0.)

---

## Step 2 — The RAG Prompt: The Piece That Defines Quality

Here is the real work of this phase. Since generation relies on good retrieval, what makes the difference in the quality of the answers is the **prompt**: the instruction you give the model, combining the retrieved context with the question. A well-designed prompt ensures the model grounds its answers in the code and cites its sources; a careless one allows the model to ramble or make things up. Complete `src/generation/prompt.py`:

```python
from llama_index.core.schema import NodeWithScore

SYSTEM_INSTRUCTIONS = """You are an expert assistant that answers questions about a specific codebase.

Strict rules:
- Answer ONLY based on the code snippets provided below.
- If the snippets do not contain the necessary information to answer, state it clearly
  ("I cannot answer with the available code") instead of making things up.
- When you use information from a snippet, cite its source file.
- If you generate new code, follow the patterns, style, and conventions you see in
  the project snippets.
"""


def format_context(nodes: list[NodeWithScore]) -> str:
    """Formats the retrieved snippets as context, including their source."""
    blocks = []
    for i, node in enumerate(nodes, start=1):
        m = node.metadata
        header = (
            f"[Snippet {i}] "
            f"File: {m.get('file_path', '?')} | "
            f"Symbol: {m.get('symbol', '-')} | "
            f"Lines: {m.get('start_line', '?')}-{m.get('end_line', '?')}"
        )
        blocks.append(f"{header}\n```\n{node.get_content()}\n```")
    return "\n\n".join(blocks)


def build_prompt(query: str, nodes: list[NodeWithScore]) -> str:
    """Builds the RAG prompt: instructions + context + question."""
    context = format_context(nodes)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"=== PROJECT CODE SNIPPETS ===\n"
        f"{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER (grounded in the snippets, citing the files) ===\n"
    )
```

The prompt has three distinct parts, and each plays a role. The **instructions** establish the model's behavior (the grounding and citation rules, which we will look at in detail). The **context** consists of the retrieved snippets, formatted with their source (file, symbol, lines) and clearly delimited so that the model knows what is project code and where each piece comes from. And the **question** is the user's query. Notice that the context includes each snippet's metadata: this is not just informative, but also gives the model the information it needs to cite correctly.

---

## Step 3 — Grounding and Avoiding Hallucinations

This is one of the two quality levers of this phase, and it is important to understand it well. The main flaw of an LLM on its own is that it **hallucinates**: when it doesn't know something, it doesn't stay silent; instead, it invents a plausible-sounding answer. In a code assistant, this is dangerous: it would tell you with absolute certainty that a function exists when it doesn't, or misdescribe how something works.

**Grounding** is the superpower of RAG compared to a bare LLM, and it is precisely what combats hallucination. Because you provide the model with *real* code snippets from the project and instruct it to answer only based on them, its response is backed by evidence, not by what it remembers from its training. The two key rules of the prompt are what achieve this: "answer only based on the provided snippets" and, crucially, "if the snippets do not contain the information, state it clearly instead of making things up."

That second rule is more important than it seems. An assistant that admits "I cannot answer with the available code" is infinitely more reliable than one that invents an answer to avoid looking bad. Preferring honesty over invention is what turns an unpredictable generator into a trusted tool, and it is a sign that you understand the risks of LLMs and know how to mitigate them. When you test the system, check exactly this: ask it a question about something that is not in the repository and verify that it acknowledges it cannot answer, instead of making something up.

---

## Step 4 — Citations: Transparency and Verifiability

The second quality lever is **citations**, and here the metadata you have been carrying over since Phase 1 yields its ultimate benefit. An answer without sources is a black box: the user has to believe whatever the model says, without being able to verify it. An answer with citations is transparent and verifiable: the user can go to the cited file and function and verify for themselves that the answer is correct. This verifiability is the hallmark of a well-implemented RAG.

Citations work on two levels. On one hand, the prompt instructs the model to cite the source files in its prose response. On the other hand, and more robustly, we return the **structured sources** separately, using the metadata from the retrieved snippets, to display them alongside the response. This way, the user always sees exactly which code snippets back up what the assistant has said, regardless of whether the model mentioned them correctly in its text. We will perform this source extraction in the complete pipeline of the next step, which is where the `file_path`, `symbol`, and line range we took so much care of turn into precise citations: "according to `validate_token` in `src/auth.py`, lines 40-58."

---

## Step 5 — Project-Aware Code Generation

One of the most impressive capabilities of the assistant is generating new code that **fits into the project**. When you ask it to "write a function for X following the project style," the prompt rule "follow the patterns, style, and conventions you see in the snippets" is what makes this possible. Since the retrieved snippets show how *this* specific project is written (its naming conventions, error handling, patterns), the model can mimic that style instead of generating generic code.

This is exactly what differentiates a project-aware assistant from a generic autocompletion tool: the code it generates is not just "syntactically correct Python code," but code that looks like it was written by someone who knows this repository. It is a capability that demonstrates the value of the RAG approach very well, and it shines especially during a demonstration. And once again, it relies on retrieval: to mimic the project style, you must first retrieve snippets representative of that style, which your system already does well.

---

## Step 6 — The Complete RAG Pipeline

Now you assemble the entire question-to-answer cycle in `src/generation/answer.py`, merging the retrieval from Phase 4 with the generation from this phase:

```python
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from src.generation.llm import get_llm
from src.generation.prompt import build_prompt
from src.retrieval.retriever import retrieve


@dataclass
class Answer:
    text: str
    sources: list[dict]


def answer_question(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    query: str,
    language: str | None = None,
) -> Answer:
    """Complete RAG pipeline: question → retrieval → generation → answer with citations."""
    # 1. Retrieve the correct snippets (Phase 4)
    retrieved = retrieve(index, nodes, query, language=language)

    # 2. Build the prompt and generate the response (Phase 5)
    prompt = build_prompt(query, retrieved)
    response = get_llm().complete(prompt)

    # 3. Extract sources for citations
    sources = [
        {
            "file_path": n.metadata.get("file_path"),
            "symbol": n.metadata.get("symbol"),
            "start_line": n.metadata.get("start_line"),
            "end_line": n.metadata.get("end_line"),
            "score": round(n.score, 3) if n.score is not None else None,
        }
        for n in retrieved
    ]
    return Answer(text=str(response), sources=sources)


if __name__ == "__main__":
    from src.config import REPO_PATH
    from src.indexing.build_index import build_index
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository
    from src.settings import configure_local_models

    configure_local_models()
    nodes = chunk_documents(load_repository(REPO_PATH))
    index = build_index(REPO_PATH)

    result = answer_question(
        index, nodes, "How is input data validated in this project?"
    )
    print(result.text)
    print("\n--- Sources ---")
    for s in result.sources:
        print(f"  {s['file_path']} [{s['symbol']}] lines {s['start_line']}-{s['end_line']}")
```

This is the complete RAG pipeline, and it is worth appreciating what it represents: the entire project converges in these few lines. The correct snippets are retrieved (thanks to the ingestion, chunking, embeddings, and retrieval from the previous phases), a prompt is built that forces grounding and citations, the local LLM generates the response, and structured sources are returned. The `Answer` object packages the response and its sources, ready to be displayed in the interface you will build in the next phase.

---

## Step 7 — Test the Complete Pipeline

Run it (`uv run python -m src.generation.answer`) and witness the magic: the system takes your question, retrieves the relevant code, and generates a response grounded in it, followed by the sources that back it up. For the first time, you have the complete assistant answering questions about real code.

Try various types of queries to see the system in action. An **explanatory** question ("how does X work in this project?") should give you a correct answer citing the relevant files. A request for **code generation** ("write a test for function Y following the project style") should produce code that fits the repository's conventions. And, very importantly, a question **about something not in the code** should make the assistant acknowledge that it cannot answer, instead of making it up—this is the ultimate proof that grounding is working.

---

## Step 8 — Tests

We verify two things: that the prompt is built correctly (this does not require the model) and that the complete pipeline works (this does, as an integration test). Complete `tests/test_generation.py`:

```python
import pytest

from llama_index.core.schema import NodeWithScore, TextNode

from src.generation.prompt import build_prompt
from src.indexing.embeddings import get_embedding_model


def _node(text: str, file_path: str, symbol: str) -> NodeWithScore:
    node = TextNode(text=text, metadata={"file_path": file_path, "symbol": symbol})
    return NodeWithScore(node=node, score=0.9)


def test_prompt_includes_context_and_grounding_rules():
    nodes = [_node("def f(): pass", "x.py", "f")]
    prompt = build_prompt("what does f do?", nodes)
    # The context and its source are in the prompt
    assert "def f()" in prompt
    assert "x.py" in prompt
    # Grounding rules are present
    assert "ONLY" in prompt
    assert "what does f do?" in prompt


def _ollama_available() -> bool:
    try:
        get_embedding_model().get_text_embedding("test")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Requires Ollama running")
def test_answer_cites_correct_source(tmp_path):
    from src.generation.answer import answer_question
    from src.indexing.build_index import build_index
    from src.indexing.vector_store import get_vector_store
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository
    from src.settings import configure_local_models

    configure_local_models()
    (tmp_path / "auth.py").write_text(
        "def validate_token(token):\n"
        "    '''Verifies that a session token is valid.'''\n"
        "    return len(token) == 32\n"
    )
    nodes = chunk_documents(load_repository(tmp_path))
    store = get_vector_store(persist_dir=tmp_path / "vs", collection="test")
    index = build_index(tmp_path, vector_store=store)

    result = answer_question(index, nodes, "how is a token validated?")
    assert len(result.text) > 0
    # The correct file must be among the cited sources
    assert any(s["file_path"] == "auth.py" for s in result.sources)
```

The first test verifies, without requiring the model, that the prompt includes the context with its source, the grounding rules, and the question. The second, an integration test, checks that when asking about token validation, the system generates a response and the correct file is among its sources. It is skipped if Ollama is not available. Run them with `make test`.

---

## Verification: The Definition of "Done"

The phase is complete when the following are met:

- [ ] `llm.py` configures the local LLM client with appropriate timeout and context window.
- [ ] `prompt.py` builds the RAG prompt by combining instructions, context, and the question.
- [ ] The prompt forces grounding (answering only from the context, acknowledging when it cannot) and citations.
- [ ] The system generates code aware of the project's patterns.
- [ ] The complete RAG pipeline (question → retrieval → generation → answer with citations) works end-to-end.
- [ ] **The key test:** You ask a question about the code and get an accurate, grounded response that cites the specific files on which it is based.

The key test is the complete cycle: if you ask a question about the repository and get a correct, grounded answer based on the real code with its sources cited, you have built a working RAG assistant. And if, additionally, when faced with a question with no answer in the code the assistant acknowledges this instead of making things up, you have built a *reliable* one—which is what really matters.

---

## Deliverables and What Comes Next

By closing Phase 5, you have completed the **entire RAG pipeline**: the question-to-answer cycle works from start to finish, with a local LLM that grounds itself in the actual code and cites its sources. You have brought together all the work from the previous phases into an assistant that truly answers questions about code, and you have added the two levers that make it reliable: grounding (which prevents hallucinations) and citations (which make it verifiable). It is the heart of the project, fully beating.

The next step, **Phase 6**, turns this engine into a **usable product**: the **API and the user interface**. You will wrap the pipeline in an API using FastAPI (with endpoints to index a repository and to ask questions) and build a chat interface to converse with the assistant and see the citations. Up until now, the system has worked from the command line; starting with the next phase, anyone will be able to use it from a chat screen. You have gone from "the assistant answers based on the code" to being on the verge of "anyone can converse with it."
