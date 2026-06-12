# Phase 2 — Syntax-Aware Code Chunking

> **This is the key phase of the project, where a large part of its quality is decided.** Remember the most important concept from the fundamentals: in RAG, quality is determined by retrieval, not generation. And retrieval is built on chunks: if you split the code poorly, you retrieve poorly, and then not even the best model will provide good answers. Chunking is, therefore, the foundation upon which the entire system rests, and doing it well for code comes with its own set of difficulties that almost no generic RAG tutorial addresses. This is why this phase distinguishes your project the most: it demonstrates that you have understood that code is not just any text, and that it must be split while respecting its structure.

**Phase goal:** chunk the code while respecting its structure.  
**Duration:** week 2.  
**By the end, you will have:** a chunking system that converts ingested documents into coherent chunks based on the code's structure (complete functions, classes), each with rich metadata (file, symbol, line range), ready for vectorization.

---

## The Big Picture

Chunking takes the clean documents produced during ingestion and splits them into chunks of the appropriate size for vectorization and retrieval, while respecting the logic of the code:

```
   Code documents (from Phase 1, with labeled language)
        │
        ▼
   [ Chunk according to language syntax ]  ──►  by functions, classes, methods
        │
        ▼
   [ Enrich metadata ]  ──►  file (inherited) + symbol + line range
        │
        ▼
   [ Handle oversized chunks ]  ──►  fallback ensuring coverage
        │
        ▼
   Coherent chunks with metadata → (feed the embeddings in Phase 3)
```

For syntax-aware chunking, we will use LlamaIndex's **CodeSplitter**, which relies on **tree-sitter** (a parser that understands the structure of programming languages). Add the dependencies:

```bash
uv add llama-index tree-sitter tree-sitter-language-pack
```

An important current note: the parser library we will use is **`tree-sitter-language-pack`**, which is the current replacement for the old `tree-sitter-languages`. This matters because the old one caused many compilation issues (requiring compilation of the parsers for each language), whereas the new one comes with pre-built parsers and works hassle-free. Using the right library saves you hours of battling compilation errors and shows that you are up to date.

---

## Step 1 — Why Naive Chunking Destroys Code

This is the most important concept of this phase, so it deserves to be developed carefully, as understanding it justifies everything else. The default approach in RAG, and the one used by generic tutorials, is **naive chunking**: cutting text every N characters or every N lines, without looking at the content. For prose, this works reasonably well: a paragraph cut in half remains largely readable, and the text degrades gracefully. **For code, it is disastrous.**

The reason is that code has a rigid structure and internal dependencies that a blind cut breaks. Imagine this function:

```python
def validar_token(token: str, clave_secreta: str) -> bool:
    """Verifica que un token JWT sea válido y no haya expirado."""
    try:
        payload = jwt.decode(token, clave_secreta, algorithms=["HS256"])
        return payload["exp"] > time.time()
    except jwt.InvalidTokenError:
        return False
```

If a naive chunker were to cut every four lines, it could split this in two: a first chunk with the function signature and the beginning of the body, and a second chunk with the end, detached, without context. Think about what happens to each piece. The first chunk has half a function, without its complete logic. The second chunk is an orphan snippet of code (`except jwt.InvalidTokenError: return False`) which, on its own, means almost nothing: which function does it belong to? what does it do?

The consequences for retrieval are severe, and there are two of them. First, **embeddings become useless**: the vector of a chunk that is "half a function" does not capture any coherent concept because the chunk does not represent a complete logical unit. When someone asks "how are tokens validated?", the system will struggle to find that split chunk because its vector does not accurately represent the idea of "validating tokens". Second, even if it were retrieved by chance, **it would confuse the model**: passing half a function as context means giving it incomplete information, which will generate a worse response.

The conclusion is clear: **blindly chunking code sabotages retrieval from the root**, no matter how good everything else is. Conversely, splitting it while respecting its structure (ensuring each chunk is a complete function or a complete class) makes each chunk a meaningful unit whose embedding captures what it does well and, when retrieved, provides the model with complete and useful context. That is the heart of this phase.

---

## Step 2 — Syntax-Aware Chunking

The solution is to chunk code based on its **syntactic structure**: instead of cutting every N lines, cut at the logical boundaries of the code (between functions, between classes, between methods). To know where those boundaries are, it is necessary to *understand* the language's syntax, and that is exactly what a parser like tree-sitter does: it reads the code and builds its syntax tree, identifying what each element is (a function, a class, a method). LlamaIndex's **CodeSplitter** uses tree-sitter under the hood to chunk code by these logical units.

The important piece, and where the fruits of Phase 1 are seen, is that CodeSplitter is **language-specific**: it needs to know which language the code is in to use the correct parser. And we have that language, because we labeled it on each document during ingestion. To create a splitter for a language, we pass the corresponding parser from the parser library:

```python
from llama_index.core.node_parser import CodeSplitter
from tree_sitter_language_pack import get_parser


def make_code_splitter(language: str) -> CodeSplitter:
    return CodeSplitter.from_defaults(
        language=language,
        parser=get_parser(language),
        chunk_lines=40,
        chunk_lines_overlap=15,
        max_chars=1500,
    )
```

It is helpful to understand the parameters, as they define the chunking behavior. The `language` and the `parser` (obtained from `tree-sitter-language-pack`) are what make the chunking syntax-aware for that specific language; passing the parser explicitly from the parser library is the most robust approach, as it avoids resolution and compilation issues. The `chunk_lines` (40) indicate the target size of the chunks in lines, and `chunk_lines_overlap` (15) is an overlap between adjacent chunks, which helps prevent context loss at the boundaries. Lastly, `max_chars` (1500) is a size limit that acts as a safeguard. These values are a good starting point; in the evaluation phase (Phase 7), you will be able to tune them and measure if they improve retrieval.

---

## Step 3 — Chunking According to the Language of Each File

Since chunking is language-specific, the pattern is to group the documents by their language and chunk each group with its parser. This is where the language metadata you set during ingestion pays off:

```python
from collections import defaultdict

from llama_index.core import Document
from llama_index.core.schema import BaseNode

# Languages for which we have syntax-aware chunking
SYNTAX_AWARE_LANGUAGES = {
    "python", "javascript", "typescript", "java", "go", "rust", "c", "cpp",
}


def chunk_by_language(documents: list[Document]) -> list[BaseNode]:
    by_language: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_language[doc.metadata["language"]].append(doc)

    all_nodes: list[BaseNode] = []
    for language, docs in by_language.items():
        if language in SYNTAX_AWARE_LANGUAGES:
            splitter = make_code_splitter(language)
            nodes = splitter.get_nodes_from_documents(docs)
        else:
            nodes = fallback_split(docs)  # languages without parser (see Step 5)
        all_nodes.extend(nodes)
    return all_nodes
```

Each language is chunked with the appropriate parser, and languages for which we do not have a parser (such as Markdown) are chunked using a fallback method that we will see in Step 5. A valuable detail: CodeSplitter **automatically inherits metadata** from the parent document to each child chunk. This means that the `file_path` and `language` you set during ingestion travel automatically to each chunk without you having to do anything. This inheritance is the basis for citations, as we will see.

---

## Step 4 — Enriched Metadata: Symbol and Line Range

Chunks inherit the source file, but for truly precise citations, we want two more pieces of metadata: the **symbol** (the name of the function or class in the chunk) and the **line range** it occupies. With these, the assistant will be able to cite with surgical precision: "I know this because of the `validar_token` function in `src/auth.py`, lines 40-58." This precision is what turns a vague citation into a verifiable one.

The **line range** is calculated from the chunk's position within its file, which LlamaIndex records in each node. By counting the line breaks of the original file up to the start and end of the chunk, we obtain its lines:

```python
def enrich_line_range(node: BaseNode, source_text: str) -> None:
    if node.start_char_idx is not None and node.end_char_idx is not None:
        node.metadata["start_line"] = source_text.count("\n", 0, node.start_char_idx) + 1
        node.metadata["end_line"] = source_text.count("\n", 0, node.end_char_idx) + 1
```

The **symbol name** requires looking at the chunk's syntax, and here we use tree-sitter directly: we parse the chunk, look for the node that defines a function or a class, and extract its name. The node types that define a symbol depend on the language (in Python, they are `function_definition` and `class_definition`), so we map them:

```python
from tree_sitter_language_pack import get_parser

DEF_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    # expand based on the languages you support
}


def extract_symbol_name(code: str, language: str) -> str | None:
    """Name of the function or class defined by the chunk, via syntax tree."""
    def_types = DEF_NODE_TYPES.get(language, set())
    if not def_types:
        return None
    tree = get_parser(language).parse(bytes(code, "utf-8"))
    for node in tree.root_node.children:
        if node.type in def_types:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return name_node.text.decode("utf-8")
    return None
```

This function parses the chunk's code, traverses the top-level nodes looking for a function or class definition, and returns the symbol name (using `child_by_field_name("name")`, which is tree-sitter's way of obtaining the name field). It is language-specific, hence the map of node types, which you can expand based on the languages you want to support with precision.

---

## Step 5 — Handling Oversized Chunks: The Fallback

A robust system must account for difficult cases, and in code chunking, there are two: a file that is a single massive function (larger than the target chunk size), or a file in a language for which you do not have a parser. The philosophy for handling them is clear: **always prefer clean and structured chunks, but guarantee that no code is lost**.

CodeSplitter already largely handles the first case: when a logical unit is too large, its `chunk_lines` and `max_chars` parameters act as a limit and subdivide it. For the second case (languages without a parser), we resort to a text-based fallback splitter:

```python
from llama_index.core.node_parser import SentenceSplitter


def fallback_split(docs: list[Document]) -> list[BaseNode]:
    """Fallback text-based chunking. Less clean, but guarantees coverage."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.get_nodes_from_documents(docs)
```

It is worth being honest about this trade-off, as it demonstrates that you understand what you are doing: fallback chunks are **less clean** than structured ones (they can start or end mid-function, exactly what we wanted to avoid). However, they serve an important purpose: they guarantee that all code is covered, with nothing lost due to lacking a parser for its language. It is a conscious trade-off: you accept imperfect chunks for these minority cases in exchange for not leaving any code out of the index. Ideally, most of the code (from the repository's primary language) is chunked in a structured manner, and only marginal files fall back.

---

## Step 6 — The Complete Chunker

Now, you put all the pieces together in `src/ingestion/chunker.py`: group by language, chunk each language in a syntax-aware manner (with a fallback), and enrich each chunk with its line range and symbol.

```python
from collections import defaultdict

from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import BaseNode
from tree_sitter_language_pack import get_parser

SYNTAX_AWARE_LANGUAGES = {
    "python", "javascript", "typescript", "java", "go", "rust", "c", "cpp",
}

DEF_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
}


def make_code_splitter(language: str) -> CodeSplitter:
    return CodeSplitter.from_defaults(
        language=language,
        parser=get_parser(language),
        chunk_lines=40,
        chunk_lines_overlap=15,
        max_chars=1500,
    )


def extract_symbol_name(code: str, language: str) -> str | None:
    def_types = DEF_NODE_TYPES.get(language, set())
    if not def_types:
        return None
    tree = get_parser(language).parse(bytes(code, "utf-8"))
    for node in tree.root_node.children:
        if node.type in def_types:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return name_node.text.decode("utf-8")
    return None


def _enrich(node: BaseNode, language: str, source_text: str | None) -> None:
    if source_text and node.start_char_idx is not None and node.end_char_idx is not None:
        node.metadata["start_line"] = source_text.count("\n", 0, node.start_char_idx) + 1
        node.metadata["end_line"] = source_text.count("\n", 0, node.end_char_idx) + 1
    symbol = extract_symbol_name(node.get_content(), language)
    if symbol is not None:
        node.metadata["symbol"] = symbol


def chunk_documents(documents: list[Document]) -> list[BaseNode]:
    """Chunks the documents while respecting the code structure, by language."""
    by_language: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_language[doc.metadata["language"]].append(doc)

    text_by_id = {doc.doc_id: doc.text for doc in documents}
    all_nodes: list[BaseNode] = []

    for language, docs in by_language.items():
        if language in SYNTAX_AWARE_LANGUAGES:
            try:
                nodes = make_code_splitter(language).get_nodes_from_documents(docs)
            except Exception:
                nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(docs)
        else:
            nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(docs)

        for node in nodes:
            _enrich(node, language, text_by_id.get(node.ref_doc_id))
        all_nodes.extend(nodes)

    return all_nodes


if __name__ == "__main__":
    from src.config import REPO_PATH
    from src.ingestion.loader import load_repository

    documents = load_repository(REPO_PATH)
    nodes = chunk_documents(documents)
    print(f"{len(documents)} files → {len(nodes)} chunks")
    for node in nodes[:3]:
        m = node.metadata
        print(f"\n--- {m.get('file_path')} [{m.get('symbol', 'no symbol')}] "
              f"lines {m.get('start_line')}-{m.get('end_line')} ---")
        print(node.get_content()[:200])
```

Running it (`uv run python -m src.ingestion.chunker`), you will see how many chunks the repository has been split into and you will be able to inspect some of them with their metadata (file, symbol, lines), manually verifying that they are coherent units of code.

---

## Step 7 — Tests

We wrap up by verifying the most important aspect of this phase: that chunks are coherent units and that metadata is generated correctly. Complete `tests/test_chunking.py`:

```python
from llama_index.core import Document

from src.ingestion.chunker import chunk_documents, extract_symbol_name


def _doc(code: str) -> Document:
    return Document(text=code, metadata={"language": "python", "file_path": "x.py"})


def test_function_stays_whole():
    code = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    contents = [n.get_content() for n in chunk_documents([_doc(code)])]
    # Each function must appear complete in some chunk, not split
    assert any("def alpha" in c and "return 1" in c for c in contents)
    assert any("def beta" in c and "return 2" in c for c in contents)


def test_extracts_symbol_name():
    assert extract_symbol_name("def saludar():\n    pass\n", "python") == "saludar"
    assert extract_symbol_name("class Motor:\n    pass\n", "python") == "Motor"


def test_metadata_inherited_and_enriched():
    nodes = chunk_documents([_doc("def f():\n    x = 1\n    return x\n")])
    node = nodes[0]
    assert node.metadata["file_path"] == "x.py"   # inherited from ingestion
    assert node.metadata["language"] == "python"
    assert "start_line" in node.metadata          # enriched in this phase
```

The first test verifies the central criterion of this phase: that a function is not split but instead appears complete in a single chunk. The second verifies that symbol extraction works. The third ensures that metadata inherited from ingestion is present and that new metadata (line range) has been added. Run them with `make test`.

---

## Verification: The "Definition of Done"

The phase is complete when the following are met:

- [ ] `chunker.py` chunks code in a syntax-aware manner, by language, using tree-sitter (via `tree-sitter-language-pack`).
- [ ] Each chunk inherits the source file and is enriched with the symbol and line range.
- [ ] There is a fallback strategy that guarantees coverage for languages without a parser or large chunks.
- [ ] The tests verify that functions are not split and that the metadata is correct.
- [ ] **The key test:** the generated chunks are meaningful units of code (a complete function, a complete class), not arbitrary cuts in the middle of a line.

The key test is that of coherence: if you inspect the chunks produced by your chunker on a real repository and they are meaningful units of code (complete functions and classes, not pieces cut mid-line), you have solved the most difficult and characteristic part of the project. And, as we saw at the beginning, this coherence is the prerequisite for retrieval (and thus the entire system) to work well.

---

## Deliverables and What Comes Next

By closing Phase 2, you have solved the most characteristic challenge of the project: chunking code while respecting its structure, so that each chunk is a complete logical unit with rich metadata. You have understood and applied the idea that separates a code RAG from a generic tutorial (that code is not just any text), and you have built the foundation upon which the quality of all retrieval will rest. The symbol and line range metadata you added will also enable surgical citations later on.

The next step, **Phase 3**, brings the search to life: **embeddings and the vector database**. You will vectorize each of these coherent chunks using the local embedding model and store them in the vector database along with their metadata, building the complete index of the repository. Here, you will see why it was worth chunking properly: since each chunk is a meaningful unit, its embedding will accurately capture what it does, and the similarity search will actually work. You have gone from "the code is chunked meaningfully" to being on the verge of "the repository is searchable by meaning."
