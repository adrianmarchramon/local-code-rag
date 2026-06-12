# Phase 1 — Ingestion: Loading and Filtering Code

> This is the first phase of building the system, and although it might seem like the least glamorous, it is the foundation for everything else. Remember the fundamental principle: the quality of a RAG system is determined by retrieval, and retrieval can only be good if what you index is good. "Garbage in, garbage out" applies here too: if you index noise (third-party dependencies, compiled files, irrelevant configuration files), that noise will contaminate your knowledge base, waste compute, and cause irrelevant search results to appear. That is why ingestion is not just about "reading files," but about **reading the right files**, filtering carefully to keep only the code that matters.

**Phase objective:** Turn a repository into a clean collection of code documents.  
**Duration:** Weeks 1-2.  
**By the end, you will have:** A function that, given a repository path, traverses it, intelligently filters out non-relevant code, detects the language of each file, associates rich metadata with it, and returns a clean collection of documents ready for the next phase.

---

## The Big Picture

Ingestion is the first link in the indexing pipeline. Its job is to transform a repository (a tangle of files of all kinds) into an organized collection of code documents, each with its metadata:

```
   Repository (folder with files of all kinds)
        │
        ▼
   [ Traverse all files ]
        │
        ▼
   [ Filter ]  ──►  ignore dependencies, builds, binaries, whatever is in .gitignore
        │
        ▼
   [ Detect language ]  ──►  python, javascript, etc.
        │
        ▼
   [ Associate metadata ]  ──►  path, language, size, lines
        │
        ▼
   Clean collection of code documents → (feeds the chunking in Phase 2)
```

We will build this by producing **LlamaIndex** `Document` objects, which is the type the rest of the pipeline works with; this way, the output of the ingestion phase fits directly into the subsequent phases. To filter out ignored files, we will use the `pathspec` library, which understands the rules of a `.gitignore`. Add the dependencies:

```bash
uv add llama-index pathspec
```

---

## Step 1 — What is "Relevant Code" and What is Noise

Before diving into the code, it is best to have a clear filtering strategy, as this is the central decision of this phase. A typical repository contains many things that are **not** source code you want to index, and understanding what they are and why to discard them is what distinguishes a carefully designed ingestion from one that feeds garbage into the system.

What you **do** want to index is the project's source code: files in the programming languages that the project uses (Python, JavaScript, etc.), which is where the logic the assistant needs to understand resides.

What you **do not** want to index is abundant, and it is important to recognize it:

**Third-party dependency folders**, such as `node_modules`, `.venv`, or `vendor`, contain code that does not belong to the project, but rather to the libraries it uses. Indexing them would be a massive mistake: they would multiply your index size tenfold or a hundredfold, filling it with external code that would drown out your own code during searches.

**Build and cache artifacts**, such as `dist`, `build`, `target`, or `__pycache__` folders, contain generated or compiled code, not source code, and offer nothing useful.

**Binary files**: images, executables, compiled files (`.pyc`), models, data. They are not readable text, so vectorizing them would produce meaningless embeddings.

And other irrelevant files, such as dependency lockfiles (`package-lock.json`), data, or secrets.

The underlying reason why this filtering matters so much is straightforward: what you index is exactly the universe in which your system can search. Every noisy file you index is wasted space, squandered embedding compute, and another candidate to show up as an irrelevant result, worsening retrieval. Aggressive and intelligent filtering is not optional; it is one of the levers that most heavily impacts the final quality of the system.

---

## Step 2 — Respecting the Repository's .gitignore

There is an elegant technique here worth highlighting because it shows good judgment: instead of guessing which files are irrelevant, you can take advantage of the fact that **the project itself has already declared them**. A repository's `.gitignore` file lists exactly what the project considers should not be version-controlled: dependencies, builds, secrets, data. It is, for free, an excellent list of what you also do not want to index, and it is perfectly aligned with the conventions of that specific project.

The `pathspec` library understands the `.gitignore` syntax and allows checking if a file matches its rules:

```python
from pathlib import Path

import pathspec


def load_gitignore(repo_path: Path) -> pathspec.PathSpec:
    """Loads the rules from the repository's .gitignore, if it exists."""
    gitignore = repo_path / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text().splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return pathspec.PathSpec.from_lines("gitwildmatch", [])
```

With this, for every file in the repository, you can query whether the `.gitignore` excludes it and skip it in that case. Relying on the project's own `.gitignore`, instead of maintaining a manual exclusion list yourself, is a detail that shows you think elegantly and respect the conventions of the repository you are indexing.

---

## Step 3 — Detecting Binaries and Language

There are two more checks per file, both of which are important.

The first is **discarding binaries**. Even if you filter by extension, it is wise to have a safeguard that detects disguised binary files. A simple and reliable heuristic is that text files do not contain null bytes, whereas binaries almost always do:

```python
def is_probably_binary(path: Path) -> bool:
    """Detects if a file is binary (not readable text)."""
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return True
    return b"\x00" in chunk  # a null byte reveals a binary file
```

The second is **detecting the language** of each file, which we will do based on its extension. This serves two purposes: it acts as a filter (we only keep known code extensions) and, very importantly, it records the language of each file, which we will need in Phase 2. This is because code-aware chunking is **language-specific** (a Python file is chunked differently from a JavaScript file), so knowing the language of each file is essential to chunking it properly later. We define a map of extensions to languages:

```python
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".md": "markdown",
    # expand with the languages you need
}
```

Any file whose extension is not in this map is discarded (it is not code we know how to handle), and those that are in it are labeled with their language.

---

## Step 4 — The Complete Loader

Now you bring all the pieces together in `src/ingestion/loader.py`: traverse the repository, apply all filters in order, detect the language, read the content, and build the documents with their metadata.

```python
from collections import Counter
from pathlib import Path

import pathspec
from llama_index.core import Document

LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cs": "csharp", ".php": "php", ".md": "markdown",
}

IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist",
    "build", ".next", "target", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "vendor", "site-packages",
}


def load_gitignore(repo_path: Path) -> pathspec.PathSpec:
    gitignore = repo_path / ".gitignore"
    if gitignore.exists():
        return pathspec.PathSpec.from_lines(
            "gitwildmatch", gitignore.read_text().splitlines()
        )
    return pathspec.PathSpec.from_lines("gitwildmatch", [])


def is_probably_binary(path: Path) -> bool:
    try:
        return b"\x00" in path.read_bytes()[:1024]
    except OSError:
        return True


def load_repository(repo_path: Path) -> list[Document]:
    """Traverses a repository and returns its code files as Documents."""
    repo_path = Path(repo_path)
    gitignore = load_gitignore(repo_path)
    documents: list[Document] = []

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(repo_path)

        # 1. Skip excluded folders (dependencies, builds, caches)
        if any(part in IGNORED_DIRS for part in rel_path.parts):
            continue
        # 2. Respect the project's .gitignore
        if gitignore.match_file(str(rel_path)):
            continue
        # 3. Keep only known code extensions (and detect language)
        language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
        if language is None:
            continue
        # 4. Discard disguised binaries
        if is_probably_binary(path):
            continue
        # 5. Read content (skipping whatever is not valid text)
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # 6. Build the Document with its metadata
        documents.append(
            Document(
                text=content,
                metadata={
                    "file_path": str(rel_path),
                    "language": language,
                    "size_bytes": path.stat().st_size,
                    "num_lines": content.count("\n") + 1,
                },
            )
        )

    return documents


if __name__ == "__main__":
    from src.config import REPO_PATH

    docs = load_repository(REPO_PATH)
    print(f"Loaded {len(docs)} code files")
    print(f"By language: {dict(Counter(d.metadata['language'] for d in docs))}")
```

The flow applies the filters in cascade and in an order designed to discard the "cheap" things first: first, the excluded folders (a simple path check), then the `.gitignore`, then the extension, then the binary check (which reads bytes from the file), and finally reading the content. Every file that survives all the filters is converted into a LlamaIndex `Document` with its text and metadata. Add the path of the repository you cloned in Phase 0 to your `src/config.py`:

```python
from pathlib import Path

REPO_PATH = Path.home() / "repos" / "project-to-index"
```

Running it (`uv run python -m src.ingestion.loader`), you will see how many code files have been loaded and a summary by language, giving you an initial X-ray of the repository.

---

## Step 5 — Metadata: The Foundation for Citations and Filtering

It is worth stopping to look at metadata because it is easy to treat it as a minor detail when in reality it is one of the most important decisions of this phase, with consequences in several future phases. Each document has its relative path, language, size, and number of lines associated with it, and this metadata **will travel with every chunk** throughout the entire pipeline.

Why this matters so much looking ahead:

The **file path** is what will make **citations** possible in Phase 5. When the assistant responds, it will be able to say, "I know this because of the file `src/auth/login.py`," and that source information comes precisely from this metadata. Without it, answers would be opaque and impossible to verify.

The **language** is what will enable **code-aware chunking** in Phase 2 (splitting each file using the parser for its specific language) and **metadata filtering** in Phase 4 (being able to search, for example, only within Python files).

The **size and line count** are useful for understanding the corpus and for chunking decisions.

The lesson is that metadata is not decorative: it is what connects the model's response to the real code, allows you to filter searches, and makes the system transparent. Attributing rich metadata right from ingestion is an investment in capabilities that will bear fruit in later phases.

---

## Step 6 — Handling Multiple Languages

A real repository often contains multiple languages (for example, Python on the backend and JavaScript on the frontend), and your ingestion must handle them all. The design we have built already does this naturally: the `LANGUAGE_BY_EXTENSION` map provides multi-language support, and expanding it is as simple as adding more extensions. Each file is labeled with its language, which allows it to be treated correctly later.

The important consideration for the future is that multi-language support does not end with ingestion: in Phase 2, code-aware chunking will require a parser for each language you want to chunk properly. Therefore, a reasonable strategy is to start by focusing on the primary language of the repository you are indexing (if it is mostly Python, perfect Python first) and add support for other languages incrementally. Labeling the language of all files now lays the groundwork for that.

---

## Step 7 — Tests

We close with tests that verify ingestion filters as it should, maintaining quality discipline. Complete `tests/test_ingestion.py`:

```python
from pathlib import Path

from src.ingestion.loader import load_repository


def _make_sample_repo(tmp_path: Path) -> Path:
    # Relevant code: YES, it should be loaded
    (tmp_path / "main.py").write_text("def hello():\n    return 'hi'\n")
    # Third-party dependency: NO, it should not be loaded
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("module.exports = {}\n")
    # Ignored by .gitignore: NO, it should not be loaded
    (tmp_path / ".gitignore").write_text("secret.py\n")
    (tmp_path / "secret.py").write_text("API_KEY = 'xxx'\n")
    # Binary: NO, it should not be loaded
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00datos")
    return tmp_path


def test_loads_only_relevant_code(tmp_path):
    docs = load_repository(_make_sample_repo(tmp_path))
    paths = {d.metadata["file_path"] for d in docs}
    assert "main.py" in paths
    assert "node_modules/lib.js" not in paths
    assert "secret.py" not in paths
    assert "logo.png" not in paths


def test_attaches_correct_metadata(tmp_path):
    docs = load_repository(_make_sample_repo(tmp_path))
    main = next(d for d in docs if d.metadata["file_path"] == "main.py")
    assert main.metadata["language"] == "python"
    assert main.metadata["num_lines"] >= 1
    assert main.metadata["size_bytes"] > 0
```

These tests build a miniature test repository and verify the essentials: that relevant code is loaded, dependencies are ignored (due to the excluded folder check), `.gitignore` rules and binaries are respected, and the metadata is correct. Testing that your ingestion properly discards garbage—and not just that it loads the good stuff—is precisely what guarantees the quality of your foundation. Run them with `make test`.

---

## Verification: The "Definition of Done"

The phase is complete when the following are met:

- [ ] `loader.py` traverses a repository and loads its code files.
- [ ] Filtering ignores dependency folders, build artifacts, binaries, and whatever is specified in the `.gitignore`.
- [ ] Each document contains rich metadata: path, language, size, and lines.
- [ ] The system handles multiple languages using the extension map.
- [ ] The tests verify that filtering works correctly.
- [ ] **The key test:** you point the ingestion at a real repository and get only the code files that matter, with no garbage.

The key test is the real repository: if you run the ingestion on the repo you chose and the result consists exclusively of its relevant code files—with no third-party dependencies, builds, or binaries sneaking in—you have built a clean foundation. And as we saw, a clean foundation is the prerequisite for everything built on top of it to function properly.

---

## Deliverables and What's Next

At the close of Phase 1, you have the first link in the pipeline: an ingestion process that converts a repository into a clean collection of code documents, intelligently filtering out noise, detecting the language of each file, and associating metadata that will bear fruit in future phases. You have solved something that is subtler than it seems (what to index and what not to), which directly affects the quality of the entire system.

The next step, **Phase 2**, is the most characteristic part of the project: **code-aware chunking**. You will take the documents produced by this ingestion and split them while respecting the structure of the code (by functions, classes, methods) rather than doing it blindly. Here is where you see the value of the language label on each document: it is what will allow you to split each file using the correct parser for its language. You have gone from "I have loaded the relevant code" to being on the verge of "I am chunking it in a way that makes retrieval actually work."
