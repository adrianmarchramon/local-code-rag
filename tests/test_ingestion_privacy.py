"""Import-time privacy: the ingestion stack must not touch the network when imported.

The autouse guard in conftest.py starts after collection, so it cannot see what a module
does while being imported. This test starts a fresh, isolated interpreter (``-I``),
activates :class:`NetworkGuard` *before* importing anything else, imports the modules in
``IMPORTS_UNDER_GUARD``, runs the Step 1-3 filters on a small synthetic repository, and
reports the guard's violations. Step 4 adds ``llama_index.core`` to the list.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IMPORTS_UNDER_GUARD: tuple[str, ...] = (
    "pathspec",
    "src.ingestion.reasons",
    "src.ingestion.rules",
    "src.ingestion.ignore",
    "src.ingestion.classify",
    "src.ingestion.binary",
    # Step 4-5 (appended): LlamaIndex and the loader stack.
    "llama_index.core",
    "src.ingestion.walker",
    "src.ingestion.content",
    "src.ingestion.documents",
    "src.ingestion.dedup",
    "src.ingestion.report",
    "src.ingestion.loader",
    # Phase 2 Steps 1-2 (appended): tree-sitter, the grammar and the segmenter stack.
    "tree_sitter",
    "tree_sitter_python",
    "src.ingestion.spans",
    "src.ingestion.chunker",
    "src.ingestion.parser_registry",
    "src.ingestion.segmenter",
    # Phase 2 Steps 3-6 (appended): strategies, fallbacks, nodes, report and the chunker.
    "src.ingestion.lines",
    "src.ingestion.markdown_splitter",
    "src.ingestion.config_splitter",
    "src.ingestion.text_splitter",
    "src.ingestion.strategies",
    "src.ingestion.nodes",
    "src.ingestion.chunk_report",
    # Phase 3 Steps 1-4 (appended): Chroma, its LlamaIndex integration and the index stack.
    "chromadb",
    "llama_index.vector_stores.chroma",
    "src.ingestion.digest",
    "src.ingestion.repo_head",
    "src.indexing.embeddings",
    "src.indexing.manifest",
    "src.indexing.vector_store",
    "src.indexing.build_index",
    # Phase 3 Steps 5-7 (appended): loading, search and retrieval evaluation.
    "src.indexing.loading",
    "src.indexing.search",
    "src.evaluation.retrieval",
)

_CHILD = textwrap.dedent(
    """
    import importlib, json, sys, tempfile
    from pathlib import Path

    repo_root, modules = sys.argv[1], json.loads(sys.argv[2])
    sys.path.insert(0, repo_root)
    from src.network_guard import NetworkGuard

    already_loaded = sorted(m for m in modules if m in sys.modules)
    with NetworkGuard() as guard:
        for name in modules:
            importlib.import_module(name)
        from src.ingestion.binary import probe_head
        from src.ingestion.classify import classify_name
        from src.ingestion.ignore import IgnorePolicy
        from src.ingestion.rules import IngestionRules

        rules = IngestionRules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            (root / ".gitignore").write_text("*.log\\n")
            (root / "sub" / ".gitignore").write_text("!keep.log\\n")
            (root / "sub" / "a.py").write_text("x = 1\\n")
            policy = IgnorePolicy(root, rules, Path(repo_root) / "config" / "ingest.ignore")
            verdicts = [
                policy.is_ignored("sub/keep.log", is_dir=False).ignored,
                policy.is_ignored("deploy/model/x.py", is_dir=False).ignored,
                classify_name("sub/a.py", rules).language,
                probe_head(root / "sub" / "a.py", rules.head_bytes).value,
            ]
    print(json.dumps({
        "already_loaded": already_loaded,
        "violations": [repr(v) for v in guard.violations],
        "verdicts": verdicts,
    }))
    """
)


def test_ingestion_imports_and_filters_make_no_network_access(tmp_path: Path) -> None:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LC_ALL": "C.UTF-8"}

    # S603: trusted input only (this interpreter, a constant script, repo-controlled args).
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-c",
            _CHILD,
            str(REPO_ROOT),
            json.dumps(IMPORTS_UNDER_GUARD),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["already_loaded"] == []  # nothing was imported before the guard
    assert report["violations"] == []
    assert report["verdicts"] == [False, True, "python", "text"]


# --- Step 4-5: a full load, LlamaIndex's bundled caches, and no writes outside a temp dir ---

_LOAD_CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, repo_root)
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from llama_index.core.schema import MetadataMode
        from llama_index.core.utils import get_tokenizer, globals_helper
        from src.ingestion.loader import load_repository_with_report

        repo = work / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "a.py").write_text("def f():\\n    return 1\\n")
        (repo / "README.md").write_text("# Title\\n\\nOne. Two.\\n")
        result = load_repository_with_report(
            repo, rules_file=Path(repo_root) / "config" / "ingest.ignore"
        )
        seen = {
            mode.name: [d.get_content(metadata_mode=mode) for d in result.documents]
            for mode in MetadataMode
        }
        # What Phase 2 will use: these must come from the caches bundled in the wheel.
        tokens = len(get_tokenizer()("def f(): return 1"))
        stopwords = len(globals_helper.stopwords)
        sentences = len(globals_helper.punkt_tokenizer.tokenize("One. Two."))
    outside_repo = sorted(
        p.relative_to(work).as_posix() for p in work.rglob("*") if "repo" not in p.parts
    )
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "doc_ids": [d.doc_id for d in result.documents],
        "embed_hides_hash": all("content_hash" not in t for t in seen["EMBED"]),
        "tokens": tokens,
        "stopwords": stopwords,
        "sentences": sentences,
        "files_written_outside_repo": outside_repo,
    }))
    """
)


def test_full_load_and_bundled_caches_need_no_network_and_write_nothing(
    tmp_path: Path,
) -> None:
    # Production-like environment: HOME, XDG_CACHE_HOME and TMPDIR (where tiktoken and
    # NLTK would download to) point to an empty directory; NLTK_DATA and
    # TIKTOKEN_CACHE_DIR stay unset, because setting them disables the bundled caches.
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work),
        "XDG_CACHE_HOME": str(work),
        "TMPDIR": str(work),
        "LC_ALL": "C.UTF-8",
    }

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _LOAD_CHILD, str(REPO_ROOT), str(work)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["doc_ids"] == ["README.md", "src/a.py"]
    assert report["embed_hides_hash"] is True
    assert report["tokens"] > 0
    assert report["stopwords"] > 0
    assert report["sentences"] == 2
    assert report["files_written_outside_repo"] == []


# --- Phase 2 Steps 1-2: loading the grammar, parsing and segmenting need no network ------

_SEGMENT_CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, repo_root)
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from src.ingestion.chunker import ChunkingRules
        from src.ingestion.parser_registry import grammar_for, load_language
        from src.ingestion.segmenter import segment

        spec = grammar_for("python")
        abi = load_language(spec).abi_version  # first use of the grammar
        text = "@d\\nclass C:\\n    def m(self):\\n        return 'caf\\u00e9'\\n"
        result = segment(text, spec, max_chars=ChunkingRules().max_chunk_chars)
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "abi": abi,
        "kinds": [s.kind.value for s in result.spans],
        "files_written": sorted(p.relative_to(work).as_posix() for p in work.rglob("*")),
    }))
    """
)


def test_grammar_load_parse_and_segment_need_no_network_and_write_nothing(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work),
        "XDG_CACHE_HOME": str(work),
        "TMPDIR": str(work),
        "LC_ALL": "C.UTF-8",
    }

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _SEGMENT_CHILD, str(REPO_ROOT), str(work)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["abi"] == 15
    assert report["kinds"] == ["class"]
    assert report["files_written"] == []


# --- Phase 2 Steps 3-6: the whole chunking pipeline needs no network and writes nothing ---

_CHUNK_CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, repo_root)
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from llama_index.core import Document
        from llama_index.core.schema import MetadataMode
        from llama_index.core.vector_stores.utils import (
            metadata_dict_to_node,
            node_to_metadata_dict,
        )
        from src.ingestion.chunker import chunk_documents_with_report

        samples = {
            "src/app.py": ("python", "@d\\nclass C:\\n    def m(self):\\n        return 1\\n"),
            "README.md": ("markdown", "# T\\n\\n```\\n# code\\n```\\n\\n## S\\n\\nText.\\n"),
            "ci.yml": ("yaml", "jobs:\\n  a:\\n    run: x\\n"),
            "pyproject.toml": ("toml", "[project]\\nname = 'x'\\n"),
            "Dockerfile": ("dockerfile", "FROM python\\nRUN echo hi\\n"),
            "Makefile": ("makefile", "all:\\n\\techo hi\\n"),
            "web/app.js": ("javascript", "function a() {}\\n"),
        }
        documents = [
            Document(
                id_=path,
                text=text,
                metadata={
                    "file_path": path,
                    "language": language,
                    "size_bytes": len(text.encode()),
                    "num_lines": text.count("\\n"),
                    "content_hash": "0" * 64,
                    "directory": path.rpartition("/")[0],
                },
            )
            for path, (language, text) in samples.items()
        ]
        result = chunk_documents_with_report(documents)
        views = [n.get_content(metadata_mode=mode) for n in result.nodes for mode in MetadataMode]
        round_trip = all(
            metadata_dict_to_node(node_to_metadata_dict(n, flat_metadata=True)).hash == n.hash
            for n in result.nodes
        )
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "strategies": sorted(result.report.by_strategy),
        "nodes": len(result.nodes),
        "views": len(views),
        "round_trip": round_trip,
        "files_written": sorted(p.relative_to(work).as_posix() for p in work.rglob("*")),
    }))
    """
)


def test_the_whole_chunking_pipeline_needs_no_network_and_writes_nothing(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work),
        "XDG_CACHE_HOME": str(work),
        "TMPDIR": str(work),
        "LC_ALL": "C.UTF-8",
    }

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _CHUNK_CHILD, str(REPO_ROOT), str(work)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["strategies"] == ["config", "markdown", "syntactic", "text"]
    assert report["nodes"] >= 7
    assert report["views"] == 4 * report["nodes"]
    assert report["round_trip"] is True
    assert report["files_written"] == []


_INDEX_CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path[:0] = [repo_root, repo_root + "/tests"]
    from src.network_guard import NetworkGuard

    with NetworkGuard() as guard:
        from fake_embedding import FakeEmbedding
        from src.indexing.build_index import build_index_with_report
        from src.indexing.vector_store import ChromaIndexStore

        store = ChromaIndexStore(work / "vs", "code")
        first = build_index_with_report(
            work / "repo", store, FakeEmbedding(), rules_file=work / "rules"
        )
        second = build_index_with_report(
            work / "repo", store, FakeEmbedding(), rules_file=work / "rules"
        )
        hits = second.index.as_retriever(similarity_top_k=1).retrieve("token")
    written = sorted(
        p.relative_to(work).as_posix()
        for p in work.rglob("*")
        if not p.relative_to(work).as_posix().startswith(("repo", "vs", "rules", ".env"))
    )
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "nodes": first.report.nodes,
        "second_embedded": second.report.embedded,
        "hits": len(hits),
        "onnxruntime": "onnxruntime" in sys.modules,
        "written_outside_repo_and_store": written,
    }))
    """
)


def test_indexing_needs_no_network_and_writes_only_its_store_even_with_a_hostile_env(
    tmp_path: Path,
) -> None:
    """Phase 3 (Steps 1-4): load, chunk, embed (fake), write, verify, activate and query in
    a fresh interpreter, with a hostile .env in the working directory and hostile CHROMA_*
    and proxy variables. Nothing may leave the machine or land outside the store."""
    work = tmp_path / "work"
    (work / "repo" / "src").mkdir(parents=True)
    (work / "repo" / "src" / "auth.py").write_text(
        "def verify_token(token):\n    return token == 'ok'\n", encoding="utf-8"
    )
    (work / "repo" / "README.md").write_text("# Tokens\n\nHow tokens work.\n", encoding="utf-8")
    (work / "rules").write_text("", encoding="utf-8")
    (work / ".env").write_text(
        "CHROMA_API_IMPL=chromadb.api.fastapi.FastAPI\nCHROMA_SERVER_HOST=203.0.113.10\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work / "home"),
        "XDG_CACHE_HOME": str(work / "home"),
        "TMPDIR": str(work / "home"),
        "LC_ALL": "C.UTF-8",
        "CHROMA_API_IMPL": "chromadb.api.fastapi.FastAPI",
        "CHROMA_SERVER_HOST": "203.0.113.10",
        "CHROMA_OTEL_COLLECTION_ENDPOINT": "http://203.0.113.10:4317",
        "HTTP_PROXY": "http://203.0.113.10:3128",
        "HTTPS_PROXY": "http://203.0.113.10:3128",
        "ALL_PROXY": "http://203.0.113.10:3128",
    }
    (work / "home").mkdir()

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _INDEX_CHILD, str(REPO_ROOT), str(work)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["nodes"] >= 2
    assert (report["second_embedded"], report["hits"]) == (0, 1)
    assert report["onnxruntime"] is False
    assert report["written_outside_repo_and_store"] == ["home"]


_SEARCH_CHILD = textwrap.dedent(
    """
    import io, json, logging, sys
    from pathlib import Path

    repo_root, work = sys.argv[1], Path(sys.argv[2])
    sys.path[:0] = [repo_root, repo_root + "/tests"]
    from src.network_guard import NetworkGuard

    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    with NetworkGuard() as guard:
        from fake_embedding import FakeEmbedding
        from src.indexing.build_index import build_index_with_report, load_index_with_report
        from src.indexing.search import search
        from src.indexing.vector_store import ChromaIndexStore
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_repository

        store = ChromaIndexStore(work / "vs", "code")
        build_index_with_report(work / "repo", store, FakeEmbedding(), rules_file=work / "rules")
        nodes = chunk_documents(load_repository(work / "repo", rules_file=work / "rules"))
        loaded = load_index_with_report(nodes, vector_store=store, embed_model=FakeEmbedding())
        hits = search(loaded.index, "SENTINEL_QUERY_9d3e verify token", top_k=2)
        loaded.close()
    logs = capture.getvalue()
    print(json.dumps({
        "violations": [repr(v) for v in guard.violations],
        "hits": len(hits),
        "query_in_logs": "SENTINEL_QUERY_9d3e" in logs,
        "code_in_logs": "SENTINEL_CODE_5a1f" in logs,
        "onnxruntime": "onnxruntime" in sys.modules,
        "home": sorted(p.name for p in (work / "home").iterdir()),
    }))
    """
)


def test_loading_and_searching_need_no_network_and_log_no_content_even_with_a_hostile_env(
    tmp_path: Path,
) -> None:
    """Phase 3 (Steps 5-6): build, load with drift check and search in a fresh interpreter
    with a hostile .env, hostile CHROMA_* and proxy variables, and the root logger at
    DEBUG. Nothing may leave the machine; neither the query nor the code may be logged."""
    work = tmp_path / "work"
    (work / "repo" / "src").mkdir(parents=True)
    (work / "repo" / "src" / "auth.py").write_text(
        "def verify_token(token):\n    return token == 'SENTINEL_CODE_5a1f'\n", encoding="utf-8"
    )
    (work / "rules").write_text("", encoding="utf-8")
    (work / "home").mkdir()
    (work / ".env").write_text(
        "CHROMA_API_IMPL=chromadb.api.fastapi.FastAPI\nCHROMA_SERVER_HOST=203.0.113.10\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(work / "home"),
        "XDG_CACHE_HOME": str(work / "home"),
        "TMPDIR": str(work / "home"),
        "LC_ALL": "C.UTF-8",
        "CHROMA_API_IMPL": "chromadb.api.fastapi.FastAPI",
        "CHROMA_SERVER_HOST": "203.0.113.10",
        "HTTP_PROXY": "http://203.0.113.10:3128",
        "HTTPS_PROXY": "http://203.0.113.10:3128",
        "ALL_PROXY": "http://203.0.113.10:3128",
    }

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _SEARCH_CHILD, str(REPO_ROOT), str(work)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["violations"] == []
    assert report["hits"] == 1  # one chunk in the repository
    assert (report["query_in_logs"], report["code_in_logs"]) == (False, False)
    assert (report["onnxruntime"], report["home"]) == (False, [])
