"""Steps 1-5 end to end on a synthetic, corpus-shaped tree (invented content only).

Extends the Steps 1-3 tree with what Step 4 adds: symlinks, a FIFO, BOM and CRLF, a file
over the size cap, a NUL byte after the first KiB, non-UTF-8 bytes, empty and
whitespace-only files, duplicates and a nested .gitignore. Every exclusion is asserted
with its reason, through the report-returning loader.
"""

import os
from pathlib import Path

from llama_index.core.schema import MetadataMode

from src.ingestion.loader import load_repository_with_report
from src.ingestion.rules import DEFAULT_MAX_FILE_BYTES

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"

INCLUDED = {
    ".github/workflows/ci.yml": "yaml",
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "README.md": "markdown",
    "docker/Dockerfile": "dockerfile",
    "docs/a.md": "markdown",
    "docs/decisions/0001-adr.md": "markdown",
    "dvc.yaml": "yaml",
    "main.py": "python",
    "pipelines/p.py": "python",
    "pyproject.toml": "toml",
    "scripts/s.py": "python",
    "src/__init__.py": "python",
    "src/a.py": "python",
    "src/crlf_bom.py": "python",
    "tests/test_a.py": "python",
}
DISCARDED_FILES = {
    "app.log": "gitignore",
    "data/raw/ignored.py": "gitignore",  # nested .gitignore
    "uv.lock": "lockfile",
    "dvc.lock": "lockfile",
    "pnpm-lock.yaml": "lockfile",
    ".env": "secret_name",
    "id_rsa": "secret_name",
    "notebooks/n.ipynb": "unmapped_type",
    "data/raw/x.csv": "unmapped_type",
    "docs/img/a.png": "unmapped_type",
    "docker/.env.example": "unmapped_type",
    ".gitignore": "unmapped_type",
    "data/raw/.gitignore": "unmapped_type",
    "src/disguised.py": "binary",  # NUL in the first KiB (probe)
    "src/late_nul.py": "binary",  # NUL after the first KiB (decoded text)
    "src/link.py": "symlink",
    "dir_link": "symlink",
    "outside_link.py": "symlink",
    "src/pipe.py": "not_regular_file",
    "src/huge.py": "too_large",
    "src/latin1.py": "undecodable",
    "src/empty.py": "empty",
    "src/blank.py": "empty",
    "src/dup_of_a.py": "duplicate",
    "z_copy/a.py": "duplicate",
}
PRUNED_DIRS = {
    ".git": "builtin_pattern",
    "__pycache__": "builtin_pattern",
    "deploy/model": "corpus_rules",
    "node_modules": "builtin_pattern",
    "project_context": "corpus_rules",
    "src/__pycache__": "builtin_pattern",
}


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_repo(base: Path) -> Path:
    repo = base / "repo"
    for index, rel in enumerate(INCLUDED):
        _write(repo, rel, f"# invented file {index}\nvalue = {index}\n".encode())
    _write(repo, "src/crlf_bom.py", b"\xef\xbb\xbfdef f():\r\n    return 1\r\n")
    _write(repo, ".gitignore", b"*.log\n")
    _write(repo, "app.log", b"log line\n")
    _write(repo, "data/raw/.gitignore", b"/ignored.py\n")
    _write(repo, "data/raw/ignored.py", b"hidden = True\n")
    _write(repo, "data/raw/x.csv", b"a,b\n")
    for rel in ("uv.lock", "dvc.lock", "pnpm-lock.yaml", ".env", "id_rsa", "notebooks/n.ipynb"):
        _write(repo, rel, b"invented\n")
    _write(repo, "docs/img/a.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    _write(repo, "docker/.env.example", b"KEY=\n")
    _write(repo, "src/disguised.py", b"code\x00binary")
    _write(repo, "src/late_nul.py", b"x = 1\n" * 400 + b"\x00\n")
    (repo / "src" / "link.py").symlink_to(repo / "src" / "a.py")
    (repo / "dir_link").symlink_to(repo / "src", target_is_directory=True)
    _write(base, "outside/secret.py", b"token = 'invented'\n")
    (repo / "outside_link.py").symlink_to(base / "outside" / "secret.py")
    os.mkfifo(repo / "src" / "pipe.py")
    _write(repo, "src/huge.py", b"#" * (DEFAULT_MAX_FILE_BYTES + 1))
    _write(repo, "src/latin1.py", b"caf\xe9 = 1\n")
    _write(repo, "src/empty.py", b"")
    _write(repo, "src/blank.py", b"  \n\t\n")
    a_py = (repo / "src" / "a.py").read_bytes()
    _write(repo, "src/dup_of_a.py", a_py)
    _write(repo, "z_copy/a.py", a_py)
    for rel in ("deploy/model/m.py", "project_context/c.md", "node_modules/x/index.js"):
        _write(repo, rel, b"noise\n")
    _write(repo, "__pycache__/a.cpython-311.pyc", b"\x00pyc")
    _write(repo, "src/__pycache__/a.cpython-311.pyc", b"\x00pyc")
    _write(repo, ".git/HEAD", b"ref: refs/heads/main\n")
    return repo


def test_load_keeps_relevant_code_and_drops_every_kind_of_noise(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)

    result = load_repository_with_report(repo, rules_file=RULES_FILE)

    assert {d.doc_id: d.metadata["language"] for d in result.documents} == INCLUDED
    assert {d.rel_path: d.reason.value for d in result.discarded} == DISCARDED_FILES
    assert {d.rel_path: d.reason.value for d in result.pruned} == PRUNED_DIRS
    report = result.report
    assert dict(report.discarded_files) == {
        "binary": 2,
        "duplicate": 2,
        "empty": 2,
        "gitignore": 2,
        "lockfile": 3,
        "not_regular_file": 1,
        "secret_name": 2,
        "symlink": 3,
        "too_large": 1,
        "undecodable": 1,
        "unmapped_type": 6,
    }
    assert dict(report.pruned_dirs) == {"builtin_pattern": 4, "corpus_rules": 2}
    assert report.included_files == len(INCLUDED)
    assert report.gitignores_read == 2  # root and data/raw; none under pruned directories


def test_loaded_documents_honour_the_text_and_metadata_contract(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)

    documents = {
        d.doc_id: d for d in load_repository_with_report(repo, rules_file=RULES_FILE).documents
    }

    crlf = documents["src/crlf_bom.py"]
    assert crlf.text == "def f():\n    return 1\n"
    assert crlf.metadata["size_bytes"] == (repo / "src" / "crlf_bom.py").stat().st_size
    assert crlf.metadata["num_lines"] == 2
    assert [d.doc_id for d in documents.values()] == sorted(INCLUDED)
    for doc in documents.values():
        seen = doc.get_content(metadata_mode=MetadataMode.EMBED)
        assert seen.startswith(f"file_path: {doc.doc_id}\nlanguage: {doc.metadata['language']}")
        assert "content_hash" not in seen
