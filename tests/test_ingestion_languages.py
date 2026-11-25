"""Step 6: multi-language handling, end to end through the public loader (invented content).

A synthetic polyglot repository holds one file per language label (and per variant of the
map: upper-case suffix, ``.jsx``, ``.tsx``, ``.h``, ``.yml``, ``Dockerfile`` and
``Makefile`` by name) plus the noise each ecosystem brings. ``tests/test_ingestion_rules.py``
pins the map itself; here the point is that every label reaches a ``Document`` and every
piece of noise leaves with its reason.

A failure in the label tests means the language map changed: follow docs/decisions.md 9.3,
"How to add a language".
"""

from pathlib import Path
from typing import Final

from src.ingestion.loader import LoadResult, load_repository_with_report
from src.ingestion.rules import IngestionRules

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"
HOW_TO = "the language map changed: follow docs/decisions.md 9.3, 'How to add a language'"

# rag_phase2.md Step 3, SYNTAX_AWARE_LANGUAGES (quoted, not imported: Phase 2 does not exist).
PHASE2_SYNTAX_AWARE: Final = frozenset(
    {"python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"}
)

INCLUDED: Final = {
    ".github/workflows/ci.yml": "yaml",
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "README.md": "markdown",
    "backend/LEGACY.PY": "python",  # the last suffix is lower-cased before the lookup
    "backend/app.py": "python",
    "config/settings.yaml": "yaml",
    "dotnet/Program.cs": "csharp",
    "frontend/dist/bundle.js": "javascript",  # nested dist/ is legitimate (D2: root only)
    "frontend/src/App.jsx": "javascript",
    "frontend/src/View.tsx": "typescript",
    "frontend/src/index.js": "javascript",
    "frontend/src/main.ts": "typescript",
    "frontend/vendor.min.js": "javascript",  # no line-length heuristic (known limitation)
    "native/engine.cpp": "cpp",
    "native/util.c": "c",
    "native/util.h": "c",
    "pyproject.toml": "toml",
    "scripts/task.rb": "ruby",
    "services/Main.java": "java",
    "services/lib.rs": "rust",
    "services/main.go": "go",
    "web/index.php": "php",
}
DISCARDED: Final = {
    ".env": "secret_name",
    ".gitignore": "unmapped_type",
    "Cargo.lock": "lockfile",
    "composer.lock": "lockfile",
    "data/sample.json": "unmapped_type",
    "docker/Dockerfile.dev": "unmapped_type",  # exact file names only (known limitation)
    "frontend/logo.png": "unmapped_type",
    "frontend/src/api.generated.ts": "gitignore",
    "frontend/yarn.lock": "lockfile",
    "package-lock.json": "lockfile",
    "tools/a/dockerfile": "unmapped_type",  # file names are case-sensitive
    "tools/b/makefile": "unmapped_type",
}
PRUNED: Final = {
    ".venv": "builtin_pattern",
    "backend/__pycache__": "builtin_pattern",
    "dist": "builtin_pattern",  # root dist/ is a build output (D2)
    "frontend/node_modules": "builtin_pattern",
    "target": "builtin_pattern",
}
NOISE_UNDER_PRUNED: Final = (
    ".venv/lib/python3.11/site-packages/pkg/__init__.py",
    "backend/__pycache__/app.cpython-311.pyc",
    "dist/root.js",
    "frontend/node_modules/react/index.js",
    "target/debug/app.rs",
)


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_polyglot_repo(root: Path) -> Path:
    # Distinct invented content per file, so deduplication never interferes.
    for rel in (*INCLUDED, *DISCARDED, *NOISE_UNDER_PRUNED):
        _write(root, rel, f"invented content of {rel}\n".encode())
    _write(root, ".gitignore", b"*.generated.ts\n")
    _write(root, "frontend/logo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    _write(root, "frontend/vendor.min.js", b"var a=1;" * 4096)  # one 32 KiB line, no newline
    return root


def _load(root: Path) -> LoadResult:
    return load_repository_with_report(root, rules_file=RULES_FILE)


def test_every_label_and_variant_reaches_a_document(tmp_path: Path) -> None:
    result = _load(_make_polyglot_repo(tmp_path))

    assert {d.doc_id: d.metadata["language"] for d in result.documents} == INCLUDED, HOW_TO
    assert set(result.report.by_language) == IngestionRules().languages, HOW_TO
    assert len(result.report.by_language) == 16, HOW_TO


def test_ecosystem_noise_leaves_with_its_reason(tmp_path: Path) -> None:
    result = _load(_make_polyglot_repo(tmp_path))

    assert {d.rel_path: d.reason.value for d in result.discarded} == DISCARDED
    assert {d.rel_path: d.reason.value for d in result.pruned} == PRUNED
    assert result.report.duplicates == ()


def test_emitted_labels_are_lowercase_plain_strings_from_the_rules(tmp_path: Path) -> None:
    labels = [d.metadata["language"] for d in _load(_make_polyglot_repo(tmp_path)).documents]

    assert all(type(label) is str and label == label.lower() for label in labels)
    assert set(labels) <= IngestionRules().languages, HOW_TO


def test_phase2_syntax_aware_languages_are_all_emitted(tmp_path: Path) -> None:
    labels = {d.metadata["language"] for d in _load(_make_polyglot_repo(tmp_path)).documents}

    assert labels >= PHASE2_SYNTAX_AWARE, HOW_TO


def test_minified_one_line_file_is_loaded_whole(tmp_path: Path) -> None:
    # Characterization of a known limitation, not a requirement: Phase 2 receives it as is.
    documents = {d.doc_id: d for d in _load(_make_polyglot_repo(tmp_path)).documents}

    minified = documents["frontend/vendor.min.js"]
    assert minified.metadata["num_lines"] == 1
    assert minified.metadata["size_bytes"] == len(minified.text) == 8 * 4096
