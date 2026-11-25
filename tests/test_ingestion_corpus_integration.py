"""Phase 1 key test (decision Q7): the real corpus gives exactly the relevant files.

Runs only against the frozen corpus at ``REPO_PATH`` and skips, with the reason, when it
is missing or its HEAD is not the frozen commit; a modified tree at the frozen commit is
not a reason to skip, it fails with a diff. Read-only: HEAD comes from ``.git`` in pure
Python (no ``git`` process), nothing is written, and the tree is snapshotted before and
after. Its logic is tested without the corpus in ``tests/test_corpus_manifest.py``.

Run it with ``make test-corpus``. Expected figures: docs/decisions.md section 9.6.
"""

import fnmatch
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from corpus_manifest import (
    FROZEN_CORPUS_SHA,
    MANIFEST_PATH,
    REGENERATE_COMMAND,
    Manifest,
    compare_rows,
    corpus_skip_reason,
    parse_manifest,
    read_head,
    rows_from_documents,
)
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from src import config
from src.ingestion.loader import LoadResult, load_repository, load_repository_with_report
from src.ingestion.report import Tally
from src.network_guard import NetworkGuard, NetworkViolation

pytestmark = pytest.mark.integration

# Literal expectations (decision log, HANDOFF and docs/decisions.md 9.6), not read from src/.
BY_LANGUAGE: Final = {
    "dockerfile": Tally(1, 4_964),
    "makefile": Tally(1, 2_647),
    "markdown": Tally(46, 263_570),
    "python": Tally(32, 192_670),
    "toml": Tally(1, 3_797),
    "yaml": Tally(7, 24_349),
}
BY_TOP_LEVEL: Final = {
    ".": Tally(8, 44_348),
    ".github": Tally(2, 8_162),
    "docker": Tally(2, 8_344),
    "docs": Tally(45, 238_798),
    "pipelines": Tally(4, 11_301),
    "scripts": Tally(2, 11_906),
    "src": Tally(19, 121_650),
    "tests": Tally(6, 47_488),
}
# Every tracked file outside the pruned directories that is not in the manifest.
TRACKED_DISCARDS: Final = {
    ".dockerignore": "unmapped_type",
    ".dvc/.gitignore": "unmapped_type",
    ".dvc/config": "unmapped_type",
    ".dvcignore": "unmapped_type",
    ".gitignore": "unmapped_type",
    ".python-version": "unmapped_type",
    "LICENSE": "unmapped_type",
    "data/monitoring/.gitignore": "unmapped_type",
    "data/processed/.gitkeep": "unmapped_type",
    "data/raw/.gitignore": "unmapped_type",
    "data/raw/.gitkeep": "unmapped_type",
    "data/raw/creditcard.csv.dvc": "unmapped_type",
    "docker/.env.example": "unmapped_type",
    "docs/images/.gitkeep": "unmapped_type",
    "docs/images/architecture.png": "unmapped_type",
    "docs/videos/intro.mp4": "unmapped_type",
    "docs/whitepaper.pdf": "unmapped_type",
    "dvc.lock": "lockfile",
    "notebooks/01_exploration.ipynb": "unmapped_type",
    "reports/metrics.json": "unmapped_type",
    "reports/validation.json": "unmapped_type",
    "uv.lock": "lockfile",
}
PRUNED_AT_LEAST: Final = {
    ".git": "builtin_pattern",
    "deploy/model": "corpus_rules",
    "project_context": "corpus_rules",
}
NOISE_DIRECTORY_NAMES: Final = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "site-packages",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".next",
    }
)
NOISE_ROOT_DIRECTORIES: Final = frozenset(
    {"build", "dist", "target", "vendor", "deploy/model", "project_context"}
)
LOCKFILE_NAMES: Final = frozenset(
    {
        "uv.lock",
        "dvc.lock",
        "poetry.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
    }
)
SECRET_GLOBS: Final = (".env", ".env.*", "*.pem", "*.key", "id_rsa*", "*credentials*", "*secret*")
MAPPED_SUFFIXES: Final = frozenset(
    {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".c", ".h", ".cpp"}
    | {".cs", ".php", ".md", ".yaml", ".yml", ".toml"}
)
MAPPED_NAMES: Final = frozenset({"Dockerfile", "Makefile"})
METADATA_KEYS: Final = {"file_path", "language", "size_bytes", "num_lines", "content_hash"} | {
    "directory"
}

Snapshot = dict[str, tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class CorpusRun:
    """One guarded, read-only pass over the corpus, shared by the tests of this module."""

    root: Path
    manifest: Manifest
    result: LoadResult
    second_load: list[Document]
    before: Snapshot
    after: Snapshot
    violations: tuple[NetworkViolation, ...]


def _snapshot(root: Path) -> Snapshot:
    """(mode, size, mtime_ns) of every entry below ``root``, symlinks not followed."""
    entries: Snapshot = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as scan:
            for entry in scan:
                info = entry.stat(follow_symlinks=False)
                entries[os.path.relpath(entry.path, root)] = (
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                )
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
    return entries


@pytest.fixture(scope="module")
def corpus() -> CorpusRun:
    repo = config.REPO_PATH
    reason = corpus_skip_reason(repo)
    if reason is not None:
        pytest.skip(reason)
    manifest = parse_manifest(MANIFEST_PATH.read_bytes())
    before = _snapshot(repo)
    # A module fixture runs before the per-test autouse guard, so it brings its own.
    with NetworkGuard() as guard:
        result = load_repository_with_report(repo)
        second_load = load_repository(repo)
    return CorpusRun(repo, manifest, result, second_load, before, _snapshot(repo), guard.violations)


def test_corpus_and_manifest_are_the_frozen_commit(corpus: CorpusRun) -> None:
    assert read_head(corpus.root) == FROZEN_CORPUS_SHA
    assert corpus.manifest.commit == FROZEN_CORPUS_SHA


def test_included_files_match_the_manifest_exactly(corpus: CorpusRun) -> None:
    problems = compare_rows(corpus.manifest.rows, rows_from_documents(corpus.result.documents))

    assert problems == [], (
        "the load differs from tests/fixtures/corpus_manifest.tsv:\n  "
        + "\n  ".join(problems)
        + "\nA modified corpus must be restored to the frozen commit. If the rules or the map "
        f"changed by an owner decision, regenerate with `{REGENERATE_COMMAND}` and re-run "
        "the independent checks of docs/decisions.md, Appendix P3."
    )


def test_totals_by_language(corpus: CorpusRun) -> None:
    report = corpus.result.report

    assert (report.included_files, report.included_bytes) == (88, 491_997)
    assert dict(report.by_language) == BY_LANGUAGE


def test_totals_by_top_level_directory(corpus: CorpusRun) -> None:
    assert dict(corpus.result.report.by_top_level) == BY_TOP_LEVEL


def test_no_noise_is_included(corpus: CorpusRun) -> None:
    for doc in corpus.result.documents:
        path = doc.doc_id
        parts = path.split("/")
        name = parts[-1]
        assert not NOISE_DIRECTORY_NAMES & set(parts[:-1]), path
        assert not any(path.startswith(f"{noise}/") for noise in NOISE_ROOT_DIRECTORIES), path
        assert name not in LOCKFILE_NAMES, path
        assert not any(fnmatch.fnmatchcase(name.lower(), glob) for glob in SECRET_GLOBS), path
        assert name in MAPPED_NAMES or Path(name).suffix.lower() in MAPPED_SUFFIXES, path


def test_discards_are_the_tracked_unmapped_files_and_lockfiles(corpus: CorpusRun) -> None:
    result = corpus.result
    tracked_kinds = {"unmapped_type", "lockfile"}

    assert {
        d.rel_path: d.reason.value for d in result.discarded if d.reason.value in tracked_kinds
    } == TRACKED_DISCARDS
    assert PRUNED_AT_LEAST.items() <= {d.rel_path: d.reason.value for d in result.pruned}.items()
    assert result.report.duplicates == ()
    assert result.report.gitignores_read == 4


def test_every_document_honours_the_metadata_contract(corpus: CorpusRun) -> None:
    ids = [doc.doc_id for doc in corpus.result.documents]
    assert len(set(ids)) == len(ids) == 88

    for doc in corpus.result.documents:
        metadata = doc.metadata
        assert set(metadata) == METADATA_KEYS
        assert doc.doc_id == metadata["file_path"]
        assert metadata["directory"] == doc.doc_id.rpartition("/")[0]
        assert not {"\0", "﻿", "\r"} & set(doc.text), doc.doc_id
        visible = doc.get_content(metadata_mode=MetadataMode.EMBED)
        assert visible.startswith(f"file_path: {doc.doc_id}\nlanguage: {metadata['language']}\n\n")
        assert set(metadata) - set(doc.excluded_embed_metadata_keys) == {"file_path", "language"}
        assert set(metadata) - set(doc.excluded_llm_metadata_keys) == {"file_path", "language"}


def test_a_second_load_is_identical(corpus: CorpusRun) -> None:
    first = [(d.doc_id, d.text, dict(d.metadata)) for d in corpus.result.documents]
    second = [(d.doc_id, d.text, dict(d.metadata)) for d in corpus.second_load]

    assert first == second


def test_corpus_is_unchanged_and_nothing_left_the_machine(corpus: CorpusRun) -> None:
    assert corpus.violations == ()
    assert corpus.after == corpus.before
