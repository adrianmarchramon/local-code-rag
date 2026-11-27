"""Phase 2 key test (decision P6): the frozen corpus chunks exactly as the manifest says.

Runs only against the frozen corpus at ``REPO_PATH``; skips, with the reason, when it is
missing or its HEAD is not the frozen commit (a modified tree at that commit fails). It is
read-only, runs under its own ``NetworkGuard`` and snapshots the tree before and after.

Content-free by construction: every assertion compares records, paths, counts or digests
(tests/chunk_validation.py, tests/chunk_manifest.py), never corpus text, symbols or
headings, so a failure cannot print corpus content. Do not run it with ``-vv``, ``-l`` or
``--showlocals``. The owner's review of the chunks themselves is the other half of P6.

Run it with ``make test-corpus``. Figures: docs/decisions.md section 12.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from chunk_manifest import (
    MANIFEST_PATH,
    build_manifest,
    compare_manifests,
    format_manifest,
    parse_manifest,
)
from chunk_validation import Finding, Snapshot, split_definitions, tree_snapshot, validate
from corpus_manifest import CORPUS_URL, FROZEN_CORPUS_SHA, corpus_skip_reason
from corpus_manifest import MANIFEST_PATH as LOAD_MANIFEST_PATH
from corpus_manifest import parse_manifest as parse_load_manifest
from llama_index.core import Document

from src import config
from src.ingestion.chunker import ChunkingRules, ChunkResult, chunk_documents_with_report
from src.ingestion.loader import load_repository
from src.network_guard import NetworkGuard, NetworkViolation

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES = ChunkingRules()


@dataclass(frozen=True, slots=True)
class ChunkRun:
    """One guarded, read-only chunking of the corpus, shared by this module's tests."""

    root: Path
    documents: tuple[Document, ...]
    result: ChunkResult
    second: ChunkResult
    findings: tuple[Finding, ...]
    manifest: dict[str, object]
    before: Snapshot
    after: Snapshot
    violations: tuple[NetworkViolation, ...]


@pytest.fixture(scope="module")
def run() -> ChunkRun:
    repo = config.REPO_PATH
    reason = corpus_skip_reason(repo)
    if reason is not None:
        pytest.skip(reason)
    before = tree_snapshot(repo)
    # A module fixture runs before the per-test autouse guard, so it brings its own.
    with NetworkGuard() as guard:
        documents = tuple(load_repository(repo))
        result = chunk_documents_with_report(documents, rules=RULES)
        second = chunk_documents_with_report(documents, rules=RULES)
        findings = tuple(validate(documents, result, RULES.max_chunk_chars))
        manifest = build_manifest(FROZEN_CORPUS_SHA, CORPUS_URL, result, RULES)
    return ChunkRun(
        repo,
        documents,
        result,
        second,
        findings,
        manifest,
        before,
        tree_snapshot(repo),
        guard.violations,
    )


def test_the_run_matches_the_versioned_manifest_exactly(run: ChunkRun) -> None:
    problems = compare_manifests(parse_manifest(MANIFEST_PATH.read_bytes()), run.manifest)

    assert problems == [], "\n".join(problems)
    assert MANIFEST_PATH.read_text(encoding="utf-8") == format_manifest(run.manifest)


def test_documents_are_the_phase1_manifest_files(run: ChunkRun) -> None:
    rows = parse_load_manifest(LOAD_MANIFEST_PATH.read_bytes()).rows

    assert [(d.doc_id, d.metadata["language"]) for d in run.documents] == [
        (row.path, row.language) for row in rows
    ]


def test_every_node_passes_the_content_free_validation(run: ChunkRun) -> None:
    # Strategy invariants, exact slices and offsets, key order and types, visibility,
    # relationship, recomputed IDs, line oracle, round trip, unique IDs and hashes, order.
    assert run.findings == ()


def test_only_python_definitions_that_do_not_fit_are_split(run: ChunkRun) -> None:
    counts = split_definitions(run.documents, run.result, RULES.max_chunk_chars)

    assert (counts.total, counts.over_cap, counts.split, counts.split_that_fit) == (188, 33, 33, 0)


def test_no_file_degraded_and_figures_match_the_handoff(run: ChunkRun) -> None:
    report = run.result.report

    assert report.degraded_files == ()
    assert report.syntax_error_files == ()
    assert (report.documents, report.nodes, report.forced) == (88, 658, 176)
    assert {k: t.nodes for k, t in report.by_strategy.items()} == {
        "config": 39,
        "markdown": 340,
        "syntactic": 279,
    }


def test_two_runs_give_identical_nodes(run: ChunkRun) -> None:
    first = [(n.node_id, n.hash) for n in run.result.nodes]
    second = [(n.node_id, n.hash) for n in run.second.nodes]

    assert first == second
    assert run.result.report == run.second.report


def test_another_process_with_another_hash_seed_gives_the_same_digest(
    run: ChunkRun, tmp_path: Path
) -> None:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "PYTHONHASHSEED": "12345"}

    result = subprocess.run(  # noqa: S603 (this interpreter, constant arguments)
        [sys.executable, "-m", "tests.chunk_manifest", "--digest", "--repo", str(run.root)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    digest = run.manifest["digest"]
    assert result.returncode == 0, "the digest subprocess failed (see its exit code)"
    assert isinstance(digest, dict)
    assert result.stdout.strip() == digest["corpus"]


def test_the_corpus_is_unchanged_and_nothing_left_the_machine(run: ChunkRun) -> None:
    assert run.violations == ()
    assert run.after == run.before


def test_the_src_digest_from_nodes_equals_the_versioned_manifest(run: ChunkRun) -> None:
    # Phase 3 (decision P6): the index manifest uses this function on the same nodes.
    from src.ingestion.digest import nodes_digest

    digest = run.manifest["digest"]
    assert isinstance(digest, dict)
    assert nodes_digest(run.result.nodes) == digest["corpus"]
