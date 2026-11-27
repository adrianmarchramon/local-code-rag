"""Versioned chunk manifest of the frozen corpus (Phase 2, Step 7; decisions P6, Gate 3 Q1).

``tests/fixtures/chunk_manifest.json`` holds the chunk report and one row per document
(path, strategy, nodes, forced spans and a digest of its nodes), plus what produced them:
corpus commit, chunking rules and package versions. No file text, symbols or headings.
The key test compares it with a fresh run by **exact** equality.

Digest (also stored in the manifest): a document's digest is the sha256 of the UTF-8
lines ``"<node id>:<node hash>"`` of its nodes, in order, joined by ``"\\n"``; the corpus
digest is the sha256 of the lines ``"<path>:<document digest>"`` in document order. A node
hash covers its text and metadata, so any changed boundary, text or metadata changes it.

Regenerate (only after an owner decision that changes the chunks) with
:data:`REGENERATE_COMMAND` from the repository root, then re-run the independent checks
of docs/decisions.md, Appendix S5. Not a test module, and not part of ``src/``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Final

from llama_index.core.schema import BaseNode

from src.errors import IngestionError
from src.ingestion.chunker import ChunkingRules, ChunkResult
from src.ingestion.digest import DIGEST_DEFINITION as DIGEST_DEFINITION
from src.ingestion.digest import corpus_digest as corpus_digest
from src.ingestion.digest import document_digest as document_digest
from src.ingestion.strategies import Strategy, strategy_for

FORMAT: Final = "local-code-rag/chunk-manifest/1"
MANIFEST_PATH: Final = Path(__file__).resolve().parent / "fixtures" / "chunk_manifest.json"
REGENERATE_COMMAND: Final = (
    "uv run --locked --offline python -m tests.chunk_manifest "
    "--out tests/fixtures/chunk_manifest.json"
)
PACKAGES: Final = ("tree-sitter", "tree-sitter-python", "llama-index-core")
TOP_KEYS: Final = (
    "format",
    "corpus",
    "regenerate",
    "rules",
    "versions",
    "digest",
    "report",
    "documents",
)
ROW_KEYS: Final = ("path", "strategy", "nodes", "forced", "digest")
_HEX64: Final = re.compile(r"[0-9a-f]{64}")


class ChunkManifestError(IngestionError):
    """The chunk manifest file breaks its format."""


def document_rows(result: ChunkResult) -> list[dict[str, object]]:
    """One row per document that produced nodes, in output (input) order."""
    rows: list[dict[str, object]] = []
    groups: dict[str, list[tuple[BaseNode, bool]]] = {}
    languages: dict[str, str] = {}
    for node, span in zip(result.nodes, result.spans, strict=True):
        path = str(node.metadata["file_path"])
        groups.setdefault(path, []).append((node, span.forced))
        languages[path] = str(node.metadata["language"])
    degraded = set(result.report.degraded_files)
    for path, pairs in groups.items():
        strategy = Strategy.TEXT if path in degraded else strategy_for(languages[path])
        rows.append(
            {
                "path": path,
                "strategy": strategy.value,
                "nodes": len(pairs),
                "forced": sum(forced for _, forced in pairs),
                "digest": document_digest([node for node, _ in pairs]),
            }
        )
    return rows


def build_manifest(
    commit: str, corpus_url: str, result: ChunkResult, rules: ChunkingRules
) -> dict[str, object]:
    """The manifest of one chunking run, with deterministic key order."""
    rows = document_rows(result)
    return {
        "format": FORMAT,
        "corpus": {"url": corpus_url, "commit": commit},
        "regenerate": REGENERATE_COMMAND,
        "rules": {"max_chunk_chars": rules.max_chunk_chars},
        "versions": {package: version(package) for package in PACKAGES},
        "digest": {"definition": DIGEST_DEFINITION, "corpus": corpus_digest(rows)},
        "report": result.report.to_dict(),
        "documents": rows,
    }


def format_manifest(manifest: dict[str, object]) -> str:
    """The canonical file text: 2-space JSON, key order as built, UTF-8, final newline."""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def _fail(message: str) -> ChunkManifestError:
    return ChunkManifestError(
        f"chunk manifest: {message}. Regenerate it with `{REGENERATE_COMMAND}` only after an "
        "owner decision, then re-run the checks of docs/decisions.md, Appendix S5."
    )


def parse_manifest(data: bytes) -> dict[str, object]:
    """Strictly read the manifest file; :class:`ChunkManifestError` on any deviation."""
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data or not data.endswith(b"\n"):
        raise _fail("must be UTF-8 without BOM, LF line endings, ending with a newline")
    try:
        text = data.decode("utf-8")
        manifest = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"not valid UTF-8 JSON ({type(error).__name__})") from error
    if not isinstance(manifest, dict) or tuple(manifest) != TOP_KEYS:
        raise _fail(f"top-level keys must be exactly {TOP_KEYS}")
    if manifest["format"] != FORMAT:
        raise _fail(f"unknown format {manifest['format']!r}")
    if format_manifest(manifest) != text:
        raise _fail("not in canonical form (key order, indentation or escaping changed)")
    rows = manifest["documents"]
    if not isinstance(rows, list) or not rows:
        raise _fail("documents must be a non-empty list")
    paths: list[str] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or tuple(row) != ROW_KEYS:
            raise _fail(f"document row {position} must have exactly the keys {ROW_KEYS}")
        if not isinstance(row["path"], str) or row["strategy"] not in {s.value for s in Strategy}:
            raise _fail(f"document row {position} has an invalid path or strategy")
        if any(type(row[key]) is not int or row[key] < 0 for key in ("nodes", "forced")):
            raise _fail(f"document row {position} has invalid counts")
        if not isinstance(row["digest"], str) or not _HEX64.fullmatch(row["digest"]):
            raise _fail(f"document row {position} has an invalid digest")
        paths.append(row["path"])
    if len(set(paths)) != len(paths):
        raise _fail("a document path appears twice")
    digest = manifest["digest"]
    if not isinstance(digest, dict) or digest.get("corpus") != corpus_digest(rows):
        raise _fail("the corpus digest does not match the document rows")
    return manifest


def compare_manifests(expected: dict[str, object], actual: dict[str, object]) -> list[str]:
    """Every difference, by field and by path (counts, digests, versions: no content)."""
    problems = [
        f"{key} differs: manifest {expected.get(key)!r}, run {actual.get(key)!r}"
        for key in ("format", "corpus", "rules", "versions", "digest")
        if expected.get(key) != actual.get(key)
    ]
    want_report, got_report = expected.get("report"), actual.get("report")
    if isinstance(want_report, dict) and isinstance(got_report, dict):
        problems += [
            f"report.{key} differs: manifest {want_report.get(key)!r}, run {got_report.get(key)!r}"
            for key in sorted(set(want_report) | set(got_report))
            if want_report.get(key) != got_report.get(key)
        ]
    want_rows = _rows_by_path(expected.get("documents"))
    got_rows = _rows_by_path(actual.get("documents"))
    problems += [
        f"missing from the run: {path}" for path in sorted(want_rows.keys() - got_rows.keys())
    ]
    problems += [
        f"not in the manifest: {path}" for path in sorted(got_rows.keys() - want_rows.keys())
    ]
    for path in sorted(want_rows.keys() & got_rows.keys()):
        want, got = want_rows[path], got_rows[path]
        problems += [
            f"{path}: {field} differs (manifest {want[field]!r}, run {got[field]!r})"
            for field in ROW_KEYS[1:]
            if want[field] != got[field]
        ]
    if not problems and expected.get("documents") != actual.get("documents"):
        problems.append("document order differs")
    return problems


def _rows_by_path(rows: object) -> dict[str, dict[str, object]]:
    if not isinstance(rows, list):
        return {}
    return {str(row.get("path")): row for row in rows if isinstance(row, dict)}


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the manifest (or print the corpus digest) from the frozen corpus."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `python -m tests....`
    from corpus_manifest import CORPUS_URL, FROZEN_CORPUS_SHA, corpus_skip_reason

    from src.config import REPO_PATH
    from src.ingestion.chunker import chunk_documents_with_report
    from src.ingestion.loader import load_repository
    from src.network_guard import NetworkGuard

    parser = argparse.ArgumentParser(description="Regenerate the Phase 2 chunk manifest.")
    parser.add_argument("--repo", type=Path, default=REPO_PATH, help="corpus (REPO_PATH)")
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH, help="manifest to write")
    parser.add_argument("--digest", action="store_true", help="print the corpus digest only")
    args = parser.parse_args(argv)
    with NetworkGuard() as guard:
        reason = corpus_skip_reason(args.repo)
        if reason is not None:
            sys.stderr.write(f"refusing: {reason}\n")
            return 1
        rules = ChunkingRules()
        result = chunk_documents_with_report(load_repository(args.repo), rules=rules)
        manifest = build_manifest(FROZEN_CORPUS_SHA, CORPUS_URL, result, rules)
    if guard.violations:
        sys.stderr.write(f"refusing: network access attempted {guard.violations}\n")
        return 1
    digest = manifest["digest"]
    if args.digest and isinstance(digest, dict):
        sys.stdout.write(f"{digest['corpus']}\n")
        return 0
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(format_manifest(manifest), encoding="utf-8", newline="\n")
    temporary.replace(args.out)
    sys.stderr.write(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
