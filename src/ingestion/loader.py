"""Phase 1 loader: a repository becomes a clean, deterministic list of LlamaIndex Documents.

API (the contract with Phases 2-6):

* :func:`load_repository` returns ``list[Document]``, sorted by ``file_path``, and raises
  :class:`src.errors.NoDocumentsError` when nothing passes the filters (decision D8);
* :func:`load_repository_with_report` returns the documents plus a :class:`LoadReport`
  and per-path discards, and returns an empty result instead of raising.

Both are deterministic and side-effect free: the same tree gives the same documents, IDs,
order and report, and nothing is written anywhere. Phase 6 re-runs them on every startup.

Evaluation order per path (the first reason that rejects it is recorded):

1. walker, from the directory entry only: ``invalid_name``; ``builtin_pattern``,
   ``corpus_rules``, ``gitignore`` (directories are pruned before descending);
   ``symlink``; ``not_regular_file``;
2. file name: ``lockfile``, ``secret_name``, ``unmapped_type``;
3. ``lstat``: ``symlink``, ``not_regular_file``, ``too_large`` (a file swapped since the
   walk);
4. first-KiB probe: ``binary``, ``unreadable``, ``not_regular_file``;
5. bounded read: ``too_large``, ``unreadable``, ``undecodable``, ``binary`` (NUL),
   ``empty``;
6. ``duplicate``, last and only among the survivors.

CLI: ``python -m src.ingestion.loader [--repo PATH] [--rules-file PATH] [--list | --json]``
(``make ingest``). The load runs under :class:`NetworkGuard`. Exit codes:

    0   documents loaded
    1   unexpected error (a bug; Python prints the traceback)
    2   invalid command-line arguments (argparse)
    10  configuration error (REPO_PATH, INGEST_IGNORE_FILE, ...)
    14  external network access attempted (blocked by the network guard)
    20  ingestion error: the root or the ignore rules are unusable
    21  no documents: the root is valid but nothing passed the filters
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final, TextIO

from llama_index.core import Document

from src.errors import (
    ConfigurationError,
    ExternalNetworkAccessError,
    IngestionError,
    NoDocumentsError,
)
from src.ingestion.binary import probe_head
from src.ingestion.classify import classify_name
from src.ingestion.content import FileContent, check_metadata, read_text_file
from src.ingestion.dedup import find_duplicates
from src.ingestion.documents import build_document
from src.ingestion.ignore import IgnorePolicy
from src.ingestion.reasons import Discard, DiscardReason
from src.ingestion.report import IncludedFile, LoadReport, build_report
from src.ingestion.rules import IngestionRules
from src.ingestion.walker import WalkEntry, resolve_root, walk_repository
from src.logging_config import configure_logging
from src.network_guard import NetworkGuard

# Fixed name: under `python -m src.ingestion.loader`, __name__ would be "__main__".
logger = logging.getLogger("src.ingestion.loader")

_COMPONENT: Final = "ingestion"


class ExitCode(IntEnum):
    """Process exit codes of the loader CLI (documented in the module docstring)."""

    OK = 0
    UNEXPECTED = 1
    CONFIGURATION = 10
    EXTERNAL_NETWORK = 14
    INGESTION = 20
    NO_DOCUMENTS = 21


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Documents sorted by ``file_path``, the report, and every discard with its path."""

    documents: tuple[Document, ...]
    report: LoadReport
    discarded: tuple[Discard, ...]  # files, sorted by path (duplicates included)
    pruned: tuple[Discard, ...]  # directories not descended into, sorted by path


@dataclass(frozen=True, slots=True)
class _Survivor:
    entry: WalkEntry
    language: str
    content: FileContent


def _default_rules_file() -> Path:
    # Imported lazily, so a bad environment surfaces as ConfigurationError at call time
    # (exit code 10 in the CLI) instead of at import time.
    config = importlib.import_module("src.config")
    rules_file: Path = config.INGEST_IGNORE_FILE
    return rules_file


def _evaluate(entry: WalkEntry, rules: IngestionRules) -> _Survivor | DiscardReason:
    """Steps 2-5 of the evaluation order for one candidate file: first rejection wins."""
    named = classify_name(entry.rel_path, rules)
    if named.language is None:
        return named.reason or DiscardReason.UNMAPPED_TYPE
    reason = check_metadata(entry.path, rules.max_file_bytes)
    if reason is None:
        reason = probe_head(entry.path, rules.head_bytes).discard_reason
    if reason is not None:
        return reason
    result = read_text_file(entry.path, rules.max_file_bytes)
    if result.content is None:
        return result.reason or DiscardReason.UNREADABLE
    return _Survivor(entry, named.language, result.content)


def load_repository_with_report(
    repo_path: Path,
    *,
    rules: IngestionRules | None = None,
    rules_file: Path | None = None,
) -> LoadResult:
    """Load ``repo_path`` and explain every exclusion; see the module docstring.

    ``rules`` defaults to :class:`IngestionRules` and ``rules_file`` to
    ``INGEST_IGNORE_FILE``. Raises :class:`src.errors.RepositoryRootError` or
    :class:`src.errors.IgnoreRulesError` (both :class:`IngestionError`) when the root or
    the ignore rules are unusable; per-file problems are discards, never exceptions.
    """
    started = time.perf_counter()
    rules = rules if rules is not None else IngestionRules()
    rules_file = rules_file if rules_file is not None else _default_rules_file()
    root = resolve_root(repo_path)
    policy = IgnorePolicy(root, rules, rules_file)
    walk = walk_repository(root, policy)

    discarded: list[Discard] = list(walk.discarded)
    survivors: list[_Survivor] = []
    for entry in walk.files:  # already sorted by POSIX path
        outcome = _evaluate(entry, rules)
        if isinstance(outcome, DiscardReason):
            discarded.append(Discard(entry.rel_path, outcome))
        else:
            survivors.append(outcome)

    duplicates = find_duplicates({s.entry.rel_path: s.content.content_hash for s in survivors})
    duplicated = {d.discarded for d in duplicates}
    discarded.extend(Discard(d.discarded, DiscardReason.DUPLICATE) for d in duplicates)
    kept = [s for s in survivors if s.entry.rel_path not in duplicated]
    documents = tuple(build_document(s.entry.rel_path, s.language, s.content) for s in kept)
    report = build_report(
        included=[IncludedFile(s.entry.rel_path, s.language, s.content.size_bytes) for s in kept],
        discarded=discarded,
        pruned=walk.pruned,
        duplicates=duplicates,
        gitignores_read=len(policy.loaded_gitignores),
    )
    result = LoadResult(
        documents=documents,
        report=report,
        discarded=tuple(sorted(discarded, key=lambda d: (d.rel_path, d.reason.value))),
        pruned=walk.pruned,
    )
    _log(result, started)
    return result


def load_repository(
    repo_path: Path,
    *,
    rules: IngestionRules | None = None,
    rules_file: Path | None = None,
) -> list[Document]:
    """Load ``repo_path`` as Documents sorted by ``file_path`` (Phase 1 contract).

    Raises :class:`src.errors.NoDocumentsError` if nothing passes the filters, so that a
    wrong path or over-broad rules never build an empty index silently (decision D8).
    """
    result = load_repository_with_report(repo_path, rules=rules, rules_file=rules_file)
    if not result.documents:
        raise NoDocumentsError(
            f"No documents were loaded from {str(repo_path)!r}: the directory exists, but no "
            "file passed the filters. Check that it is the right repository and that the "
            "rules file (INGEST_IGNORE_FILE) and .gitignore files do not exclude everything; "
            "`python -m src.ingestion.loader --repo <path>` shows the discard reasons."
        )
    return list(result.documents)


def _log(result: LoadResult, started: float) -> None:
    for discard in (*result.pruned, *result.discarded):
        logger.debug(
            "ingestion.discarded",
            extra={"component": _COMPONENT, "path": discard.rel_path, "reason": discard.reason},
        )
    report = result.report
    logger.info(
        "ingestion.loaded",
        extra={
            "component": _COMPONENT,
            "included_files": report.included_files,
            "included_bytes": report.included_bytes,
            "by_language": {k: t.files for k, t in report.by_language.items()},
            "discarded_files": dict(report.discarded_files),
            "pruned_dirs": dict(report.pruned_dirs),
            "duplicates": len(report.duplicates),
            "gitignores_read": report.gitignores_read,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )


# --- CLI ------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.loader",
        description="Load a repository with the Phase 1 filters and print what was kept.",
    )
    parser.add_argument("--repo", type=Path, help="repository root (default: REPO_PATH)")
    parser.add_argument(
        "--rules-file", type=Path, help="corpus rules file (default: INGEST_IGNORE_FILE)"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--list", action="store_true", help="print included paths, one per line")
    output.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _fail(out: TextIO, code: ExitCode, what: str, error: Exception) -> int:
    logger.error(
        "ingestion.failed",
        extra={"component": _COMPONENT, "exit_code": int(code), "error": str(error)},
    )
    out.write(f"[FAIL] {what}: {error}\n")
    out.write(f"RESULT: FAIL (exit {code.value} {code.name})\n")
    return int(code)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging()
    out = stdout if stdout is not None else sys.stdout
    try:
        config = importlib.import_module("src.config")
        repo: Path = args.repo if args.repo is not None else config.REPO_PATH
        rules_file: Path = (
            args.rules_file if args.rules_file is not None else config.INGEST_IGNORE_FILE
        )
    except ConfigurationError as error:
        return _fail(out, ExitCode.CONFIGURATION, "configuration", error)
    with NetworkGuard() as guard:
        try:
            result = load_repository_with_report(repo, rules_file=rules_file)
        except ExternalNetworkAccessError as error:
            return _fail(out, ExitCode.EXTERNAL_NETWORK, "network", error)
        except IngestionError as error:
            return _fail(out, ExitCode.INGESTION, "ingestion", error)
    if guard.violations:  # something swallowed the guard's exception
        first = guard.violations[0]
        blocked = ExternalNetworkAccessError(first.operation, first.destination)
        return _fail(out, ExitCode.EXTERNAL_NETWORK, "network", blocked)
    code = ExitCode.OK if result.documents else ExitCode.NO_DOCUMENTS
    if args.json:  # machine-readable output only; the exit code carries the verdict
        out.write(json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    elif args.list:
        out.write("".join(f"{doc.doc_id}\n" for doc in result.documents))
    else:
        out.write(result.report.render_text())
        verdict = "OK" if code is ExitCode.OK else "FAIL (no file passed the filters)"
        out.write(f"RESULT: {verdict} (exit {code.value} {code.name})\n")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
