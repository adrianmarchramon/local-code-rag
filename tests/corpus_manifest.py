"""Test support for the Phase 1 key test: corpus HEAD, manifest format and comparison.

Not a test module (pytest does not collect it) and not part of ``src/``. Everything here
is pure or reads only inside the given repository, so ``tests/test_corpus_manifest.py``
exercises it on synthetic data on any machine; the key test itself
(``tests/test_ingestion_corpus_integration.py``) needs the real corpus.

* :func:`read_head` resolves HEAD from ``.git`` in pure Python. The corpus is untrusted
  data: running ``git`` there would honour its ``.git/config``, so ``git`` is never run.
  Symlinks are never followed and nothing outside ``.git`` is opened.
* The manifest (``tests/fixtures/corpus_manifest.tsv``) is UTF-8 text with LF line
  endings: a ``#`` header, then one ``path<TAB>language<TAB>size_bytes<TAB>sha256`` line
  per included file, sorted by the path string (code points, like the loader). It holds
  paths, labels, sizes and hashes only, never file contents.

Regenerate it (only after an owner decision that changes the expected files) with
:data:`REGENERATE_COMMAND`, run from the repository root; docs/decisions.md, Appendix P3,
lists the independent checks that must follow.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from llama_index.core import Document

from src.errors import IngestionError
from src.ingestion.repo_head import RepositoryHeadError
from src.ingestion.repo_head import read_head as read_head

FROZEN_CORPUS_SHA: Final = "5c288080c88f0bfec435af0093a5155baaf7f343"
CORPUS_URL: Final = "https://github.com/adrianmarchramon/fraud-mlops-pipeline.git"  # public
MANIFEST_PATH: Final = Path(__file__).resolve().parent / "fixtures" / "corpus_manifest.tsv"
COLUMNS: Final = ("path", "language", "size_bytes", "sha256")
REGENERATE_COMMAND: Final = (
    "uv run --locked --offline python -m tests.corpus_manifest "
    "--out tests/fixtures/corpus_manifest.tsv"
)

_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_LABEL_RE: Final = re.compile(r"[a-z][a-z0-9_+-]*")
_SIZE_RE: Final = re.compile(r"0|[1-9][0-9]*")
_TOTALS_RE: Final = re.compile(r"([0-9]+) files, ([0-9]+) bytes")
_HEADER_RE: Final = re.compile(r"# ([a-z_]+): (.*)")


class ManifestError(IngestionError):
    """The manifest file, or the rows it is generated from, break the format."""


# --- HEAD: read by src.ingestion.repo_head (moved there by Phase 3 decision P6) ----------------
# The names stay importable from here for the Phase 1 tests.

CorpusHeadError = RepositoryHeadError


def corpus_skip_reason(repo: Path, expected_sha: str = FROZEN_CORPUS_SHA) -> str | None:
    """Why the key test cannot run on ``repo`` (Q7: missing, or not the frozen commit)."""
    if not repo.is_dir():
        return f"corpus not found: REPO_PATH={repo} is not a directory"
    try:
        head = read_head(repo)
    except CorpusHeadError as error:
        return f"cannot tell whether {repo} is the frozen corpus {expected_sha}: {error}"
    if head != expected_sha:
        return f"corpus at {repo} has HEAD {head}, not the frozen {expected_sha}"
    return None


# --- manifest rows ------------------------------------------------------------------------------


def _check_path(path: str) -> None:
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in path
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in path)
    ):
        raise ManifestError(
            f"path {path!r} cannot go in the manifest: it must be a relative POSIX path "
            "without empty, '.' or '..' parts, backslashes, tabs or control characters"
        )


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One included file: what the key test compares, field by field."""

    path: str
    language: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _check_path(self.path)
        if not _LABEL_RE.fullmatch(self.language):
            raise ManifestError(f"{self.path}: invalid language label {self.language!r}")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ManifestError(f"{self.path}: invalid size {self.size_bytes!r}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ManifestError(f"{self.path}: sha256 must be 64 lowercase hex digits")


@dataclass(frozen=True, slots=True)
class Manifest:
    """The corpus commit the manifest describes and its rows, sorted by path."""

    commit: str
    rows: tuple[ManifestRow, ...]


def rows_from_documents(documents: Iterable[Document]) -> tuple[ManifestRow, ...]:
    """Manifest rows for loaded documents, in load order."""
    return tuple(
        ManifestRow(
            path=str(doc.metadata["file_path"]),
            language=str(doc.metadata["language"]),
            size_bytes=int(doc.metadata["size_bytes"]),
            sha256=str(doc.metadata["content_hash"]),
        )
        for doc in documents
    )


def _check_rows(rows: Sequence[ManifestRow]) -> None:
    if not rows:
        raise ManifestError("the manifest has no rows")
    for previous, current in itertools.pairwise(rows):
        if current.path == previous.path:
            raise ManifestError(f"duplicate path {current.path!r}")
        if current.path < previous.path:
            raise ManifestError(
                f"rows are not sorted by path: {current.path!r} comes after {previous.path!r}"
            )


def format_manifest(commit: str, rows: Sequence[ManifestRow], *, generated: str) -> str:
    """The manifest file's text. Fails loudly on unsorted, duplicate or invalid rows."""
    if not _SHA1_RE.fullmatch(commit):
        raise ManifestError(f"commit must be 40 lowercase hex digits, got {commit!r}")
    _check_rows(rows)
    total = sum(row.size_bytes for row in rows)
    header = [
        "# Phase 1 key-test manifest: the files ingestion must load from the frozen corpus.",
        "# Paths, labels, sizes and sha256 only; never file contents. Sorted by path.",
        f"# corpus: {CORPUS_URL}",
        f"# commit: {commit}",
        f"# generated: {generated}",
        f"# command: {REGENERATE_COMMAND}",
        f"# columns: {chr(9).join(COLUMNS)}",
        f"# totals: {len(rows)} files, {total} bytes",
    ]
    lines = [f"{r.path}\t{r.language}\t{r.size_bytes}\t{r.sha256}" for r in rows]
    return "\n".join(header + lines) + "\n"


def _fail(number: int, message: str) -> ManifestError:
    return ManifestError(
        f"manifest line {number}: {message}. Fix the file, or regenerate it with "
        f"`{REGENERATE_COMMAND}` (docs/decisions.md, Appendix P3)"
    )


def _parse_row(number: int, line: str) -> ManifestRow:
    fields = line.split("\t")
    if len(fields) != len(COLUMNS):
        raise _fail(number, f"expected {len(COLUMNS)} tab-separated fields, got {len(fields)}")
    path, language, size, sha256 = fields
    if not _SIZE_RE.fullmatch(size):
        raise _fail(number, f"size_bytes {size!r} is not a decimal integer")
    try:
        return ManifestRow(path, language, int(size), sha256)
    except ManifestError as error:
        raise _fail(number, str(error)) from error


def parse_manifest(data: bytes) -> Manifest:
    """Strictly parse a manifest file's bytes; :class:`ManifestError` on any deviation."""
    if data.startswith(b"\xef\xbb\xbf"):
        raise _fail(1, "starts with a byte-order mark")
    if b"\r" in data:
        raise _fail(1, "contains a carriage return (LF line endings only)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail(1, f"is not UTF-8 ({error.reason})") from error
    if not text.endswith("\n"):
        raise _fail(text.count("\n") + 1, "the file must end with a newline")
    header: dict[str, str] = {}
    rows: list[ManifestRow] = []
    for number, line in enumerate(text[:-1].split("\n"), start=1):
        if line.startswith("#"):
            if rows:
                raise _fail(number, "comment after the first data line")
            match = _HEADER_RE.fullmatch(line)
            if match is not None:
                if match.group(1) in header:
                    raise _fail(number, f"header key {match.group(1)!r} repeated")
                header[match.group(1)] = match.group(2)
            continue
        if not line:
            raise _fail(number, "blank line")
        rows.append(_parse_row(number, line))
    for key in ("commit", "columns", "totals"):
        if key not in header:
            raise _fail(1, f"header key {key!r} is missing")
    if header["columns"] != "\t".join(COLUMNS):
        raise _fail(1, f"columns must be {COLUMNS}")
    try:
        _check_rows(rows)
    except ManifestError as error:
        raise _fail(1, str(error)) from error
    totals = _TOTALS_RE.fullmatch(header["totals"])
    size = sum(row.size_bytes for row in rows)
    if totals is None or (int(totals.group(1)), int(totals.group(2))) != (len(rows), size):
        raise _fail(1, f"totals header disagrees with the rows ({len(rows)} files, {size} bytes)")
    if not _SHA1_RE.fullmatch(header["commit"]):
        raise _fail(1, "commit must be 40 lowercase hex digits")
    return Manifest(header["commit"], tuple(rows))


def compare_rows(expected: Sequence[ManifestRow], actual: Sequence[ManifestRow]) -> list[str]:
    """Every difference between the manifest and a load, as actionable lines; [] if none."""
    want = {row.path: row for row in expected}
    got = {row.path: row for row in actual}
    problems = [f"missing from the load: {path}" for path in sorted(want.keys() - got.keys())]
    problems += [f"not in the manifest: {path}" for path in sorted(got.keys() - want.keys())]
    for path in sorted(want.keys() & got.keys()):
        for field in COLUMNS[1:]:
            wanted, loaded = getattr(want[path], field), getattr(got[path], field)
            if wanted != loaded:
                problems.append(f"{path}: {field} differs (manifest {wanted!r}, load {loaded!r})")
    if not problems and [r.path for r in expected] != [r.path for r in actual]:
        pairs = zip(expected, actual, strict=True)  # same paths, so same length
        index = next(i for i, (a, b) in enumerate(pairs) if a.path != b.path)
        problems.append(
            f"order differs at position {index}: manifest {expected[index].path!r}, "
            f"load {actual[index].path!r}"
        )
    return problems


# --- generation (manual, after an owner decision) -------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the manifest from the frozen corpus with the production loader."""
    # Imported here so that the pure helpers above stay importable without configuration.
    from src.config import REPO_PATH
    from src.ingestion.loader import load_repository_with_report
    from src.network_guard import NetworkGuard

    parser = argparse.ArgumentParser(description="Regenerate the Phase 1 corpus manifest.")
    parser.add_argument("--repo", type=Path, default=REPO_PATH, help="corpus (REPO_PATH)")
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH, help="manifest to write")
    args = parser.parse_args(argv)
    with NetworkGuard() as guard:
        reason = corpus_skip_reason(args.repo)
        if reason is not None:
            sys.stderr.write(f"refusing to generate: {reason}\n")
            return 1
        rows = rows_from_documents(load_repository_with_report(args.repo).documents)
    if guard.violations:
        sys.stderr.write(f"refusing to write: network access attempted {guard.violations}\n")
        return 1
    text = format_manifest(
        FROZEN_CORPUS_SHA, rows, generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(args.out)
    sys.stderr.write(f"wrote {len(rows)} rows to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
