"""Ignore policy (Phase 1, Step 2): decide whether a repository path is noise by its path.

Three sources, all in gitignore syntax, combined by union (a path is ignored if any of
them excludes it), checked in this order so the first one that excludes gives the reason:

1. built-in generic noise patterns from :class:`IngestionRules` -> ``builtin_pattern``;
2. the versioned corpus rules file (``INGEST_IGNORE_FILE``) -> ``corpus_rules``;
3. the corpus's own ``.gitignore`` files, root and nested -> ``gitignore``.

``!`` re-inclusion works inside the rules file and inside the ``.gitignore`` chain, never
across sources, so no source can undo another one's exclusion.

Git semantics (decision Q8). A path ``c1/.../cn`` is evaluated one directory level at a
time: each prefix as a directory, then the leaf. At each level the ``.gitignore`` files of
its ancestors are consulted from the deepest to the root, and the deepest file with a
matching pattern decides (inside a file, the last matching pattern wins). Once a directory
is excluded, everything below it is excluded: its ``.gitignore`` files are never read and
nothing below it can be re-included. ``pathspec.GitIgnoreSpec`` does the pattern matching;
the level-by-level walk is needed because, queried with a full path, it re-includes
``build/keep.py`` for ``build/`` + ``!build/keep.py``, which Git does not.

Not modelled, on purpose: ``.git/info/exclude`` and the global ``core.excludesFile`` (not
part of the repository), ``.gitattributes``, case-insensitive matching.

The corpus and its ``.gitignore`` files are untrusted data. They are only read, never
executed; symlinks are never followed (neither the file nor its parent directories), each
file has a size cap, and an unusable ``.gitignore`` stops ingestion (decision D3).
"""

from __future__ import annotations

import errno
import logging
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pathspec import GitIgnoreSpec
from pathspec.patterns.gitignore import GitIgnorePatternError
from pathspec.patterns.gitignore.spec import GitIgnoreSpecPattern

from src.errors import IgnoreRulesError
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules

logger = logging.getLogger(__name__)

GITIGNORE_NAME: Final = ".gitignore"
BUILTIN_SOURCE: Final = "builtin"
RULES_FILE_SOURCE: Final = "rules_file"

_O_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK: Final = getattr(os, "O_NONBLOCK", 0)
_O_CLOEXEC: Final = getattr(os, "O_CLOEXEC", 0)
_READ_CHUNK: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class IgnoreVerdict:
    """Outcome of :meth:`IgnorePolicy.is_ignored`.

    ``reason`` is None when the path is kept. ``source`` names what excluded it:
    ``"builtin"``, ``"rules_file"``, or the corpus-relative path of the deciding
    ``.gitignore``. It is meant for diagnostics at DEBUG level.
    """

    reason: DiscardReason | None
    source: str | None = None

    @property
    def ignored(self) -> bool:
        return self.reason is not None


KEPT: Final = IgnoreVerdict(reason=None)


def _split_relative(rel_path: str) -> tuple[str, ...]:
    """Validate a POSIX path relative to the repository root and return its components."""
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("Expected a non-empty POSIX path relative to the repository root.")
    if rel_path.startswith("/") or "\\" in rel_path or "\0" in rel_path:
        raise ValueError(
            f"Invalid repository-relative path {rel_path!r}: it must not be absolute or "
            "contain backslashes or NUL characters."
        )
    parts = tuple(rel_path.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"Invalid repository-relative path {rel_path!r}: empty, '.' and '..' components "
            "are not allowed."
        )
    return parts


def _read_bounded(path: Path, *, limit: int, follow_symlinks: bool, label: str) -> bytes:
    """Read at most ``limit`` bytes of a regular file; FileNotFoundError means absent."""
    flags = os.O_RDONLY | _O_NONBLOCK | _O_CLOEXEC | (0 if follow_symlinks else _O_NOFOLLOW)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise IgnoreRulesError(
                f"{label} is a symbolic link; ignore files are never followed outside the "
                "repository. Replace the link with a regular file."
            ) from exc
        raise IgnoreRulesError(
            f"{label} cannot be read ({exc.strerror}). Check its permissions; ingestion "
            "stops rather than index files the project may have meant to hide."
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise IgnoreRulesError(f"{label} is not a regular file; replace or remove it.")
        if info.st_size > limit:
            raise IgnoreRulesError(
                f"{label} has {info.st_size} bytes, above the {limit}-byte cap for ignore "
                "files. Shorten it or raise IngestionRules.max_gitignore_bytes."
            )
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(_READ_CHUNK, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError as exc:
        raise IgnoreRulesError(f"{label} cannot be read ({exc.strerror}).") from exc
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > limit:
        raise IgnoreRulesError(f"{label} grew beyond the {limit}-byte cap while being read.")
    return data


def _decode(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8-sig")  # a UTF-8 byte order mark is tolerated, like Git
    except UnicodeDecodeError as exc:
        raise IgnoreRulesError(
            f"{label} is not valid UTF-8 (byte {exc.start}). Re-save it as UTF-8."
        ) from exc


def _compile(lines: Iterable[str], *, label: str, strict: bool) -> tuple[GitIgnoreSpec, int]:
    """Compile gitignore lines one by one, so one bad line cannot hide the others.

    Lines are split on ``\\n`` with a trailing ``\\r`` dropped, as Git does. An invalid
    line raises in strict mode and is skipped with a WARNING (file and line number, never
    its content) otherwise. ``pathspec`` raises ``GitIgnorePatternError`` for some invalid
    lines and ``re.error`` for invalid ranges such as ``[z-a]``.
    """
    patterns: list[GitIgnoreSpecPattern] = []
    for number, raw in enumerate(lines, start=1):
        line = raw.removesuffix("\r")
        try:
            pattern = GitIgnoreSpecPattern(line)
        except (GitIgnorePatternError, re.error) as exc:
            if strict:
                raise IgnoreRulesError(
                    f"{label}, line {number}, is not a valid gitignore pattern. Fix or "
                    "remove that line."
                ) from exc
            logger.warning(
                "ingestion.ignore.invalid_line_skipped",
                extra={"ignore_file": label, "line_number": number},
            )
            continue
        if pattern.include is not None:  # blank lines and comments compile to no-ops
            patterns.append(pattern)
    return GitIgnoreSpec(patterns), len(patterns)


class IgnorePolicy:
    """Path-based ignore decisions for one repository; see the module docstring.

    The built-in patterns and the rules file are read and compiled once, here; a missing,
    unreadable or invalid rules file raises :class:`IgnoreRulesError` (it is versioned in
    this repository, so its absence means a broken checkout). Corpus ``.gitignore`` files
    are read lazily, at most once each, and directory verdicts are cached, so the cost per
    query is amortized O(depth). The policy does no tree traversal and never imports
    ``src.config``: the caller passes the rules file (Step 4 defaults it to
    ``INGEST_IGNORE_FILE``).
    """

    def __init__(self, root: Path, rules: IngestionRules, rules_file: Path | None = None) -> None:
        self._root = Path(root)
        self._rules = rules
        self._builtin, builtin_count = _compile(
            rules.builtin_patterns, label="built-in ignore patterns", strict=True
        )
        self._rules_spec: GitIgnoreSpec | None = None
        rules_count = 0
        if rules_file is not None:
            label = f"Corpus rules file {str(rules_file)!r}"
            try:
                data = _read_bounded(
                    Path(rules_file),
                    limit=rules.max_gitignore_bytes,
                    follow_symlinks=True,
                    label=label,
                )
            except FileNotFoundError as exc:
                raise IgnoreRulesError(
                    f"{label} does not exist. It is versioned in this repository "
                    "(config/ingest.ignore): restore it, or point INGEST_IGNORE_FILE to an "
                    "existing file."
                ) from exc
            text = _decode(data, label)
            self._rules_spec, rules_count = _compile(text.split("\n"), label=label, strict=True)
        self._gitignores: dict[tuple[str, ...], GitIgnoreSpec | None] = {}
        self._verdicts: dict[tuple[str, ...], IgnoreVerdict] = {}
        self._loaded: list[str] = []
        logger.debug(
            "ingestion.ignore.policy_ready",
            extra={"builtin_patterns": builtin_count, "rules_file_patterns": rules_count},
        )

    @property
    def loaded_gitignores(self) -> tuple[str, ...]:
        """Corpus-relative paths of the ``.gitignore`` files read so far, in read order."""
        return tuple(self._loaded)

    def is_ignored(self, rel_path: str, *, is_dir: bool) -> IgnoreVerdict:
        """Return the verdict for a POSIX path relative to the root.

        ``is_dir`` says whether the path is a directory (directory-only patterns such as
        ``build/`` match directories only). Raises ValueError for absolute paths, ``..``,
        empty components or backslashes, and :class:`IgnoreRulesError` if a ``.gitignore``
        on the way is unusable.
        """
        parts = _split_relative(rel_path)
        for depth in range(1, len(parts)):
            verdict = self._directory_verdict(parts[:depth])
            if verdict.ignored:
                return verdict
        if is_dir:
            return self._directory_verdict(parts)
        return self._evaluate(parts, is_dir=False)

    def _directory_verdict(self, parts: tuple[str, ...]) -> IgnoreVerdict:
        cached = self._verdicts.get(parts)
        if cached is None:
            cached = self._evaluate(parts, is_dir=True)
            self._verdicts[parts] = cached
        return cached

    def _evaluate(self, parts: tuple[str, ...], *, is_dir: bool) -> IgnoreVerdict:
        """Verdict for one path whose ancestors are all known to be kept."""
        suffix = "/" if is_dir else ""
        query = "/".join(parts) + suffix
        if self._builtin.check_file(query).include:
            return IgnoreVerdict(DiscardReason.BUILTIN_PATTERN, BUILTIN_SOURCE)
        if self._rules_spec is not None and self._rules_spec.check_file(query).include:
            return IgnoreVerdict(DiscardReason.CORPUS_RULES, RULES_FILE_SOURCE)
        for level in range(len(parts) - 1, -1, -1):  # deepest .gitignore first
            spec = self._gitignore(parts[:level])
            if spec is None:
                continue
            include = spec.check_file("/".join(parts[level:]) + suffix).include
            if include is None:
                continue  # no opinion here: a shallower .gitignore decides
            if include:
                return IgnoreVerdict(DiscardReason.GITIGNORE, _gitignore_label(parts[:level]))
            return KEPT  # re-included by `!` in the deepest file with an opinion
        return KEPT

    def _gitignore(self, directory: tuple[str, ...]) -> GitIgnoreSpec | None:
        if directory in self._gitignores:
            return self._gitignores[directory]
        label = _gitignore_label(directory)
        if not self._is_real_directory(directory, label):
            self._gitignores[directory] = None  # nothing on disk: no .gitignore to read
            return None
        path = self._root.joinpath(*directory, GITIGNORE_NAME)
        try:
            data = _read_bounded(
                path,
                limit=self._rules.max_gitignore_bytes,
                follow_symlinks=False,
                label=f"Corpus {label!r}",
            )
        except FileNotFoundError:
            spec = None
        else:
            text = _decode(data, f"Corpus {label!r}")
            spec, count = _compile(text.split("\n"), label=label, strict=False)
            self._loaded.append(label)
            logger.debug(
                "ingestion.ignore.gitignore_loaded",
                extra={"ignore_file": label, "patterns": count},
            )
        self._gitignores[directory] = spec
        return spec

    def _is_real_directory(self, directory: tuple[str, ...], label: str) -> bool:
        """True if every component below the root is a real directory (lstat, no follow).

        A missing component, or one that is not a directory, holds no ``.gitignore``. A
        symlinked component raises: its ``.gitignore`` could live outside the repository.
        """
        current = self._root
        for part in directory:
            current = current / part
            try:
                mode = os.lstat(current).st_mode
            except (FileNotFoundError, NotADirectoryError):
                return False
            except OSError as exc:
                raise IgnoreRulesError(
                    f"Cannot inspect the directory of corpus {label!r} ({exc.strerror})."
                ) from exc
            if stat.S_ISLNK(mode):
                raise IgnoreRulesError(
                    f"Corpus {label!r} lies under a symbolic link; symlinks are never "
                    "followed. Query paths only below real directories."
                )
            if not stat.S_ISDIR(mode):
                return False
        return True


def _gitignore_label(directory: tuple[str, ...]) -> str:
    return "/".join((*directory, GITIGNORE_NAME))
