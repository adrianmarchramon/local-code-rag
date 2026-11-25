"""Repository walker (Phase 1, Step 4): enumerate candidate files, pruning ignored dirs.

Per directory entry, using only what ``os.scandir`` knows without following links:

1. a name that cannot be represented safely -> ``invalid_name`` (not valid UTF-8, or it
   contains a backslash or a control character such as a newline). Checked first because
   such names cannot be matched, logged, hashed or used as IDs;
2. the ignore policy (built-in patterns, corpus rules file, ``.gitignore`` chain). An
   ignored directory is pruned *before* descending, so nothing below it is listed and its
   ``.gitignore`` files are never read. Symlinks are queried as non-directories, as Git does;
3. a symbolic link (to a file or a directory, broken or not) -> ``symlink``; never followed;
4. anything that is not a regular file or a directory (FIFO, socket, device) ->
   ``not_regular_file``; never opened.

Regular files that survive become candidates for name classification and reading.
Directories that cannot be listed are counted as ``unreadable`` without aborting the walk;
only the root itself is fatal. Discards of directories are counted per directory (their
contents are not listed). The traversal is iterative (no recursion limit), and every result
is sorted at the end by its POSIX path string, so the output never depends on the order in
which the filesystem returns entries.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from src.errors import RepositoryRootError
from src.ingestion.ignore import IgnorePolicy
from src.ingestion.reasons import Discard, DiscardReason

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkEntry:
    """A regular file that passed the path filters."""

    rel_path: str  # POSIX, relative to the root
    path: Path  # absolute, below the resolved root


@dataclass(frozen=True, slots=True)
class WalkResult:
    """Candidates and walker-level discards, each sorted by POSIX path."""

    files: tuple[WalkEntry, ...]
    discarded: tuple[Discard, ...]  # files (and symlinks, special files) left out
    pruned: tuple[Discard, ...]  # directories not descended into


class _Kind(Enum):
    DIRECTORY = auto()
    REGULAR = auto()
    SYMLINK = auto()
    OTHER = auto()


def resolve_root(repo_path: Path) -> Path:
    """Resolve the repository root once; fatal if it is missing or not a directory.

    A symlinked root is accepted (it is the caller's explicit choice) and resolved here;
    links inside the repository are never followed.
    """
    shown = str(repo_path)
    try:
        root = Path(repo_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop on Python 3.11
        raise RepositoryRootError(
            f"Repository root {shown!r} does not exist or cannot be resolved "
            f"({type(exc).__name__}). Check the path given to load_repository, REPO_PATH, "
            "or --repo."
        ) from exc
    if not root.is_dir():
        raise RepositoryRootError(
            f"Repository root {shown!r} is not a directory. Point it at the top folder of "
            "the repository to index."
        )
    return root


def is_valid_name(name: str) -> bool:
    """True if a file or directory name is safe to use in paths, logs and IDs."""
    try:
        name.encode("utf-8")  # fails on undecodable bytes (surrogateescape)
    except UnicodeEncodeError:
        return False
    return "\\" not in name and not any(char < " " or char == "\x7f" for char in name)


def _kind(entry: os.DirEntry[str]) -> _Kind | None:
    """Entry type without following symlinks; None if it cannot be determined."""
    try:
        if entry.is_symlink():
            return _Kind.SYMLINK
        if entry.is_dir(follow_symlinks=False):
            return _Kind.DIRECTORY
        if entry.is_file(follow_symlinks=False):
            return _Kind.REGULAR
    except OSError:
        return None
    return _Kind.OTHER


def _shown(rel_dir: str, name: str) -> str:
    """Printable form of a path whose last component has an invalid name."""
    escaped = ascii(name)[1:-1]  # backslash escapes for control and non-ASCII characters
    return f"{rel_dir}/{escaped}" if rel_dir else escaped


def walk_repository(root: Path, policy: IgnorePolicy) -> WalkResult:
    """Walk ``root`` (already resolved) and return candidate files and discards.

    Raises :class:`RepositoryRootError` if the root cannot be listed and
    :class:`src.errors.IgnoreRulesError` if a ``.gitignore`` on the way is unusable.
    """
    files: list[WalkEntry] = []
    discarded: list[Discard] = []
    pruned: list[Discard] = []
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        directory, rel_dir = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as exc:
            if not rel_dir:
                raise RepositoryRootError(
                    f"Cannot list the repository root ({exc.strerror}). Check its permissions."
                ) from exc
            pruned.append(Discard(rel_dir, DiscardReason.UNREADABLE))
            logger.debug("ingestion.walk.unreadable_dir", extra={"path": rel_dir})
            continue
        for entry in entries:
            kind = _kind(entry)
            if not is_valid_name(entry.name):
                target = pruned if kind is _Kind.DIRECTORY else discarded
                target.append(Discard(_shown(rel_dir, entry.name), DiscardReason.INVALID_NAME))
                continue
            rel_path = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            if kind is None:
                discarded.append(Discard(rel_path, DiscardReason.UNREADABLE))
                continue
            is_dir = kind is _Kind.DIRECTORY
            verdict = policy.is_ignored(rel_path, is_dir=is_dir)
            if verdict.reason is not None:
                (pruned if is_dir else discarded).append(Discard(rel_path, verdict.reason))
            elif is_dir:
                stack.append((Path(entry.path), rel_path))
            elif kind is _Kind.SYMLINK:
                discarded.append(Discard(rel_path, DiscardReason.SYMLINK))
            elif kind is _Kind.OTHER:
                discarded.append(Discard(rel_path, DiscardReason.NOT_REGULAR_FILE))
            else:
                files.append(WalkEntry(rel_path, Path(entry.path)))
    return WalkResult(
        files=tuple(sorted(files, key=lambda entry: entry.rel_path)),
        discarded=_sorted(discarded),
        pruned=_sorted(pruned),
    )


def _sorted(discards: list[Discard]) -> tuple[Discard, ...]:
    return tuple(sorted(discards, key=lambda d: (d.rel_path, d.reason.value)))
