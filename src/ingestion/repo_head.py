"""HEAD of a repository, read from ``.git`` in pure Python (moved to ``src/`` by decision P6).

The indexed repository is untrusted data: running ``git`` there would honour its
``.git/config``, so ``git`` is never run. Symlinks are never followed and nothing outside
``.git`` is opened. Phase 1's key test uses this to check the frozen corpus; Phase 3
records the result in the index manifest (decision P7).
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Final

from src.errors import IngestionError

_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}")
_REF_RE: Final = re.compile(r"refs/[A-Za-z0-9._/-]+")
_MAX_HEAD_BYTES: Final = 4 * 1024
_MAX_PACKED_REFS_BYTES: Final = 1024 * 1024
_MAX_SYMREF_DEPTH: Final = 5
_O_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class RepositoryHeadError(IngestionError):
    """HEAD of a repository could not be resolved from ``.git`` without running git."""


def _read_git_file(git_dir: Path, rel: str, limit: int) -> str | None:
    """Text of ``git_dir/rel``, or None if it does not exist. Never follows a symlink."""
    path = git_dir / rel
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RepositoryHeadError(f".git/{rel} cannot be inspected: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise RepositoryHeadError(f".git/{rel} is not a regular file (symlinks are not followed)")
    if info.st_size > limit:
        raise RepositoryHeadError(f".git/{rel} has {info.st_size} bytes, over the {limit} limit")
    try:
        fd = os.open(path, _O_FLAGS)
        try:
            data = os.read(fd, limit + 1)
        finally:
            os.close(fd)
        return data.decode("ascii")
    except OSError as error:
        raise RepositoryHeadError(f".git/{rel} cannot be read: {error}") from error
    except UnicodeDecodeError as error:
        raise RepositoryHeadError(f".git/{rel} is not ASCII text: {error.reason}") from error


def _check_ref_name(ref: str) -> str:
    parts = ref.split("/")
    if not _REF_RE.fullmatch(ref) or any(part in {"", ".", ".."} for part in parts):
        raise RepositoryHeadError(f"unsupported or unsafe ref name {ref!r} (expected refs/...)")
    return ref


def _packed_ref(git_dir: Path, ref: str) -> str | None:
    text = _read_git_file(git_dir, "packed-refs", _MAX_PACKED_REFS_BYTES)
    if text is None:
        return None
    for number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith(("#", "^")):  # header, or a peeled tag
            continue
        sha, _, name = line.partition(" ")
        if not _SHA1_RE.fullmatch(sha) or not name:
            raise RepositoryHeadError(f".git/packed-refs line {number} is malformed")
        if name == ref:
            return sha
    return None


def read_head(repo: Path) -> str:
    """The commit SHA that HEAD points to: detached, loose ref or ``packed-refs``.

    Raises :class:`RepositoryHeadError` when ``.git`` is missing, is not a directory (a
    worktree or submodule ``.git`` file points elsewhere and is not followed), or holds
    something malformed.
    """
    git_dir = repo / ".git"
    try:
        info = git_dir.lstat()
    except FileNotFoundError as error:
        raise RepositoryHeadError(f"{repo} has no .git directory") from error
    except OSError as error:
        raise RepositoryHeadError(f"{git_dir} cannot be inspected: {error}") from error
    if not stat.S_ISDIR(info.st_mode):
        raise RepositoryHeadError(
            f"{git_dir} is not a directory (a symlink, or a worktree/submodule .git file "
            "pointing elsewhere); it is not followed"
        )
    name = "HEAD"
    for _ in range(_MAX_SYMREF_DEPTH):
        text = _read_git_file(git_dir, name, _MAX_HEAD_BYTES)
        if text is None:
            sha = _packed_ref(git_dir, name) if name != "HEAD" else None
            if sha is None:
                raise RepositoryHeadError(f"{name} is neither a loose nor a packed ref")
            return sha
        content = text.rstrip("\n")
        if _SHA1_RE.fullmatch(content):
            return content
        if not content.startswith("ref: "):
            raise RepositoryHeadError(f".git/{name} is neither a SHA nor 'ref: <name>'")
        name = _check_ref_name(content.removeprefix("ref: "))
    raise RepositoryHeadError(f"more than {_MAX_SYMREF_DEPTH} levels of symbolic refs")
