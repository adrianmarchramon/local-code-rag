"""Tests for the Step 4 walker: pruning, symlinks, special files, names, order, root errors."""

import inspect
import os
import random
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.errors import RepositoryRootError
from src.ingestion.ignore import IgnorePolicy
from src.ingestion.reasons import Discard, DiscardReason
from src.ingestion.rules import IngestionRules
from src.ingestion.walker import WalkResult, is_valid_name, resolve_root, walk_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"


def _write(root: Path, rel: str, content: bytes = b"x = 1\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _walk(root: Path, rules_file: Path | None = None) -> WalkResult:
    resolved = resolve_root(root)
    return walk_repository(resolved, IgnorePolicy(resolved, IngestionRules(), rules_file))


def _files(result: WalkResult) -> list[str]:
    return [entry.rel_path for entry in result.files]


def _reasons(discards: tuple[Discard, ...]) -> dict[str, DiscardReason]:
    return {d.rel_path: d.reason for d in discards}


# --- pruning ------------------------------------------------------------------------------------


def test_ignored_directories_are_pruned_without_descending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "src/a.py")
    _write(tmp_path, "node_modules/pkg/index.js")
    _write(tmp_path, "deploy/model/m.py")
    _write(tmp_path, "out/x.py")
    _write(tmp_path, "out/.gitignore", b"\xff would fail to decode if read\n")
    _write(tmp_path, ".gitignore", b"out/\n")
    listed: list[str] = []
    real_scandir = os.scandir

    def recording_scandir(path: str | os.PathLike[str]) -> Iterator[os.DirEntry[str]]:
        listed.append(Path(path).relative_to(tmp_path).as_posix())
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", recording_scandir)
    result = _walk(tmp_path, RULES_FILE)

    assert _files(result) == [".gitignore", "src/a.py"]
    assert _reasons(result.pruned) == {
        "deploy/model": DiscardReason.CORPUS_RULES,
        "node_modules": DiscardReason.BUILTIN_PATTERN,
        "out": DiscardReason.GITIGNORE,
    }
    assert sorted(listed) == [".", "deploy", "src"]  # pruned directories never listed


def test_git_directory_and_git_file_are_excluded(tmp_path: Path) -> None:
    _write(tmp_path, ".git/HEAD", b"ref: refs/heads/main\n")
    _write(tmp_path, "sub/.git", b"gitdir: ../.git/worktrees/sub\n")
    _write(tmp_path, "sub/a.py")

    result = _walk(tmp_path)

    assert _files(result) == ["sub/a.py"]
    assert _reasons(result.pruned) == {".git": DiscardReason.BUILTIN_PATTERN}
    assert _reasons(result.discarded) == {"sub/.git": DiscardReason.BUILTIN_PATTERN}


def test_nested_gitignore_semantics_hold_through_the_walker(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", b"*.log\n")
    _write(tmp_path, "sub/.gitignore", b"!keep.log\n/local.py\n")
    for rel in ("a.log", "sub/keep.log", "sub/local.py", "sub/deeper/local.py"):
        _write(tmp_path, rel)

    result = _walk(tmp_path)

    assert _files(result) == [".gitignore", "sub/.gitignore", "sub/deeper/local.py", "sub/keep.log"]
    assert _reasons(result.discarded) == {
        "a.log": DiscardReason.GITIGNORE,
        "sub/local.py": DiscardReason.GITIGNORE,
    }


# --- symlinks and special files -----------------------------------------------------------------


def test_symlinks_are_never_followed_and_are_counted(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write(outside, "secret.py", b"token = 1\n")
    repo = tmp_path / "repo"
    _write(repo, "real/a.py")
    (repo / "file_link.py").symlink_to(repo / "real" / "a.py")
    (repo / "dir_link").symlink_to(repo / "real", target_is_directory=True)
    (repo / "outside_link").symlink_to(outside, target_is_directory=True)
    (repo / "outside_file.py").symlink_to(outside / "secret.py")
    (repo / "broken.py").symlink_to(repo / "nowhere.py")
    (repo / "loop").symlink_to(repo / "loop")  # a symlink cycle

    result = _walk(repo)

    assert _files(result) == ["real/a.py"]
    assert _reasons(result.discarded) == {
        name: DiscardReason.SYMLINK
        for name in (
            "broken.py",
            "dir_link",
            "file_link.py",
            "loop",
            "outside_file.py",
            "outside_link",
        )
    }
    assert result.pruned == ()


def test_fifo_is_counted_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe.py")
    _write(tmp_path, "a.py")
    outcome: list[WalkResult] = []

    worker = threading.Thread(target=lambda: outcome.append(_walk(tmp_path)), daemon=True)
    worker.start()
    worker.join(5)

    assert not worker.is_alive(), "the walker blocked on a FIFO"
    assert _files(outcome[0]) == ["a.py"]
    assert _reasons(outcome[0].discarded) == {"pipe.py": DiscardReason.NOT_REGULAR_FILE}


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unreadable_directory_is_counted_without_aborting(tmp_path: Path) -> None:
    _write(tmp_path, "locked/a.py")
    _write(tmp_path, "open/b.py")
    (tmp_path / "locked").chmod(0)
    try:
        result = _walk(tmp_path)
    finally:
        (tmp_path / "locked").chmod(0o700)

    assert _files(result) == ["open/b.py"]
    assert _reasons(result.pruned) == {"locked": DiscardReason.UNREADABLE}


# --- names --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("a.py", True),
        ("café.md", True),
        ("with space.py", True),
        ("back\\slash.py", False),
        ("new\nline.py", False),
        ("tab\t.py", False),
        ("del\x7f.py", False),
        ("bad\udcff.py", False),  # an undecodable byte, as os.scandir reports it
    ],
)
def test_name_validity(name: str, valid: bool) -> None:
    assert is_valid_name(name) is valid


def test_invalid_names_are_discarded_and_counted(tmp_path: Path) -> None:
    raw = os.fsencode(tmp_path)
    for name in (b"back\\slash.py", b"new\nline.py", b"bad\xff.py"):
        try:
            fd = os.open(os.path.join(raw, name), os.O_WRONLY | os.O_CREAT, 0o600)
        except OSError:
            pytest.skip("the filesystem does not accept this file name")
        os.close(fd)
    os.mkdir(os.path.join(raw, b"dir\xfe"))
    _write(tmp_path, "ok.py")

    result = _walk(tmp_path)

    assert _files(result) == ["ok.py"]
    assert sorted(_reasons(result.discarded)) == [
        "back\\\\slash.py",
        "bad\\udcff.py",
        "new\\nline.py",
    ]
    assert set(_reasons(result.discarded).values()) == {DiscardReason.INVALID_NAME}
    assert _reasons(result.pruned) == {"dir\\udcfe": DiscardReason.INVALID_NAME}


# --- ordering -----------------------------------------------------------------------------------


def test_order_is_by_full_posix_string(tmp_path: Path) -> None:
    for rel in ("a.py", "a/b.py", "a-b.py", "B.py", "a/a.py", "a.b/c.py"):
        _write(tmp_path, rel)

    # '-' (0x2D) < '.' (0x2E) < '/' (0x2F); uppercase sorts before lowercase.
    assert _files(_walk(tmp_path)) == ["B.py", "a-b.py", "a.b/c.py", "a.py", "a/a.py", "a/b.py"]


def test_order_does_not_depend_on_enumeration_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(30):
        _write(tmp_path, f"d{index % 4}/f{index}.py")
    (tmp_path / "link.py").symlink_to(tmp_path / "d0" / "f0.py")
    expected = _walk(tmp_path)
    shuffler = random.Random(1234)  # noqa: S311 (test ordering only, not cryptography)
    real_scandir = os.scandir

    class ShuffledScandir:
        def __init__(self, path: str | os.PathLike[str]) -> None:
            with real_scandir(path) as iterator:
                self._entries = list(iterator)
            shuffler.shuffle(self._entries)

        def __enter__(self) -> list[os.DirEntry[str]]:
            return self._entries

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(os, "scandir", ShuffledScandir)

    assert _walk(tmp_path) == expected


def test_very_deep_tree_needs_no_recursion(tmp_path: Path) -> None:
    depth = 300
    deep = tmp_path
    for _ in range(depth):  # level by level: mkdir(parents=True) recurses on Python 3.11
        deep = deep / "d"
        deep.mkdir()
    (deep / "leaf.py").write_bytes(b"x = 1\n")
    root = resolve_root(tmp_path)
    policy = IgnorePolicy(root, IngestionRules())
    limit = sys.getrecursionlimit()
    # A recursive walker needs one frame per level: allow far fewer than `depth` extra frames.
    sys.setrecursionlimit(len(inspect.stack()) + 100)
    try:
        result = walk_repository(root, policy)
    finally:
        sys.setrecursionlimit(limit)

    assert _files(result) == ["/".join(["d"] * depth + ["leaf.py"])]


# --- root ---------------------------------------------------------------------------------------


def test_missing_root_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(RepositoryRootError, match="does not exist"):
        resolve_root(tmp_path / "missing")


def test_file_root_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(RepositoryRootError, match="not a directory"):
        resolve_root(_write(tmp_path, "file.py"))


def test_symlinked_root_is_resolved_once(tmp_path: Path) -> None:
    _write(tmp_path, "real/a.py")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)

    root = resolve_root(tmp_path / "alias")

    assert root == (tmp_path / "real").resolve()
    assert _files(_walk(tmp_path / "alias")) == ["a.py"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unlistable_root_is_fatal(tmp_path: Path) -> None:
    root = tmp_path / "locked"
    root.mkdir()
    root.chmod(0o300)  # searchable but not readable
    try:
        with pytest.raises(RepositoryRootError, match="Cannot list"):
            _walk(root)
    finally:
        root.chmod(0o700)


# --- no I/O outside the root --------------------------------------------------------------------


def test_walk_touches_nothing_outside_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    _write(outside, "a.py")
    repo = tmp_path / "repo"
    _write(repo, "src/a.py")
    _write(repo, ".gitignore", b"*.log\n")
    (repo / "out_link").symlink_to(outside, target_is_directory=True)
    root = resolve_root(repo)
    policy = IgnorePolicy(root, IngestionRules(), RULES_FILE)  # reads the rules file now
    touched: list[str] = []
    for name in ("scandir", "lstat", "stat", "open"):
        real = getattr(os, name)

        def recorder(path: object, *args: object, _real: object = real, **kwargs: object) -> object:
            touched.append(os.fsdecode(path))  # type: ignore[arg-type]
            return _real(path, *args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(os, name, recorder)

    walk_repository(root, policy)

    assert touched
    assert all(Path(p).is_relative_to(root) for p in touched), touched
