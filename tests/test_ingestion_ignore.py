"""Tests for the Step 2 ignore policy, on synthetic repositories in ``tmp_path``.

Discards come first: every test states what must be excluded and why (the reason code).
The last section compares the policy with ``git check-ignore`` in an isolated Git.
"""

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from src.errors import IgnoreRulesError
from src.ingestion.ignore import KEPT, IgnorePolicy, IgnoreVerdict
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"
NO_BUILTINS = IngestionRules(builtin_patterns=())


def _write(root: Path, rel: str, content: str | bytes = "x\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")
    return path


def _policy(
    root: Path, rules: IngestionRules = NO_BUILTINS, rules_file: Path | None = None
) -> IgnorePolicy:
    return IgnorePolicy(root, rules, rules_file)


def _reason(policy: IgnorePolicy, rel: str, *, is_dir: bool = False) -> DiscardReason | None:
    return policy.is_ignored(rel, is_dir=is_dir).reason


# --- gitignore syntax (root .gitignore) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "path", "is_dir", "ignored"),
    [
        ("secret.py\n", "secret.py", False, True),
        ("secret.py\n", "a/b/secret.py", False, True),  # no slash: any depth
        ("secret.py\n", "secret.pyc", False, False),
        ("*.log\n", "a.log", False, True),
        ("*.log\n", "x/y/b.log", False, True),
        ("*.log\n", "a.log.txt", False, False),
        ("**/foo\n", "foo", False, True),
        ("**/foo\n", "a/b/foo", False, True),
        ("/x\n", "x", False, True),  # anchored to the .gitignore's directory
        ("/x\n", "a/x", False, False),
        ("foo/\n", "foo", True, True),  # directory-only pattern
        ("foo/\n", "foo", False, False),  # ... does not match a file named foo
        ("foo/\n", "foo/bar.py", False, True),  # ... but excludes what is under the dir
        ("*.py\n!keep.py\n", "a.py", False, True),
        ("*.py\n!keep.py\n", "keep.py", False, False),  # last matching pattern wins
        ("*.py\n!keep.py\n", "sub/keep.py", False, False),
        ("\\#hash\n", "#hash", False, True),  # escaped comment character
        ("\\!bang\n", "!bang", False, True),  # escaped negation character
        ("# comment\n", "# comment", False, False),  # comments are not patterns
        ("\n\n\n", "anything.py", False, False),  # blank lines are no-ops
        ("trail   \n", "trail", False, True),  # unescaped trailing spaces are ignored
        ("a.py\r\nb.py\r\n", "b.py", False, True),  # CRLF line endings
        ("\ufeffa.py\n", "a.py", False, True),  # UTF-8 byte order mark
    ],
)
def test_gitignore_syntax(
    tmp_path: Path, lines: str, path: str, is_dir: bool, ignored: bool
) -> None:
    _write(tmp_path, ".gitignore", lines)

    verdict = _policy(tmp_path).is_ignored(path, is_dir=is_dir)

    assert verdict.ignored is ignored
    if ignored:
        assert verdict == IgnoreVerdict(DiscardReason.GITIGNORE, ".gitignore")


# --- nested .gitignore files (decision Q8) ------------------------------------------------------


def test_nested_gitignore_is_relative_to_its_own_directory(tmp_path: Path) -> None:
    _write(tmp_path, "a/.gitignore", "/x.py\n")
    policy = _policy(tmp_path)

    assert policy.is_ignored("a/x.py", is_dir=False) == IgnoreVerdict(
        DiscardReason.GITIGNORE, "a/.gitignore"
    )
    assert _reason(policy, "x.py") is None
    assert _reason(policy, "a/b/x.py") is None


def test_nested_pattern_without_slash_matches_at_any_depth_below_its_directory(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a/.gitignore", "y.py\n")
    policy = _policy(tmp_path)

    assert _reason(policy, "a/y.py") == DiscardReason.GITIGNORE
    assert _reason(policy, "a/b/c/y.py") == DiscardReason.GITIGNORE
    assert _reason(policy, "y.py") is None
    assert _reason(policy, "b/y.py") is None


def test_deeper_gitignore_can_re_include(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.py\n")
    _write(tmp_path, "sub/.gitignore", "!keep.py\n")
    policy = _policy(tmp_path)

    assert _reason(policy, "sub/keep.py") is None
    assert _reason(policy, "sub/other.py") == DiscardReason.GITIGNORE
    assert _reason(policy, "keep.py") == DiscardReason.GITIGNORE


def test_deeper_gitignore_can_exclude_what_the_root_re_includes(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.tmp\n!keep.tmp\n")
    _write(tmp_path, "sub/.gitignore", "keep.tmp\n")
    policy = _policy(tmp_path)

    assert policy.is_ignored("sub/keep.tmp", is_dir=False) == IgnoreVerdict(
        DiscardReason.GITIGNORE, "sub/.gitignore"
    )
    assert _reason(policy, "keep.tmp") is None


def test_file_under_an_excluded_directory_cannot_be_re_included(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "out/\n!out/keep.py\n")
    _write(tmp_path, "out/.gitignore", "!keep.py\n")
    policy = _policy(tmp_path)

    assert _reason(policy, "out/keep.py") == DiscardReason.GITIGNORE
    assert _reason(policy, "out", is_dir=True) == DiscardReason.GITIGNORE
    assert "out/.gitignore" not in policy.loaded_gitignores  # never read


def test_gitignore_files_under_excluded_directories_are_not_read(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "ignored/\n")
    _write(tmp_path, "ignored/.gitignore", b"\xff would fail to decode if read\n")
    _write(tmp_path, "ignored/deeper/.gitignore", "*\n")
    _write(tmp_path, "kept/.gitignore", "*.log\n")
    policy = _policy(tmp_path)

    for path in ("ignored/a.py", "ignored/deeper/b.py", "kept/a.py", "kept/c.log"):
        policy.is_ignored(path, is_dir=False)

    assert policy.loaded_gitignores == (".gitignore", "kept/.gitignore")


# --- the versioned corpus rules file (decisions Q1, Q2, D1) -------------------------------------


@pytest.mark.parametrize(
    ("path", "is_dir"),
    [
        ("deploy/model", True),
        ("deploy/model/x.py", False),
        ("deploy/model/sub/y.yaml", False),
        ("project_context", True),
        ("project_context/c.md", False),
    ],
)
def test_rules_file_excludes_the_agreed_corpus_paths(
    tmp_path: Path, path: str, is_dir: bool
) -> None:
    policy = _policy(tmp_path, IngestionRules(), RULES_FILE)

    assert policy.is_ignored(path, is_dir=is_dir) == IgnoreVerdict(
        DiscardReason.CORPUS_RULES, "rules_file"
    )


@pytest.mark.parametrize(
    "path",
    [
        "src/deploy/model/x.py",  # the rules are anchored at the corpus root
        "src/project_context/c.md",
        "deploy/other.py",
        "docs/a.md",
        "docs/decisions/0001-x.md",
        "README.md",
        "src/a.py",
        "tests/test_a.py",
        "pipelines/p.py",
        "scripts/s.py",
        "main.py",
        "dvc.yaml",
        ".github/workflows/ci.yml",
        "Dockerfile",
        "docker/Dockerfile",
        "Makefile",
        "pyproject.toml",
    ],
)
def test_rules_file_keeps_everything_else(tmp_path: Path, path: str) -> None:
    policy = _policy(tmp_path, IngestionRules(), RULES_FILE)

    assert policy.is_ignored(path, is_dir=False) == KEPT


# --- built-in patterns (decision D2) ------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "node_modules/x/index.js",
        "web/node_modules/x.js",
        "__pycache__/a.cpython-311.pyc",
        "src/__pycache__/a.cpython-311.pyc",
        ".venv/lib/x.py",
        "venv/x.py",
        "a/.venv/x.py",
        ".pytest_cache/v/x",
        ".ruff_cache/x",
        ".mypy_cache/x.json",
        "lib/site-packages/x.py",
        ".next/x.js",
        "build/x.py",  # ambiguous names: at the root only
        "dist/x.js",
        "target/x.rs",
        "vendor/x.go",
        ".git/config",
        "sub/.git",  # the .git *file* of a worktree or submodule
    ],
)
def test_builtin_patterns_exclude_generic_noise(tmp_path: Path, path: str) -> None:
    policy = _policy(tmp_path, IngestionRules())

    assert policy.is_ignored(path, is_dir=False) == IgnoreVerdict(
        DiscardReason.BUILTIN_PATTERN, "builtin"
    )


@pytest.mark.parametrize(
    "path", ["src/build/x.py", "pkg/vendor/x.go", "app/dist/x.js", "crate/target/x.rs"]
)
def test_nested_ambiguous_directories_survive(tmp_path: Path, path: str) -> None:
    assert _policy(tmp_path, IngestionRules()).is_ignored(path, is_dir=False) == KEPT


def test_git_directory_is_always_excluded(tmp_path: Path) -> None:
    rules_file = _write(tmp_path / "cfg", "rules.ignore", "!.git\n!.git/\n")
    repo = tmp_path / "repo"
    _write(repo, ".gitignore", "!.git\n!.git/\n")
    policy = _policy(repo, IngestionRules(), rules_file)

    assert _reason(policy, ".git", is_dir=True) == DiscardReason.BUILTIN_PATTERN
    assert _reason(policy, ".git/HEAD") == DiscardReason.BUILTIN_PATTERN


# --- precedence and no cross-source re-inclusion ------------------------------------------------


def test_first_excluding_source_gives_the_reason(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "node_modules/\n*.py\n")
    policy = _policy(tmp_path, IngestionRules(), RULES_FILE)

    assert _reason(policy, "node_modules/x.js") == DiscardReason.BUILTIN_PATTERN
    assert _reason(policy, "deploy/model/x.py") == DiscardReason.CORPUS_RULES
    assert _reason(policy, "src/x.py") == DiscardReason.GITIGNORE


def test_no_source_can_re_include_what_another_excludes(tmp_path: Path) -> None:
    rules_file = _write(tmp_path / "cfg", "rules.ignore", "/generated/\n!/src/a.py\n")
    repo = tmp_path / "repo"
    _write(repo, ".gitignore", "a.py\n!generated/\n!generated/x.py\n!node_modules/\n")
    policy = _policy(repo, IngestionRules(), rules_file)

    assert _reason(policy, "src/a.py") == DiscardReason.GITIGNORE
    assert _reason(policy, "generated/x.py") == DiscardReason.CORPUS_RULES
    assert _reason(policy, "node_modules/y.js") == DiscardReason.BUILTIN_PATTERN


def test_negation_works_inside_the_rules_file(tmp_path: Path) -> None:
    rules_file = _write(tmp_path / "cfg", "rules.ignore", "/docs/*\n!/docs/keep.md\n")
    policy = _policy(tmp_path / "repo", NO_BUILTINS, rules_file)

    assert _reason(policy, "docs/drop.md") == DiscardReason.CORPUS_RULES
    assert _reason(policy, "docs/keep.md") is None


# --- failures (decision D3) ---------------------------------------------------------------------


def _run_with_timeout(action: Callable[[], object], seconds: float = 5.0) -> object:
    """Run ``action`` in a thread so a blocking open fails the test instead of hanging it."""
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["value"] = action()
        except IgnoreRulesError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "the call blocked"
    return outcome.get("error", outcome.get("value"))


def test_gitignore_that_is_a_directory_fails_closed(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").mkdir()

    with pytest.raises(IgnoreRulesError, match="not a regular file"):
        _policy(tmp_path).is_ignored("a.py", is_dir=False)


def test_undecodable_gitignore_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", b"*.log\n\xff\xfe\n")

    with pytest.raises(IgnoreRulesError, match="UTF-8"):
        _policy(tmp_path).is_ignored("a.py", is_dir=False)


def test_gitignore_over_the_size_cap_fails_closed(tmp_path: Path) -> None:
    rules = IngestionRules(builtin_patterns=(), max_gitignore_bytes=16)
    _write(tmp_path, ".gitignore", "a" * 16 + "\n")  # 17 bytes
    _write(tmp_path, "ok/.gitignore", "b" * 15 + "\n")  # 16 bytes: at the cap

    policy = IgnorePolicy(tmp_path, rules)
    assert _reason(IgnorePolicy(tmp_path / "ok", rules), "b" * 15) == DiscardReason.GITIGNORE
    with pytest.raises(IgnoreRulesError, match="cap"):
        policy.is_ignored("a.py", is_dir=False)


@pytest.mark.parametrize("target_inside", [True, False])
def test_symlinked_gitignore_fails_closed(tmp_path: Path, target_inside: bool) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = _write(repo if target_inside else tmp_path, "real.ignore", "*.py\n")
    (repo / ".gitignore").symlink_to(target)

    with pytest.raises(IgnoreRulesError, match="symbolic link"):
        _policy(repo).is_ignored("a.py", is_dir=False)


def test_gitignore_under_a_symlinked_directory_fails_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write(outside, ".gitignore", "*.py\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(IgnoreRulesError, match="symbolic link"):
        _policy(repo).is_ignored("link/a.py", is_dir=False)


def test_fifo_gitignore_fails_closed_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / ".gitignore")

    error = _run_with_timeout(lambda: _policy(tmp_path).is_ignored("a.py", is_dir=False))

    assert isinstance(error, IgnoreRulesError)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_gitignore_fails_closed(tmp_path: Path) -> None:
    path = _write(tmp_path, ".gitignore", "*.py\n")
    path.chmod(0)
    try:
        with pytest.raises(IgnoreRulesError, match="cannot be read"):
            _policy(tmp_path).is_ignored("a.py", is_dir=False)
    finally:
        path.chmod(0o600)


def test_invalid_gitignore_line_is_skipped_with_a_warning_and_no_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path, ".gitignore", "[z-a]\n*.log\n!\n")

    with caplog.at_level(logging.WARNING, logger="src.ingestion.ignore"):
        reason = _reason(_policy(tmp_path), "a.log")

    assert reason == DiscardReason.GITIGNORE  # the valid line still applies
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [(r.__dict__["ignore_file"], r.__dict__["line_number"]) for r in warnings] == [
        (".gitignore", 1),
        (".gitignore", 3),
    ]
    assert "[z-a]" not in caplog.text


def test_missing_gitignore_is_normal(tmp_path: Path) -> None:
    assert _policy(tmp_path).is_ignored("a/b/c.py", is_dir=False) == KEPT


def test_missing_rules_file_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(IgnoreRulesError, match="does not exist"):
        _policy(tmp_path, IngestionRules(), tmp_path / "missing.ignore")


def test_rules_file_is_strict_about_invalid_lines(tmp_path: Path) -> None:
    rules_file = _write(tmp_path, "rules.ignore", "/ok/\n[z-a]\n")

    with pytest.raises(IgnoreRulesError, match="line 2"):
        _policy(tmp_path, IngestionRules(), rules_file)


def test_rules_file_that_is_a_directory_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(IgnoreRulesError, match="not a regular file"):
        _policy(tmp_path, IngestionRules(), tmp_path)


# --- input validation, determinism and caching --------------------------------------------------


@pytest.mark.parametrize(
    "path", ["", "/abs/a.py", "a/../b.py", "..", "a\\b.py", "a//b.py", "./a.py", "a/", "a\0b"]
)
def test_invalid_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        _policy(tmp_path).is_ignored(path, is_dir=False)


def _sample_repo(root: Path) -> list[tuple[str, bool]]:
    _write(root, ".gitignore", "*.log\nout/\n")
    _write(root, "a/.gitignore", "!keep.log\n/local.py\n")
    _write(root, "a/b/.gitignore", "*.py\n")
    return [
        ("x.log", False),
        ("a/keep.log", False),
        ("a/local.py", False),
        ("a/b/c.py", False),
        ("a/b", True),
        ("out", True),
        ("out/x.py", False),
        ("node_modules/y.js", False),
        ("src/main.py", False),
    ]


def test_same_queries_give_same_verdicts_in_any_order(tmp_path: Path) -> None:
    queries = _sample_repo(tmp_path)
    rules = IngestionRules()

    forward = {q: IgnorePolicy(tmp_path, rules).is_ignored(q[0], is_dir=q[1]) for q in queries}
    reverse_policy = IgnorePolicy(tmp_path, rules)
    reverse = {q: reverse_policy.is_ignored(q[0], is_dir=q[1]) for q in reversed(queries)}

    assert forward == reverse


def test_each_gitignore_is_read_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = _sample_repo(tmp_path)
    opened: list[str] = []
    real_open = os.open

    def counting_open(path: str | os.PathLike[str], flags: int, *args: int) -> int:
        if os.fspath(path).endswith(".gitignore"):
            opened.append(os.fspath(path))
        return real_open(path, flags, *args)

    policy = _policy(tmp_path, IngestionRules())
    monkeypatch.setattr(os, "open", counting_open)
    for _ in range(3):
        for path, is_dir in queries:
            policy.is_ignored(path, is_dir=is_dir)

    assert len(opened) == len(set(opened))
    assert policy.loaded_gitignores == (".gitignore", "a/.gitignore", "a/b/.gitignore")


# --- differential comparison with Git -----------------------------------------------------------

GIT = shutil.which("git")

DIFF_GITIGNORES = {
    ".gitignore": "*.log\n!keep.log\n/anchored.txt\nout/\n!out/keep.py\n"
    "docs/**/*.tmp\nfoo/\n\\#hash\ntrail   \n*.py[cod]\n",
    "sub/.gitignore": "!x.log\n/local.txt\nnested.txt\n",
    "sub/deep/.gitignore": "*.txt\n!keep.txt\n",
    "data/raw/.gitignore": "/*\n!.gitignore\n",
    "crlf/.gitignore": "a.txt\r\nb.txt\r\n",
    "bom/.gitignore": "\ufeffz.txt\n",
}
DIFF_FILES = (
    "a.log",
    "keep.log",
    "sub/a.log",
    "sub/x.log",
    "anchored.txt",
    "sub/anchored.txt",
    "out/keep.py",
    "out/x.py",
    "docs/a/b/c.tmp",
    "docs/c.tmp",
    "foo/x.py",
    "bar/foo",
    "#hash",
    "trail",
    "a.pyc",
    "sub/local.txt",
    "sub/nested.txt",
    "sub/deep/local.txt",
    "sub/deep/nested.txt",
    "sub/deep/keep.txt",
    "data/raw/x.csv",
    "crlf/a.txt",
    "crlf/b.txt",
    "crlf/c.txt",
    "bom/z.txt",
    "src/main.py",
    "README.md",
)


def _isolated_git_env(home: Path) -> dict[str, str]:
    """Git with no user, global or system configuration (no excludesFile, no hooks)."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "LC_ALL": "C",
    }


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_policy_matches_git_check_ignore(tmp_path: Path) -> None:
    assert GIT is not None
    repo, home = tmp_path / "repo", tmp_path / "home"
    home.mkdir()
    for rel, content in DIFF_GITIGNORES.items():
        _write(repo, rel, content)
    for rel in DIFF_FILES:
        _write(repo, rel)
    env = _isolated_git_env(home)
    # S603: trusted input only (git from PATH, constant arguments, paths under tmp_path).
    subprocess.run([GIT, "init", "-q", str(repo)], env=env, check=True, timeout=60)  # noqa: S603
    queries = sorted(
        {
            str(path.relative_to(repo))
            for path in repo.rglob("*")
            if ".git" not in path.relative_to(repo).parts
        }
    )

    result = subprocess.run(  # noqa: S603 (same trusted input as above)
        [GIT, "-C", str(repo), "check-ignore", "--no-index", "--stdin", "-z"],
        input="\0".join(queries) + "\0",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode in {0, 1}, result.stderr  # 1 means "nothing ignored"
    git_ignored = {p for p in result.stdout.split("\0") if p}
    policy = _policy(repo)
    ours = {q for q in queries if policy.is_ignored(q, is_dir=(repo / q).is_dir()).ignored}

    assert ours == git_ignored
    assert "out/keep.py" in ours  # the case where pathspec alone would disagree with Git
