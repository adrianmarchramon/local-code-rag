"""Hermetic tests for the key-test support (tests/corpus_manifest.py), on synthetic data.

The key test needs the real corpus and skips everywhere else; its logic (HEAD resolution,
manifest parsing, comparison) is checked here so that it runs on any machine.
"""

import subprocess
from collections import Counter
from pathlib import Path

import pytest
from corpus_manifest import (
    FROZEN_CORPUS_SHA,
    MANIFEST_PATH,
    REGENERATE_COMMAND,
    CorpusHeadError,
    Manifest,
    ManifestError,
    ManifestRow,
    compare_rows,
    corpus_skip_reason,
    format_manifest,
    parse_manifest,
    read_head,
    rows_from_documents,
)
from llama_index.core import Document

SHA_A = "a" * 40
SHA_B = "b" * 40
HASH = "0" * 64
GENERATED = "2026-11-25T00:00:00Z"


def _git(repo: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = repo / ".git" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii")
    return repo


def _row(path: str, language: str = "python", size: int = 1, sha: str = HASH) -> ManifestRow:
    return ManifestRow(path, language, size, sha)


ROWS = (_row("README.md", "markdown", 10), _row("src/a.py", size=5), _row("src/b.py", size=7))


# --- read_head -----------------------------------------------------------------------------------


def test_detached_head(tmp_path: Path) -> None:
    assert read_head(_git(tmp_path, {"HEAD": f"{SHA_A}\n"})) == SHA_A


def test_loose_ref(tmp_path: Path) -> None:
    repo = _git(tmp_path, {"HEAD": "ref: refs/heads/main\n", "refs/heads/main": f"{SHA_A}\n"})

    assert read_head(repo) == SHA_A


def test_packed_ref(tmp_path: Path) -> None:
    packed = f"# pack-refs with: peeled fully-peeled sorted\n{SHA_B} refs/tags/v1\n^{SHA_A}\n"
    packed += f"{SHA_A} refs/heads/main\n"
    repo = _git(tmp_path, {"HEAD": "ref: refs/heads/main\n", "packed-refs": packed})

    assert read_head(repo) == SHA_A


def test_loose_ref_wins_over_packed_ref(tmp_path: Path) -> None:
    repo = _git(
        tmp_path,
        {
            "HEAD": "ref: refs/heads/main\n",
            "refs/heads/main": f"{SHA_B}\n",
            "packed-refs": f"{SHA_A} refs/heads/main\n",
        },
    )

    assert read_head(repo) == SHA_B


def test_missing_git_directory(tmp_path: Path) -> None:
    with pytest.raises(CorpusHeadError, match=r"no \.git directory"):
        read_head(tmp_path)


def test_git_file_of_a_worktree_is_not_followed(tmp_path: Path) -> None:
    elsewhere = _git(tmp_path / "elsewhere", {"HEAD": f"{SHA_A}\n"})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text(f"gitdir: {elsewhere / '.git'}\n", encoding="ascii")

    with pytest.raises(CorpusHeadError, match="not a directory"):
        read_head(repo)


def test_symlinked_git_directory_is_not_followed(tmp_path: Path) -> None:
    elsewhere = _git(tmp_path / "elsewhere", {"HEAD": f"{SHA_A}\n"})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").symlink_to(elsewhere / ".git", target_is_directory=True)

    with pytest.raises(CorpusHeadError, match="not a directory"):
        read_head(repo)


def test_symlinked_head_is_not_followed(tmp_path: Path) -> None:
    (tmp_path / "outside").write_text(f"{SHA_A}\n", encoding="ascii")
    repo = _git(tmp_path / "repo", {"config": ""})
    (repo / ".git" / "HEAD").symlink_to(tmp_path / "outside")

    with pytest.raises(CorpusHeadError, match="not a regular file"):
        read_head(repo)


@pytest.mark.parametrize(
    ("files", "match"),
    [
        ({"HEAD": "garbage\n"}, "neither a SHA"),
        ({"HEAD": "ref: refs/heads/../../../outside\n"}, "unsafe ref name"),
        ({"HEAD": "ref: ../outside\n"}, "unsafe ref name"),
        ({"HEAD": "ref: refs/heads/main\n"}, "neither a loose nor a packed ref"),
        ({"HEAD": "ref: refs/heads/main\n", "packed-refs": "nonsense\n"}, "line 1 is malformed"),
        ({"HEAD": "ref: refs/heads/main\n", "refs/heads/main": "short\n"}, "neither a SHA"),
        ({"HEAD": "ref: refs/heads/a\n", "refs/heads/a": "ref: refs/heads/a\n"}, "levels"),
        ({"HEAD": "x" * 5000}, "over the 4096 limit"),
    ],
)
def test_malformed_git_data_is_a_typed_error(
    tmp_path: Path, files: dict[str, str], match: str
) -> None:
    with pytest.raises(CorpusHeadError, match=match):
        read_head(_git(tmp_path, files))


def test_non_ascii_head_is_a_typed_error(tmp_path: Path) -> None:
    repo = _git(tmp_path, {"config": ""})
    (repo / ".git" / "HEAD").write_bytes(b"ref: refs/heads/m\xc3\xa1in\n")

    with pytest.raises(CorpusHeadError, match="not ASCII"):
        read_head(repo)


def test_git_is_never_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a subprocess was started")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    repo = _git(tmp_path, {"HEAD": "ref: refs/heads/main\n", "refs/heads/main": f"{SHA_A}\n"})

    assert read_head(repo) == SHA_A


# --- corpus_skip_reason (Q7: skip only if missing or not the frozen commit) ---------------------


def test_skip_reason_for_a_missing_corpus(tmp_path: Path) -> None:
    reason = corpus_skip_reason(tmp_path / "absent")

    assert reason is not None
    assert "not found" in reason
    assert str(tmp_path / "absent") in reason


def test_skip_reason_for_a_directory_that_is_not_a_repository(tmp_path: Path) -> None:
    reason = corpus_skip_reason(tmp_path)

    assert reason is not None
    assert "cannot tell" in reason


def test_skip_reason_for_another_commit_names_both_shas(tmp_path: Path) -> None:
    reason = corpus_skip_reason(_git(tmp_path, {"HEAD": f"{SHA_A}\n"}), expected_sha=SHA_B)

    assert reason is not None
    assert SHA_A in reason
    assert SHA_B in reason


def test_no_skip_for_the_expected_commit(tmp_path: Path) -> None:
    assert corpus_skip_reason(_git(tmp_path, {"HEAD": f"{SHA_A}\n"}), expected_sha=SHA_A) is None


# --- manifest format -----------------------------------------------------------------------------


def test_format_and_parse_round_trip() -> None:
    text = format_manifest(SHA_A, ROWS, generated=GENERATED)

    assert parse_manifest(text.encode()) == Manifest(SHA_A, ROWS)
    assert text.endswith("\n")
    assert "\r" not in text
    assert "# totals: 3 files, 22 bytes\n" in text
    assert f"# command: {REGENERATE_COMMAND}\n" in text


def test_rows_are_sorted_by_code_point_like_the_loader() -> None:
    rows = (_row("a-b.py"), _row("a.py"), _row("a/b.py"))  # '-' < '.' < '/'

    assert parse_manifest(format_manifest(SHA_A, rows, generated=GENERATED).encode()).rows == rows
    with pytest.raises(ManifestError, match="not sorted"):
        format_manifest(SHA_A, rows[::-1], generated=GENERATED)


@pytest.mark.parametrize(
    "path", ["a\tb.py", "a\nb.py", "a\x7fb.py", "/abs.py", "a/../b.py", "a//b.py", "a\\b.py", ""]
)
def test_generation_fails_loudly_on_unsafe_paths(path: str) -> None:
    with pytest.raises(ManifestError, match="cannot go in the manifest"):
        _row(path)


@pytest.mark.parametrize(
    ("language", "size", "sha", "match"),
    [
        ("Python", 1, HASH, "invalid language"),
        ("python", -1, HASH, "invalid size"),
        ("python", 1, "A" * 64, "64 lowercase hex"),
        ("python", 1, "0" * 63, "64 lowercase hex"),
    ],
)
def test_invalid_rows_are_rejected(language: str, size: int, sha: str, match: str) -> None:
    with pytest.raises(ManifestError, match=match):
        _row("a.py", language, size, sha)


def test_format_rejects_duplicates_empty_and_bad_commit() -> None:
    with pytest.raises(ManifestError, match="duplicate"):
        format_manifest(SHA_A, (_row("a.py"), _row("a.py")), generated=GENERATED)
    with pytest.raises(ManifestError, match="no rows"):
        format_manifest(SHA_A, (), generated=GENERATED)
    with pytest.raises(ManifestError, match="commit"):
        format_manifest("HEAD", ROWS, generated=GENERATED)


GOOD = format_manifest(SHA_A, ROWS, generated=GENERATED)
DATA_LINE = f"src/a.py\tpython\t5\t{HASH}"


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"\xef\xbb\xbf" + GOOD.encode(), "byte-order mark"),
        (GOOD.replace("\n", "\r\n").encode(), "carriage return"),
        (GOOD.encode() + b"\xff\n", "not UTF-8"),
        (GOOD[:-1].encode(), "end with a newline"),
        ((GOOD + "\n").encode(), "blank line"),
        ((GOOD + "# late comment\n").encode(), "comment after"),
        (GOOD.replace(DATA_LINE, "src/a.py\tpython\t5").encode(), "4 tab-separated"),
        (GOOD.replace(DATA_LINE, f"src/a.py\tpython\t05\t{HASH}").encode(), "decimal"),
        (GOOD.replace(DATA_LINE, f"src/a.py\tpython\t5\t{'Z' * 64}").encode(), "hex"),
        (GOOD.replace(DATA_LINE, f"src/../a.py\tpython\t5\t{HASH}").encode(), "manifest"),
        (GOOD.replace("src/b.py", "src/a.py").encode(), "duplicate"),
        (GOOD.replace("src/b.py", "src/0.py").encode(), "not sorted"),
        (
            GOOD.replace("# totals: 3 files, 22 bytes", "# totals: 3 files, 23 bytes").encode(),
            "totals",
        ),
        (GOOD.replace(f"# commit: {SHA_A}\n", "").encode(), "'commit' is missing"),
        (GOOD.replace(f"# commit: {SHA_A}", "# commit: main").encode(), "40 lowercase hex"),
        (GOOD.replace("# columns: path\t", "# columns: file\t").encode(), "columns must be"),
        ((GOOD + GOOD).encode(), "comment after"),
    ],
)
def test_parser_is_strict_and_actionable(data: bytes, match: str) -> None:
    with pytest.raises(ManifestError, match=match) as raised:
        parse_manifest(data)

    assert "manifest line " in str(raised.value)
    assert REGENERATE_COMMAND in str(raised.value)


def test_repeated_header_key_is_rejected() -> None:
    data = GOOD.replace("# totals:", f"# commit: {SHA_B}\n# totals:").encode()

    with pytest.raises(ManifestError, match="repeated"):
        parse_manifest(data)


# --- comparison ----------------------------------------------------------------------------------


def test_identical_rows_have_no_differences() -> None:
    assert compare_rows(ROWS, ROWS) == []


def test_every_kind_of_difference_is_reported() -> None:
    loaded = (
        _row("README.md", "text", 10),  # language differs
        _row("src/a.py", size=6, sha="1" * 64),  # size and hash differ
        _row("src/c.py"),  # extra; src/b.py is missing
    )

    assert compare_rows(ROWS, loaded) == [
        "missing from the load: src/b.py",
        "not in the manifest: src/c.py",
        "README.md: language differs (manifest 'markdown', load 'text')",
        "src/a.py: size_bytes differs (manifest 5, load 6)",
        f"src/a.py: sha256 differs (manifest {HASH!r}, load {'1' * 64!r})",
    ]


def test_order_difference_is_reported() -> None:
    assert compare_rows(ROWS, (ROWS[1], ROWS[0], ROWS[2])) == [
        "order differs at position 0: manifest 'README.md', load 'src/a.py'"
    ]


def test_rows_from_documents_reads_the_metadata_contract() -> None:
    doc = Document(
        id_="src/a.py",
        text="x = 1\n",
        metadata={
            "file_path": "src/a.py",
            "language": "python",
            "size_bytes": 6,
            "content_hash": HASH,
        },
    )

    assert rows_from_documents([doc]) == (_row("src/a.py", size=6),)


# --- the versioned manifest itself (no corpus needed) ------------------------------------------


def test_versioned_manifest_matches_the_decision_log() -> None:
    manifest = parse_manifest(MANIFEST_PATH.read_bytes())

    assert manifest.commit == FROZEN_CORPUS_SHA
    assert len(manifest.rows) == 88
    assert sum(row.size_bytes for row in manifest.rows) == 491_997
    files = Counter(row.language for row in manifest.rows)
    size = Counter[str]()
    for row in manifest.rows:
        size[row.language] += row.size_bytes
    assert dict(files) == {
        "markdown": 46,
        "python": 32,
        "yaml": 7,
        "toml": 1,
        "dockerfile": 1,
        "makefile": 1,
    }
    assert dict(size) == {
        "markdown": 263_570,
        "python": 192_670,
        "yaml": 24_349,
        "toml": 3_797,
        "dockerfile": 4_964,
        "makefile": 2_647,
    }
