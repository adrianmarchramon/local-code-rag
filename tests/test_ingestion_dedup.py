"""Tests for Step 4 deduplication: canonical path, ordering pitfall, survivors only."""

from pathlib import Path

from src.ingestion.dedup import Duplicate, find_duplicates
from src.ingestion.loader import load_repository_with_report
from src.ingestion.reasons import DiscardReason

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_two_copies_keep_the_smallest_path() -> None:
    assert find_duplicates({"src/b.py": "h1", "src/a.py": "h1", "src/c.py": "h2"}) == (
        Duplicate("src/b.py", "src/a.py"),
    )


def test_three_copies_share_one_canonical() -> None:
    assert find_duplicates({"z.py": "h", "m.py": "h", "a.py": "h"}) == (
        Duplicate("m.py", "a.py"),
        Duplicate("z.py", "a.py"),
    )


def test_canonical_is_chosen_by_full_posix_string() -> None:
    # '-' < '.' < '/': a per-directory ordering would pick a different file.
    result = find_duplicates({"a.py": "h", "a/b.py": "h", "a-b.py": "h"})

    assert {d.canonical for d in result} == {"a-b.py"}
    assert [d.discarded for d in result] == ["a.py", "a/b.py"]


def test_no_duplicates() -> None:
    assert find_duplicates({"a.py": "h1", "b.py": "h2"}) == ()
    assert find_duplicates({}) == ()


def test_copies_with_different_extensions_are_resolved_deterministically(tmp_path: Path) -> None:
    _write(tmp_path, "notes.py", b"# same bytes\n")
    _write(tmp_path, "notes.md", b"# same bytes\n")

    result = load_repository_with_report(tmp_path, rules_file=RULES_FILE)

    assert [(d.doc_id, d.metadata["language"]) for d in result.documents] == [
        ("notes.md", "markdown")
    ]
    assert result.report.duplicates == (Duplicate("notes.py", "notes.md"),)


def test_a_copy_excluded_by_a_path_rule_never_hides_a_legitimate_one(tmp_path: Path) -> None:
    same = b"def predict():\n    return 1\n"
    _write(tmp_path, "deploy/model/code/predict.py", same)  # excluded by the rules file
    _write(tmp_path, "build/predict.py", same)  # excluded by a built-in pattern
    _write(tmp_path, ".gitignore", b"/generated/\n")
    _write(tmp_path, "generated/predict.py", same)  # excluded by .gitignore
    _write(tmp_path, "src/predict.py", same)

    result = load_repository_with_report(tmp_path, rules_file=RULES_FILE)

    assert [d.doc_id for d in result.documents] == ["src/predict.py"]  # .gitignore: unmapped
    assert result.report.duplicates == ()


def test_empty_files_are_empty_not_duplicates(tmp_path: Path) -> None:
    _write(tmp_path, "a/__init__.py", b"")
    _write(tmp_path, "b/__init__.py", b"")
    _write(tmp_path, "c/__init__.py", b"  \n")
    _write(tmp_path, "main.py", b"x = 1\n")

    result = load_repository_with_report(tmp_path, rules_file=RULES_FILE)

    assert result.report.duplicates == ()
    assert {d.rel_path: d.reason for d in result.discarded} == {
        "a/__init__.py": DiscardReason.EMPTY,
        "b/__init__.py": DiscardReason.EMPTY,
        "c/__init__.py": DiscardReason.EMPTY,
    }
