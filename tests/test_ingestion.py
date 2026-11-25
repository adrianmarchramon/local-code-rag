"""The two tests of rag_phase1.md Step 7, adapted to the real API (invented content only).

The doc's versions assert only absence, and do not isolate what they claim to test:
``logo.png`` falls on its unmapped extension before the binary check runs, and
``secret.py`` is dropped by ``.gitignore`` before its secret-like name is looked at. Here
each noise file has a single cause, the reason is asserted through the report-returning
loader, and ``load_repository`` (the API later phases call) gives the loaded documents.
"""

from pathlib import Path

from src.ingestion.loader import load_repository, load_repository_with_report
from src.ingestion.reasons import DiscardReason

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"


def _make_sample_repo(root: Path) -> Path:
    # Relevant code: loaded.
    (root / "main.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    # Third-party dependency: pruned as a built-in pattern, never descended into.
    (root / "node_modules").mkdir()
    (root / "node_modules" / "lib.js").write_text("module.exports = {}\n", encoding="utf-8")
    # Ignored by the project's .gitignore only: an ordinary, mapped, non-secret name.
    (root / ".gitignore").write_text("generated_client.py\n", encoding="utf-8")
    (root / "generated_client.py").write_text("CLIENT = None\n", encoding="utf-8")
    # Disguised binary: a mapped extension, so only the NUL check can drop it.
    (root / "logo.py").write_bytes(b"\x89PNG\x00\x00data")
    return root


def test_loads_only_relevant_code(tmp_path: Path) -> None:
    repo = _make_sample_repo(tmp_path)

    documents = load_repository(repo, rules_file=RULES_FILE)
    result = load_repository_with_report(repo, rules_file=RULES_FILE)

    assert [d.metadata["file_path"] for d in documents] == ["main.py"]
    assert {d.rel_path: d.reason for d in result.discarded} == {
        ".gitignore": DiscardReason.UNMAPPED_TYPE,
        "generated_client.py": DiscardReason.GITIGNORE,
        "logo.py": DiscardReason.BINARY,
    }
    assert {d.rel_path: d.reason for d in result.pruned} == {
        "node_modules": DiscardReason.BUILTIN_PATTERN
    }


def test_attaches_correct_metadata(tmp_path: Path) -> None:
    repo = _make_sample_repo(tmp_path)
    source = (repo / "main.py").read_bytes()

    main = next(d for d in load_repository(repo, rules_file=RULES_FILE) if d.doc_id == "main.py")

    assert main.metadata["language"] == "python"
    assert main.metadata["num_lines"] == 2
    assert main.metadata["size_bytes"] == len(source)
    assert main.metadata["directory"] == ""
