"""Tests for the Step 4 orchestrator, the load report and the CLI."""

import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from llama_index.core import Document

from src.errors import NoDocumentsError, RepositoryRootError
from src.ingestion import loader
from src.ingestion.loader import (
    ExitCode,
    LoadResult,
    load_repository,
    load_repository_with_report,
    main,
)
from src.ingestion.reasons import DiscardReason
from src.ingestion.report import Tally
from src.ingestion.rules import IngestionRules

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sample_repo(root: Path) -> Path:
    _write(root, "README.md", b"# Project\n")
    _write(root, "src/app.py", b"def main():\n    return 0\n")
    _write(root, "src/util.py", b"def helper():\n    return 1\n")
    _write(root, "src/copy_of_app.py", b"def main():\n    return 0\n")  # duplicate
    _write(root, "config.yaml", b"key: value\n")
    _write(root, "uv.lock", b"version = 1\n")
    _write(root, ".env", b"TOKEN=invented\n")
    _write(root, "data.json", b"{}\n")
    _write(root, "logo.py", b"\x89PNG\x00\x00")
    _write(root, "empty.py", b"")
    _write(root, "latin1.py", b"caf\xe9 = 1\n")
    _write(root, "node_modules/x/index.js", b"module.exports = {}\n")
    _write(root, "deploy/model/m.py", b"stale = True\n")
    (root / "link.py").symlink_to(root / "src" / "app.py")
    return root


def _snapshot(doc: Document) -> tuple[object, ...]:
    return (
        doc.doc_id,
        doc.text,
        dict(doc.metadata),
        list(doc.excluded_embed_metadata_keys),
        list(doc.excluded_llm_metadata_keys),
    )


def _load(root: Path) -> LoadResult:
    return load_repository_with_report(root, rules_file=RULES_FILE)


# --- wiring ---------------------------------------------------------------------------------------


def test_full_load_keeps_code_and_explains_every_exclusion(tmp_path: Path) -> None:
    result = _load(_sample_repo(tmp_path))

    assert [d.doc_id for d in result.documents] == [
        "README.md",
        "config.yaml",
        "src/app.py",
        "src/util.py",
    ]
    assert {d.rel_path: d.reason for d in result.discarded} == {
        ".env": DiscardReason.SECRET_NAME,
        "data.json": DiscardReason.UNMAPPED_TYPE,
        "empty.py": DiscardReason.EMPTY,
        "latin1.py": DiscardReason.UNDECODABLE,
        "link.py": DiscardReason.SYMLINK,
        "logo.py": DiscardReason.BINARY,
        "src/copy_of_app.py": DiscardReason.DUPLICATE,
        "uv.lock": DiscardReason.LOCKFILE,
    }
    assert {d.rel_path: d.reason for d in result.pruned} == {
        "deploy/model": DiscardReason.CORPUS_RULES,
        "node_modules": DiscardReason.BUILTIN_PATTERN,
    }


def test_report_is_consistent_with_the_documents(tmp_path: Path) -> None:
    result = _load(_sample_repo(tmp_path))
    report = result.report
    sizes = [int(d.metadata["size_bytes"]) for d in result.documents]

    assert report.included_files == len(result.documents) == 4
    assert report.included_bytes == sum(sizes)
    assert sum(t.files for t in report.by_language.values()) == report.included_files
    assert sum(t.bytes for t in report.by_top_level.values()) == report.included_bytes
    assert dict(report.by_language) == {
        "markdown": Tally(1, 10),
        "python": Tally(2, 52),
        "yaml": Tally(1, 11),
    }
    assert list(report.by_top_level) == [".", "src"]
    assert sum(report.discarded_files.values()) == len(result.discarded)
    assert dict(report.pruned_dirs) == {"builtin_pattern": 1, "corpus_rules": 1}
    assert [(d.discarded, d.canonical) for d in report.duplicates] == [
        ("src/copy_of_app.py", "src/app.py")
    ]
    assert report.gitignores_read == 0


def test_report_is_json_ready_and_has_no_absolute_paths(tmp_path: Path) -> None:
    result = _load(_sample_repo(tmp_path))

    rendered = json.dumps(result.report.to_dict()) + result.report.render_text()

    assert str(tmp_path) not in rendered
    assert json.loads(json.dumps(result.report.to_dict()))["included"] == {
        "files": 4,
        "bytes": 73,
    }


def test_two_loads_are_identical(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)

    first, second = _load(root), _load(root)

    assert [_snapshot(d) for d in first.documents] == [_snapshot(d) for d in second.documents]
    assert (first.report, first.discarded, first.pruned) == (
        second.report,
        second.discarded,
        second.pruned,
    )


def test_load_repository_takes_just_a_path(tmp_path: Path) -> None:
    _write(tmp_path, "main.py", b"print('hi')\n")

    documents = load_repository(tmp_path)  # default rules and INGEST_IGNORE_FILE

    assert [d.doc_id for d in documents] == ["main.py"]
    assert isinstance(documents, list)


def test_custom_rules_are_used(tmp_path: Path) -> None:
    _write(tmp_path, "big.py", b"x = 1\n" * 10)
    rules = IngestionRules(max_file_bytes=16)

    result = load_repository_with_report(tmp_path, rules=rules, rules_file=RULES_FILE)

    assert {d.rel_path: d.reason for d in result.discarded} == {"big.py": DiscardReason.TOO_LARGE}


def test_loading_writes_nothing(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path / "repo")

    def tree() -> dict[str, tuple[int, int]]:
        return {
            str(p.relative_to(tmp_path)): (p.lstat().st_size, p.lstat().st_mtime_ns)
            for p in tmp_path.rglob("*")
        }

    before = tree()
    _load(root)

    assert tree() == before


# --- empty loads (decision D8) and fatal errors --------------------------------------------------


def test_no_documents_raises_in_load_repository_only(tmp_path: Path) -> None:
    _write(tmp_path, "data.json", b"{}\n")
    _write(tmp_path, "notes.txt", b"hello\n")

    result = _load(tmp_path)

    assert result.documents == ()
    assert result.report.included_files == 0
    with pytest.raises(NoDocumentsError, match="no file passed the filters"):
        load_repository(tmp_path, rules_file=RULES_FILE)


def test_missing_root_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(RepositoryRootError):
        load_repository(tmp_path / "missing", rules_file=RULES_FILE)


# --- CLI ------------------------------------------------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, stdout=out)
    return code, out.getvalue()


def test_cli_prints_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run(["--repo", str(_sample_repo(tmp_path))])

    assert code == ExitCode.OK
    assert "Included: 4 files, 73 bytes" in output
    assert output.endswith("RESULT: OK (exit 0 OK)\n")
    logs = capsys.readouterr().err
    assert '"event": "ingestion.loaded"' in logs
    assert "def main" not in logs  # never file contents


def test_cli_list_is_sorted(tmp_path: Path) -> None:
    code, output = _run(["--repo", str(_sample_repo(tmp_path)), "--list"])

    assert code == ExitCode.OK
    assert output.splitlines() == ["README.md", "config.yaml", "src/app.py", "src/util.py"]


def test_cli_json_is_the_report(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)

    code, output = _run(["--repo", str(root), "--json"])

    assert code == ExitCode.OK
    assert json.loads(output) == _load(root).report.to_dict()


def test_cli_uses_the_given_rules_file(tmp_path: Path) -> None:
    rules = tmp_path / "rules.ignore"
    rules.write_text("/src/\n", encoding="utf-8")
    repo = _sample_repo(tmp_path / "repo")

    code, output = _run(["--repo", str(repo), "--rules-file", str(rules), "--list"])

    # The given file replaces the default one: src/ is excluded, deploy/model/ is not.
    assert code == ExitCode.OK
    assert output.splitlines() == ["README.md", "config.yaml", "deploy/model/m.py"]


def test_cli_missing_root_exits_20_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _run(["--repo", str(tmp_path / "missing")])

    assert code == ExitCode.INGESTION
    assert "does not exist" in output
    assert "Traceback" not in output + capsys.readouterr().err


def test_cli_missing_rules_file_exits_20(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", b"x = 1\n")

    code, output = _run(["--repo", str(tmp_path), "--rules-file", str(tmp_path / "none")])

    assert code == ExitCode.INGESTION
    assert "does not exist" in output


def test_cli_no_documents_exits_21(tmp_path: Path) -> None:
    _write(tmp_path, "data.json", b"{}\n")

    code, output = _run(["--repo", str(tmp_path)])

    assert code == ExitCode.NO_DOCUMENTS
    assert "no file passed the filters" in output


def test_cli_maps_invalid_configuration_to_exit_10_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REPO_PATH", "relative/path")
    monkeypatch.delitem(sys.modules, "src.config")

    code, output = _run([])

    assert code == ExitCode.CONFIGURATION
    assert "REPO_PATH" in output
    assert "Traceback" not in output + capsys.readouterr().err


def test_cli_blocks_and_reports_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "a.py", b"x = 1\n")
    real = loader.load_repository_with_report

    def leaky(*args: object, **kwargs: object) -> LoadResult:
        socket.getaddrinfo("example.com", 443)  # blocked by the CLI's own guard
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(loader, "load_repository_with_report", leaky)

    code, output = _run(["--repo", str(tmp_path)])

    assert code == ExitCode.EXTERNAL_NETWORK
    assert "example.com" in output


def test_exit_codes_are_stable() -> None:
    assert {code.name: code.value for code in ExitCode} == {
        "OK": 0,
        "UNEXPECTED": 1,
        "CONFIGURATION": 10,
        "EXTERNAL_NETWORK": 14,
        "INGESTION": 20,
        "NO_DOCUMENTS": 21,
    }


def test_module_runs_as_a_script(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", b"x = 1\n")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}

    result = subprocess.run(  # noqa: S603 (this interpreter, constant arguments)
        [sys.executable, "-m", "src.ingestion.loader", "--repo", str(tmp_path), "--list"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "a.py\n"
    assert "RuntimeWarning" not in result.stderr
