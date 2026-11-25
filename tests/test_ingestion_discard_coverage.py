"""Meta-test: every DiscardReason is produced end to end and its code asserted.

Each reason has a minimal scenario that runs through the public, report-returning loader
(walker, filters, reader and deduplication together) and must end with that reason on
the expected path. ``DiscardReason`` is append-only: adding a member without a scenario
here fails the suite.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.ingestion.loader import load_repository_with_report
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import DEFAULT_MAX_FILE_BYTES

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _gitignored(root: Path) -> None:
    _write(root, ".gitignore", b"generated.py\n")
    _write(root, "generated.py", b"x = 1\n")


def _unreadable(root: Path) -> None:
    _write(root, "locked.py", b"x = 1\n")
    (root / "locked.py").chmod(0)


def _symlink(root: Path) -> None:
    _write(root, "main.py", b"x = 1\n")
    (root / "link.py").symlink_to(root / "main.py")


def _duplicate(root: Path) -> None:
    _write(root, "a.py", b"same = 1\n")
    _write(root, "b.py", b"same = 1\n")


@dataclass(frozen=True, slots=True)
class Scenario:
    """Builds a tree whose ``rel_path`` (a file, or a pruned directory) gets the reason."""

    build: Callable[[Path], None]
    rel_path: str
    pruned: bool = False


SCENARIOS: dict[DiscardReason, Scenario] = {
    DiscardReason.BUILTIN_PATTERN: Scenario(
        lambda r: _write(r, "node_modules/x/index.js", b"x\n"), "node_modules", pruned=True
    ),
    DiscardReason.CORPUS_RULES: Scenario(
        lambda r: _write(r, "project_context/notes.md", b"# x\n"), "project_context", pruned=True
    ),
    DiscardReason.GITIGNORE: Scenario(_gitignored, "generated.py"),
    DiscardReason.LOCKFILE: Scenario(lambda r: _write(r, "uv.lock", b"version = 1\n"), "uv.lock"),
    DiscardReason.SECRET_NAME: Scenario(lambda r: _write(r, ".env", b"K=invented\n"), ".env"),
    DiscardReason.UNMAPPED_TYPE: Scenario(lambda r: _write(r, "data.json", b"{}\n"), "data.json"),
    DiscardReason.BINARY: Scenario(lambda r: _write(r, "logo.py", b"\x89PNG\x00"), "logo.py"),
    DiscardReason.UNREADABLE: Scenario(_unreadable, "locked.py"),
    DiscardReason.NOT_REGULAR_FILE: Scenario(lambda r: os.mkfifo(r / "pipe.py"), "pipe.py"),
    DiscardReason.SYMLINK: Scenario(_symlink, "link.py"),
    DiscardReason.TOO_LARGE: Scenario(
        lambda r: _write(r, "big.py", b"#" * (DEFAULT_MAX_FILE_BYTES + 1)), "big.py"
    ),
    DiscardReason.UNDECODABLE: Scenario(
        lambda r: _write(r, "latin1.py", b"caf\xe9 = 1\n"), "latin1.py"
    ),
    DiscardReason.EMPTY: Scenario(lambda r: _write(r, "empty.py", b""), "empty.py"),
    DiscardReason.DUPLICATE: Scenario(_duplicate, "b.py"),
    DiscardReason.INVALID_NAME: Scenario(
        lambda r: _write(r, "bad\\name.py", b"x = 1\n"),
        "bad\\\\name.py",  # shown escaped
    ),
}


def test_every_discard_reason_has_a_scenario() -> None:
    missing = set(DiscardReason) - set(SCENARIOS)
    assert not missing, f"add an end-to-end scenario to SCENARIOS for: {sorted(missing)}"


@pytest.mark.parametrize("reason", list(DiscardReason), ids=str)
def test_reason_is_produced_through_the_loader(tmp_path: Path, reason: DiscardReason) -> None:
    if reason is DiscardReason.UNREADABLE and os.geteuid() == 0:
        pytest.skip("running as root: file permissions are ignored, so unreadable cannot occur")
    scenario = SCENARIOS[reason]
    scenario.build(tmp_path)

    result = load_repository_with_report(tmp_path, rules_file=RULES_FILE)

    recorded = result.pruned if scenario.pruned else result.discarded
    assert {d.rel_path: d.reason for d in recorded}.get(scenario.rel_path) is reason
    counts = result.report.pruned_dirs if scenario.pruned else result.report.discarded_files
    assert counts[reason.value] == 1
