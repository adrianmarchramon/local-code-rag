"""Steps 1-3 together on a synthetic, corpus-shaped tree (invented content only).

The helper below is test-only: it lists every file and applies the filters in the agreed
order (ignore policy -> name classification -> binary probe), recording the first reason
that rejects each file. The production walker, with directory pruning, is Step 4.
"""

import os
from collections import Counter
from pathlib import Path

from src.ingestion.binary import probe_head
from src.ingestion.classify import classify_name
from src.ingestion.ignore import IgnorePolicy
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"

INCLUDED = {
    "README.md": "markdown",
    "docs/a.md": "markdown",
    "docs/decisions/0001-adr.md": "markdown",
    "src/a.py": "python",
    "src/__init__.py": "python",
    "tests/test_a.py": "python",
    "pipelines/p.py": "python",
    "scripts/s.py": "python",
    "main.py": "python",
    "dvc.yaml": "yaml",
    ".github/workflows/ci.yml": "yaml",
    "pyproject.toml": "toml",
    "Dockerfile": "dockerfile",
    "docker/Dockerfile": "dockerfile",
    "Makefile": "makefile",
}
EXCLUDED = {
    "deploy/model/m.py": DiscardReason.CORPUS_RULES,
    "deploy/model/conda.yaml": DiscardReason.CORPUS_RULES,
    "deploy/model/uv.lock": DiscardReason.CORPUS_RULES,  # the first filter gives the reason
    "project_context/c.md": DiscardReason.CORPUS_RULES,
    "node_modules/x/index.js": DiscardReason.BUILTIN_PATTERN,
    "__pycache__/a.cpython-311.pyc": DiscardReason.BUILTIN_PATTERN,
    "src/__pycache__/a.cpython-311.pyc": DiscardReason.BUILTIN_PATTERN,
    "app.log": DiscardReason.GITIGNORE,  # root .gitignore
    "data/raw/ignored.py": DiscardReason.GITIGNORE,  # nested .gitignore only
    "uv.lock": DiscardReason.LOCKFILE,
    "dvc.lock": DiscardReason.LOCKFILE,
    "pnpm-lock.yaml": DiscardReason.LOCKFILE,
    ".env": DiscardReason.SECRET_NAME,
    "id_rsa": DiscardReason.SECRET_NAME,
    "notebooks/n.ipynb": DiscardReason.UNMAPPED_TYPE,
    "data/raw/x.csv": DiscardReason.UNMAPPED_TYPE,
    "docs/img/a.png": DiscardReason.UNMAPPED_TYPE,
    "docker/.env.example": DiscardReason.UNMAPPED_TYPE,
    ".gitignore": DiscardReason.UNMAPPED_TYPE,
    "data/raw/.gitignore": DiscardReason.UNMAPPED_TYPE,
    "src/disguised.py": DiscardReason.BINARY,
    "src/link.py": DiscardReason.NOT_REGULAR_FILE,
}


def _make_corpus_shaped_repo(root: Path) -> None:
    for rel in INCLUDED:
        _write(root, rel, b"# invented content\nvalue = 1\n")
    for rel in EXCLUDED:
        if rel not in {"src/disguised.py", "src/link.py", ".gitignore", "data/raw/.gitignore"}:
            _write(root, rel, b"invented\n")
    _write(root, ".gitignore", b"*.log\n")
    _write(root, "data/raw/.gitignore", b"/ignored.py\n")
    _write(root, "docs/img/a.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    _write(root, "src/disguised.py", b"looks like code\x00but is binary")
    (root / "src" / "link.py").symlink_to(root / "src" / "a.py")


def _write(root: Path, rel: str, content: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _filter_tree(
    root: Path, rules: IngestionRules, rules_file: Path
) -> tuple[dict[str, str], dict[str, DiscardReason]]:
    """Test-only composition of Steps 1-3 over every file below ``root``."""
    policy = IgnorePolicy(root, rules, rules_file)
    kept: dict[str, str] = {}
    dropped: dict[str, DiscardReason] = {}
    for current, _dirs, files in os.walk(root):  # does not follow directory symlinks
        for name in files:
            rel = (Path(current) / name).relative_to(root).as_posix()
            ignore = policy.is_ignored(rel, is_dir=False)
            if ignore.reason is not None:
                dropped[rel] = ignore.reason
                continue
            named = classify_name(rel, rules)
            if named.reason is not None:
                dropped[rel] = named.reason
                continue
            probe = probe_head(root / rel, rules.head_bytes).discard_reason
            if probe is not None:
                dropped[rel] = probe
                continue
            assert named.language is not None
            kept[rel] = named.language
    return kept, dropped


def test_steps_1_to_3_keep_relevant_code_and_drop_noise_with_reasons(tmp_path: Path) -> None:
    _make_corpus_shaped_repo(tmp_path)

    kept, dropped = _filter_tree(tmp_path, IngestionRules(), RULES_FILE)

    assert kept == INCLUDED
    assert dropped == EXCLUDED
    assert Counter(dropped.values()) == {
        DiscardReason.CORPUS_RULES: 4,
        DiscardReason.BUILTIN_PATTERN: 3,
        DiscardReason.GITIGNORE: 2,
        DiscardReason.LOCKFILE: 3,
        DiscardReason.SECRET_NAME: 2,
        DiscardReason.UNMAPPED_TYPE: 6,
        DiscardReason.BINARY: 1,
        DiscardReason.NOT_REGULAR_FILE: 1,
    }


def test_excluded_directories_can_be_pruned_before_descending(tmp_path: Path) -> None:
    _make_corpus_shaped_repo(tmp_path)
    policy = IgnorePolicy(tmp_path, IngestionRules(), RULES_FILE)

    pruned = {
        rel: policy.is_ignored(rel, is_dir=True).reason
        for rel in ("deploy/model", "project_context", "node_modules", "src/__pycache__")
    }

    assert pruned == {
        "deploy/model": DiscardReason.CORPUS_RULES,
        "project_context": DiscardReason.CORPUS_RULES,
        "node_modules": DiscardReason.BUILTIN_PATTERN,
        "src/__pycache__": DiscardReason.BUILTIN_PATTERN,
    }
    assert not policy.is_ignored("deploy", is_dir=True).ignored
    assert not policy.is_ignored("data/raw", is_dir=True).ignored


def test_filtering_is_deterministic(tmp_path: Path) -> None:
    _make_corpus_shaped_repo(tmp_path)

    first = _filter_tree(tmp_path, IngestionRules(), RULES_FILE)
    second = _filter_tree(tmp_path, IngestionRules(), RULES_FILE)

    assert first == second
