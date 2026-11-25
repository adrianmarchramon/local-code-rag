"""Import-time privacy: the ingestion stack must not touch the network when imported.

The autouse guard in conftest.py starts after collection, so it cannot see what a module
does while being imported. This test starts a fresh, isolated interpreter (``-I``),
activates :class:`NetworkGuard` *before* importing anything else, imports the modules in
``IMPORTS_UNDER_GUARD``, runs the Step 1-3 filters on a small synthetic repository, and
reports the guard's violations. Step 4 adds ``llama_index.core`` to the list.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IMPORTS_UNDER_GUARD: tuple[str, ...] = (
    "pathspec",
    "src.ingestion.reasons",
    "src.ingestion.rules",
    "src.ingestion.ignore",
    "src.ingestion.classify",
    "src.ingestion.binary",
)

_CHILD = textwrap.dedent(
    """
    import importlib, json, sys, tempfile
    from pathlib import Path

    repo_root, modules = sys.argv[1], json.loads(sys.argv[2])
    sys.path.insert(0, repo_root)
    from src.network_guard import NetworkGuard

    already_loaded = sorted(m for m in modules if m in sys.modules)
    with NetworkGuard() as guard:
        for name in modules:
            importlib.import_module(name)
        from src.ingestion.binary import probe_head
        from src.ingestion.classify import classify_name
        from src.ingestion.ignore import IgnorePolicy
        from src.ingestion.rules import IngestionRules

        rules = IngestionRules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            (root / ".gitignore").write_text("*.log\\n")
            (root / "sub" / ".gitignore").write_text("!keep.log\\n")
            (root / "sub" / "a.py").write_text("x = 1\\n")
            policy = IgnorePolicy(root, rules, Path(repo_root) / "config" / "ingest.ignore")
            verdicts = [
                policy.is_ignored("sub/keep.log", is_dir=False).ignored,
                policy.is_ignored("deploy/model/x.py", is_dir=False).ignored,
                classify_name("sub/a.py", rules).language,
                probe_head(root / "sub" / "a.py", rules.head_bytes).value,
            ]
    print(json.dumps({
        "already_loaded": already_loaded,
        "violations": [repr(v) for v in guard.violations],
        "verdicts": verdicts,
    }))
    """
)


def test_ingestion_imports_and_filters_make_no_network_access(tmp_path: Path) -> None:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LC_ALL": "C.UTF-8"}

    # S603: trusted input only (this interpreter, a constant script, repo-controlled args).
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-c",
            _CHILD,
            str(REPO_ROOT),
            json.dumps(IMPORTS_UNDER_GUARD),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["already_loaded"] == []  # nothing was imported before the guard
    assert report["violations"] == []
    assert report["verdicts"] == [False, True, "python", "text"]
