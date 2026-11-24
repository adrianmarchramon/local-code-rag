"""Static audit: no string constant in ``src/`` may contain a URL with a non-loopback host.

Docstrings are skipped (documentation may cite external references); every other string
literal, including f-string parts, is checked. Runtime code must only talk to the local
Ollama, whose address comes from configuration.
"""

import ast
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from src.network_guard import is_loopback_host

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
URL_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>`)]+")
# Hosts that may appear in src/ string literals although they are not loopback.
# Keep it empty; every addition needs a one-line justification next to it.
ALLOWED_NON_LOOPBACK_HOSTS: frozenset[str] = frozenset()


def _docstring_nodes(tree: ast.AST) -> set[int]:
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, holders) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def find_external_urls(source: str, filename: str) -> list[str]:
    """Return ``file:line: url`` for each non-loopback URL in non-docstring literals."""
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        for match in URL_PATTERN.finditer(node.value):
            url = match.group()
            try:
                host = urlsplit(url).hostname
            except ValueError:
                findings.append(f"{filename}:{node.lineno}: {url} (unparseable)")
                continue
            if host and not is_loopback_host(host) and host not in ALLOWED_NON_LOOPBACK_HOSTS:
                findings.append(f"{filename}:{node.lineno}: {url}")
    return findings


def test_src_has_no_non_loopback_urls() -> None:
    sources = sorted(SRC_DIR.rglob("*.py"))
    assert sources, f"no Python sources found under {SRC_DIR}"

    findings = [
        finding
        for path in sources
        for finding in find_external_urls(path.read_text(encoding="utf-8"), str(path))
    ]

    assert findings == []


@pytest.mark.parametrize(
    ("source", "expected_count"),
    [
        ('URL = "https://example.com/api"\n', 1),
        ('URL = f"http://10.0.0.5:{1}/x"\n', 1),
        ('URL = "http://localhost:11434"\n', 0),
        ('URL = "http://127.0.0.1:11434"\n', 0),
        ('URL = "http://[::1]:11434"\n', 0),
        ('"""See https://example.com for details."""\n', 0),
        ('def f() -> None:\n    """Docs: https://example.com."""\n', 0),
        ('MSG = "no url here"\n', 0),
    ],
)
def test_audit_detects_external_urls_and_ignores_docstrings(
    source: str, expected_count: int
) -> None:
    assert len(find_external_urls(source, "sample.py")) == expected_count
