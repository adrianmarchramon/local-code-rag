"""Tests for the Step 1 data: IngestionRules defaults, immutability, validation, reasons."""

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from src.errors import IngestionError, LocalStackError
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules, SecretNamePattern

EXPECTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}
EXPECTED_LOCKFILES = {
    "uv.lock",
    "dvc.lock",
    "poetry.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "Cargo.lock",
    "composer.lock",
}
# The `language` contract with Phases 2, 4 and 5 (rag_phase2.md Step 3 + decision Q3).
EXPECTED_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "ruby",
    "c",
    "cpp",
    "csharp",
    "php",
    "markdown",
    "yaml",
    "toml",
    "dockerfile",
    "makefile",
}
PHASE2_SYNTAX_AWARE = {"python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"}


# --- defaults match the recorded decisions ----------------------------------------------------


def test_default_extension_map_is_the_agreed_one() -> None:
    assert dict(IngestionRules().language_by_extension) == EXPECTED_EXTENSIONS


def test_default_filename_map_has_exactly_dockerfile_and_makefile() -> None:
    assert dict(IngestionRules().language_by_filename) == {
        "Dockerfile": "dockerfile",
        "Makefile": "makefile",
    }


def test_default_lockfiles_are_the_nine_agreed_names() -> None:
    assert IngestionRules().lockfile_names == EXPECTED_LOCKFILES


def test_default_builtin_patterns_follow_decision_d2() -> None:
    patterns = IngestionRules().builtin_patterns

    assert len(patterns) == 14
    assert ".git" in patterns  # no trailing slash: also the .git file of worktrees
    for root_only in ("/build/", "/dist/", "/target/", "/vendor/"):
        assert root_only in patterns
    assert not any(pattern.startswith("!") for pattern in patterns)


def test_default_secret_patterns_follow_decision_d4() -> None:
    patterns = {p.glob: p.exceptions for p in IngestionRules().secret_name_patterns}

    assert patterns == {
        ".env": (),
        ".env.*": ("*.example", "*.sample", "*.template"),
        "*.pem": (),
        "*.key": (),
        "id_rsa*": (),
        "*credentials*": (),
        "*secret*": (),
    }


def test_default_sizes() -> None:
    rules = IngestionRules()

    assert rules.head_bytes == 1024
    assert rules.max_gitignore_bytes == 1024 * 1024


# --- the language label contract --------------------------------------------------------------


def test_language_labels_are_the_exact_contract_set() -> None:
    assert IngestionRules().languages == EXPECTED_LANGUAGES


def test_language_labels_are_lowercase_plain_strings() -> None:
    for label in IngestionRules().languages:
        assert type(label) is str
        assert label == label.lower().strip()


def test_phase2_syntax_aware_languages_are_all_labels_we_emit() -> None:
    # Phase 2 routes these to CodeSplitter; every other label falls back to text chunking.
    assert IngestionRules().languages >= PHASE2_SYNTAX_AWARE


# --- immutability -------------------------------------------------------------------------------


def test_fields_cannot_be_reassigned() -> None:
    rules = IngestionRules()

    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.head_bytes = 4  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["language_by_extension", "language_by_filename"])
def test_maps_cannot_be_mutated(field_name: str) -> None:
    mapping: Any = getattr(IngestionRules(), field_name)

    with pytest.raises(TypeError):
        mapping[".x"] = "x"


def test_collections_are_immutable_types() -> None:
    rules = IngestionRules()

    assert isinstance(rules.lockfile_names, frozenset)
    assert isinstance(rules.builtin_patterns, tuple)
    assert isinstance(rules.secret_name_patterns, tuple)


def test_caller_mappings_are_copied_not_aliased() -> None:
    source = {".py": "python"}
    rules = IngestionRules(language_by_extension=source, language_by_filename={})

    source[".js"] = "javascript"

    assert dict(rules.language_by_extension) == {".py": "python"}


def test_custom_iterables_are_frozen() -> None:
    rules = IngestionRules(
        lockfile_names=["a.lock"],  # type: ignore[arg-type]
        builtin_patterns=["x/"],  # type: ignore[arg-type]
    )

    assert rules.lockfile_names == frozenset({"a.lock"})
    assert rules.builtin_patterns == ("x/",)


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language_by_extension": {"py": "python"}},  # no leading dot
        {"language_by_extension": {".PY": "python"}},  # not lowercase
        {"language_by_extension": {".tar.gz": "archive"}},  # only the last suffix is seen
        {"language_by_extension": {".py": "Python"}},  # label not lowercase
        {"language_by_extension": {".py": ""}},  # empty label
        {"language_by_extension": {".py": "py thon"}},  # whitespace in label
        {"language_by_extension": [(".py", "python")]},  # not a mapping
        {"language_by_filename": {"docker/Dockerfile": "dockerfile"}},  # not a bare name
        {"language_by_filename": {"": "x"}},
        {"lockfile_names": frozenset({"sub/uv.lock"})},
        {"builtin_patterns": ("",)},
        {"builtin_patterns": ("   ",)},
        {"builtin_patterns": ("# comment",)},
        {"builtin_patterns": ("!keep.py",)},  # built-ins never re-include
        {"builtin_patterns": ("a\nb",)},
        {"secret_name_patterns": (".env",)},  # not a SecretNamePattern
        {"head_bytes": 0},
        {"head_bytes": -1},
        {"head_bytes": True},
        {"max_gitignore_bytes": 0},
    ],
)
def test_invalid_rules_fail_at_construction(kwargs: Mapping[str, Any]) -> None:
    with pytest.raises(IngestionError):
        IngestionRules(**kwargs)


@pytest.mark.parametrize(
    ("glob", "exceptions"),
    [("", ()), ("*.PEM", ()), ("dir/*.pem", ()), (".env.*", ("*.EXAMPLE",))],
)
def test_invalid_secret_patterns_fail_at_construction(
    glob: str, exceptions: tuple[str, ...]
) -> None:
    with pytest.raises(IngestionError):
        SecretNamePattern(glob, exceptions)


def test_ingestion_errors_belong_to_the_project_hierarchy() -> None:
    assert issubclass(IngestionError, LocalStackError)


# --- reason codes are stable --------------------------------------------------------------------


def test_discard_reason_codes_are_stable() -> None:
    assert [reason.value for reason in DiscardReason] == [
        "builtin_pattern",
        "corpus_rules",
        "gitignore",
        "lockfile",
        "secret_name",
        "unmapped_type",
        "binary",
        "unreadable",
        "not_regular_file",
    ]


def test_discard_reasons_are_plain_strings() -> None:
    assert isinstance(DiscardReason.LOCKFILE, str)
    assert str(DiscardReason.LOCKFILE) == "lockfile"
    assert f"{DiscardReason.SECRET_NAME}" == "secret_name"
