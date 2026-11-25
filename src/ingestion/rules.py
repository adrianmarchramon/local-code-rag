"""Ingestion filtering rules as immutable data (Phase 1, Step 1: relevant code vs noise).

Adding a language, lockfile or pattern is a data change here, never a code change in the
filters. Instances are truly immutable: the dataclass is frozen, mappings are read-only
proxies over private copies, sets are frozensets and sequences are tuples. Invalid data
fails at construction with :class:`src.errors.IngestionError`.

The ``language`` labels are a contract with later phases: Phase 2 compares them with its
syntax-aware set, Phase 4 filters on them. They are lowercase plain ``str`` values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from src.errors import IngestionError

# Q3 (owner decision, 2026-11-25): the rag_phase1.md Step 3 map plus yaml and toml.
DEFAULT_LANGUAGE_BY_EXTENSION: Final[Mapping[str, str]] = MappingProxyType(
    {
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
)
# Matched by exact, case-sensitive file name (these files have no extension).
DEFAULT_LANGUAGE_BY_FILENAME: Final[Mapping[str, str]] = MappingProxyType(
    {"Dockerfile": "dockerfile", "Makefile": "makefile"}
)
# Q3: generated dependency pins; large and of no use for answering questions about code.
DEFAULT_LOCKFILE_NAMES: Final[frozenset[str]] = frozenset(
    {
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
)
# D2: always-on generic noise, in gitignore syntax. Unambiguous names match at any depth;
# names that can also be legitimate source folders (build, dist, target, vendor) match at
# the repository root only, leaving nested ones to the project's .gitignore.
DEFAULT_BUILTIN_PATTERNS: Final[tuple[str, ...]] = (
    ".git",  # no trailing slash: also matches the .git *file* of worktrees and submodules
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "site-packages/",
    ".next/",
    "/build/",
    "/dist/",
    "/target/",
    "/vendor/",
)
DEFAULT_HEAD_BYTES: Final = 1024  # rag_phase1.md Step 3: NUL check on the first KiB
DEFAULT_MAX_GITIGNORE_BYTES: Final = 1024 * 1024  # D3: larger corpus .gitignore files fail
DEFAULT_MAX_FILE_BYTES: Final = 1024 * 1024  # D7: larger files are skipped as too_large

_EXTENSION_RE: Final = re.compile(r"\.[a-z0-9_+-]+")
_LABEL_RE: Final = re.compile(r"[a-z][a-z0-9_+-]*")


def _check_file_name(value: object, what: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(char in value for char in "/\\\0")
    ):
        raise IngestionError(
            f"Invalid {what} {value!r}: expected a bare file name (no path separators). "
            "Fix the value in IngestionRules."
        )
    return value


def _check_glob(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.lower() or "/" in value:
        raise IngestionError(
            f"Invalid {what} {value!r}: expected a non-empty, lowercase file-name glob "
            "without '/' (it is matched against the lowercased base name)."
        )
    return value


@dataclass(frozen=True, slots=True)
class SecretNamePattern:
    """A base-name glob for files that look like secrets, with globs that exempt a match.

    Matching is case-insensitive: both the glob and its exceptions are lowercase and are
    compared with the lowercased base name.
    """

    glob: str
    exceptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_glob(self.glob, "secret-name glob")
        object.__setattr__(self, "exceptions", tuple(self.exceptions))
        for exception in self.exceptions:
            _check_glob(exception, f"exception of secret-name glob {self.glob!r}")


# D4: excluded and counted before the language check, so a mapped extension cannot let a
# secret-looking file in. The template exceptions apply to the .env family only.
DEFAULT_SECRET_NAME_PATTERNS: Final[tuple[SecretNamePattern, ...]] = (
    SecretNamePattern(".env"),
    SecretNamePattern(".env.*", exceptions=("*.example", "*.sample", "*.template")),
    SecretNamePattern("*.pem"),
    SecretNamePattern("*.key"),
    SecretNamePattern("id_rsa*"),
    SecretNamePattern("*credentials*"),
    SecretNamePattern("*secret*"),
)


def _readonly_mapping(value: Mapping[str, str], what: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise IngestionError(f"{what} must be a mapping, got {type(value).__name__}.")
    return MappingProxyType(dict(value))


def _check_label(label: object, key: str) -> None:
    if type(label) is not str or not _LABEL_RE.fullmatch(label):
        raise IngestionError(
            f"Invalid language label {label!r} for {key!r}: labels are a contract with "
            "later phases and must be lowercase plain strings such as 'python'."
        )


def _check_patterns(patterns: Iterable[str]) -> tuple[str, ...]:
    result = tuple(patterns)
    for pattern in result:
        if not isinstance(pattern, str) or not pattern.strip() or pattern.startswith("#"):
            raise IngestionError(
                f"Invalid built-in ignore pattern {pattern!r}: expected a non-empty "
                "gitignore pattern (not blank, not a comment)."
            )
        if pattern.startswith("!") or any(char in pattern for char in "\n\r\0"):
            raise IngestionError(
                f"Invalid built-in ignore pattern {pattern!r}: built-in patterns only "
                "exclude ('!' re-inclusion is not allowed) and must be a single line."
            )
    return result


def _check_positive(value: object, what: str) -> None:
    if type(value) is not int or value <= 0:
        raise IngestionError(f"{what} must be a positive integer, got {value!r}.")


@dataclass(frozen=True, slots=True)
class IngestionRules:
    """Everything the Step 1-3 filters decide with: plain data, validated on creation.

    The defaults are the owner's decisions recorded in docs/decisions.md section 7.
    """

    language_by_extension: Mapping[str, str] = field(
        default_factory=lambda: DEFAULT_LANGUAGE_BY_EXTENSION
    )
    language_by_filename: Mapping[str, str] = field(
        default_factory=lambda: DEFAULT_LANGUAGE_BY_FILENAME
    )
    lockfile_names: frozenset[str] = DEFAULT_LOCKFILE_NAMES
    builtin_patterns: tuple[str, ...] = DEFAULT_BUILTIN_PATTERNS
    secret_name_patterns: tuple[SecretNamePattern, ...] = DEFAULT_SECRET_NAME_PATTERNS
    head_bytes: int = DEFAULT_HEAD_BYTES
    max_gitignore_bytes: int = DEFAULT_MAX_GITIGNORE_BYTES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    def __post_init__(self) -> None:
        by_extension = _readonly_mapping(self.language_by_extension, "language_by_extension")
        for extension, label in by_extension.items():
            if not isinstance(extension, str) or not _EXTENSION_RE.fullmatch(extension):
                raise IngestionError(
                    f"Invalid extension {extension!r}: expected a lowercase single suffix "
                    "starting with a dot, such as '.py' (files are matched on their last "
                    "suffix, lowercased)."
                )
            _check_label(label, extension)
        by_filename = _readonly_mapping(self.language_by_filename, "language_by_filename")
        for name, label in by_filename.items():
            _check_file_name(name, "language file name")
            _check_label(label, name)
        lockfiles = frozenset(self.lockfile_names)
        for name in lockfiles:
            _check_file_name(name, "lockfile name")
        secrets = tuple(self.secret_name_patterns)
        for pattern in secrets:
            if not isinstance(pattern, SecretNamePattern):
                raise IngestionError(
                    f"secret_name_patterns must contain SecretNamePattern, got {pattern!r}."
                )
        _check_positive(self.head_bytes, "head_bytes")
        _check_positive(self.max_gitignore_bytes, "max_gitignore_bytes")
        _check_positive(self.max_file_bytes, "max_file_bytes")

        object.__setattr__(self, "language_by_extension", by_extension)
        object.__setattr__(self, "language_by_filename", by_filename)
        object.__setattr__(self, "lockfile_names", lockfiles)
        object.__setattr__(self, "builtin_patterns", _check_patterns(self.builtin_patterns))
        object.__setattr__(self, "secret_name_patterns", secrets)

    @property
    def languages(self) -> frozenset[str]:
        """Every ``language`` label these rules can assign."""
        return frozenset(self.language_by_extension.values()) | frozenset(
            self.language_by_filename.values()
        )
