"""Classification by file name (Phase 1, Step 3): lockfile, secret-like name, language.

Pure: it looks at the path string only and never touches the disk, so a secret-looking
file is rejected without ever being opened. Checks run in this order, and the first one
that applies decides:

1. lockfile, by exact base name -> ``lockfile``;
2. secret-like base name (case-insensitive globs, decision D4) -> ``secret_name``;
3. language by exact, case-sensitive file name (``Dockerfile``, ``Makefile``), then by
   the lowercased last extension; none -> ``unmapped_type``.

Only the last suffix counts (``a.test.py`` is Python, ``x.tar.gz`` is ``.gz``) and
dotfiles such as ``.gitignore`` have no extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath

from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules, SecretNamePattern


@dataclass(frozen=True, slots=True)
class NameClassification:
    """Either a ``language`` (the file may be ingested) or a discard ``reason``, never both."""

    language: str | None = None
    reason: DiscardReason | None = None

    def __post_init__(self) -> None:
        if (self.language is None) == (self.reason is None):
            raise ValueError("A NameClassification has exactly one of language or reason.")

    @property
    def accepted(self) -> bool:
        return self.language is not None


def looks_like_secret(name: str, patterns: tuple[SecretNamePattern, ...]) -> bool:
    """True if the base name matches a secret glob and none of that glob's exceptions."""
    lowered = name.lower()
    return any(
        fnmatchcase(lowered, pattern.glob)
        and not any(fnmatchcase(lowered, exception) for exception in pattern.exceptions)
        for pattern in patterns
    )


def classify_name(rel_path: str, rules: IngestionRules) -> NameClassification:
    """Classify a repository-relative POSIX path by its base name only (no I/O)."""
    name = rel_path.rpartition("/")[2]
    if not name:
        raise ValueError(f"Expected a path to a file, got {rel_path!r}.")
    if name in rules.lockfile_names:
        return NameClassification(reason=DiscardReason.LOCKFILE)
    if looks_like_secret(name, rules.secret_name_patterns):
        return NameClassification(reason=DiscardReason.SECRET_NAME)
    language = rules.language_by_filename.get(name) or rules.language_by_extension.get(
        PurePosixPath(name).suffix.lower()
    )
    if language is None:
        return NameClassification(reason=DiscardReason.UNMAPPED_TYPE)
    return NameClassification(language=language)
