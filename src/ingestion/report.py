"""Immutable, deterministic summary of one repository load (Phase 1, Step 4).

The report carries counts and repository-relative paths only: no timings, no timestamps,
no absolute root path, never file contents. Two loads of the same tree produce equal
reports. Timings go to the logs.

Counting rule: a directory pruned by a rule is counted once, as a directory (its contents
are never listed), so these counts are not file-for-file comparable with a listing such
as ``git ls-files``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from src.ingestion.dedup import Duplicate
from src.ingestion.reasons import Discard

ROOT_GROUP = "."  # top-level group for files directly in the repository root


@dataclass(frozen=True, slots=True)
class IncludedFile:
    """The part of a loaded document the report needs."""

    rel_path: str
    language: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Tally:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What a load kept and what it left out, and why."""

    included_files: int
    included_bytes: int
    by_language: Mapping[str, Tally]
    by_top_level: Mapping[str, Tally]
    discarded_files: Mapping[str, int]  # reason code -> files (duplicates included)
    pruned_dirs: Mapping[str, int]  # reason code -> directories not descended into
    duplicates: tuple[Duplicate, ...]
    gitignores_read: int

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form, with deterministic key order."""
        return {
            "included": {"files": self.included_files, "bytes": self.included_bytes},
            "by_language": _tallies(self.by_language),
            "by_top_level": _tallies(self.by_top_level),
            "discarded_files": dict(self.discarded_files),
            "pruned_dirs": dict(self.pruned_dirs),
            "duplicates": [
                {"discarded": d.discarded, "canonical": d.canonical} for d in self.duplicates
            ],
            "gitignores_read": self.gitignores_read,
        }

    def render_text(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [f"Included: {self.included_files} files, {self.included_bytes:,} bytes"]
        lines += _tally_lines("By language", self.by_language)
        lines += _tally_lines("By top-level directory", self.by_top_level)
        lines += _count_lines("Discarded files", self.discarded_files)
        lines += _count_lines("Pruned directories", self.pruned_dirs)
        lines.append(f"Duplicates: {len(self.duplicates)}")
        lines += [f"  {d.discarded} (same bytes as {d.canonical})" for d in self.duplicates]
        lines.append(f".gitignore files read: {self.gitignores_read}")
        return "\n".join(lines) + "\n"


def _tallies(tallies: Mapping[str, Tally]) -> dict[str, dict[str, int]]:
    return {key: {"files": t.files, "bytes": t.bytes} for key, t in tallies.items()}


def _tally_lines(title: str, tallies: Mapping[str, Tally]) -> list[str]:
    return [f"{title}:"] + [
        f"  {key}: {t.files} files, {t.bytes:,} bytes" for key, t in tallies.items()
    ]


def _count_lines(title: str, counts: Mapping[str, int]) -> list[str]:
    total = sum(counts.values())
    return [f"{title}: {total}"] + [f"  {key}: {count}" for key, count in counts.items()]


def top_level(rel_path: str) -> str:
    """First path component, or ``"."`` for a file in the repository root."""
    return rel_path.split("/", 1)[0] if "/" in rel_path else ROOT_GROUP


def _group(
    files: Iterable[IncludedFile], key_of: Callable[[IncludedFile], str]
) -> Mapping[str, Tally]:
    files_by: Counter[str] = Counter()
    bytes_by: Counter[str] = Counter()
    for item in files:
        key = key_of(item)
        files_by[key] += 1
        bytes_by[key] += item.size_bytes
    return MappingProxyType({key: Tally(files_by[key], bytes_by[key]) for key in sorted(files_by)})


def _reason_counts(discards: Iterable[Discard]) -> Mapping[str, int]:
    counts = Counter(discard.reason.value for discard in discards)
    return MappingProxyType(dict(sorted(counts.items())))


def build_report(
    included: Sequence[IncludedFile],
    discarded: Iterable[Discard],
    pruned: Iterable[Discard],
    duplicates: Sequence[Duplicate],
    gitignores_read: int,
) -> LoadReport:
    """Aggregate one load; every mapping is read-only and sorted by key."""
    return LoadReport(
        included_files=len(included),
        included_bytes=sum(item.size_bytes for item in included),
        by_language=_group(included, lambda item: item.language),
        by_top_level=_group(included, lambda item: top_level(item.rel_path)),
        discarded_files=_reason_counts(discarded),
        pruned_dirs=_reason_counts(pruned),
        duplicates=tuple(duplicates),
        gitignores_read=gitignores_read,
    )
