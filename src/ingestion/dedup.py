"""Exact-duplicate detection by content hash (Phase 1, Step 4; decision Q2).

Runs last, and only over files that passed every other filter: a file excluded by a path
rule (for example, a stale copy under ``deploy/model/``) can never become the canonical
copy that hides a legitimate one. Within a group of files with identical bytes, the
canonical copy is the lexicographically smallest POSIX path string; every other copy is
discarded as ``duplicate`` and recorded with its canonical path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Duplicate:
    """A file left out because ``canonical`` has the same bytes and a smaller path."""

    discarded: str
    canonical: str


def find_duplicates(hash_by_path: Mapping[str, str]) -> tuple[Duplicate, ...]:
    """Return the duplicates among ``{rel_path: content_hash}``, sorted by discarded path."""
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    for rel_path, digest in hash_by_path.items():
        paths_by_hash[digest].append(rel_path)
    duplicates: list[Duplicate] = []
    for paths in paths_by_hash.values():
        if len(paths) < 2:
            continue
        canonical = min(paths)  # Unicode code point order of the full POSIX string
        duplicates.extend(Duplicate(path, canonical) for path in paths if path != canonical)
    return tuple(sorted(duplicates, key=lambda duplicate: duplicate.discarded))
