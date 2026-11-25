"""Stable reason codes for every file that ingestion leaves out.

The codes appear in logs and in the Step 4 loading report, so they are a contract:
existing values never change; later steps only add new members (``symlink``, ``empty``,
``too_large``, ``undecodable``, ``duplicate``).
"""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class DiscardReason(StrEnum):
    """Why a path was not ingested; only the first filter that rejects it is recorded.

    Members are listed in evaluation order, cheapest check first.
    """

    BUILTIN_PATTERN = "builtin_pattern"  # generic noise: dependencies, caches, builds, .git
    CORPUS_RULES = "corpus_rules"  # the versioned rules file (INGEST_IGNORE_FILE)
    GITIGNORE = "gitignore"  # the corpus's own .gitignore chain
    LOCKFILE = "lockfile"  # dependency lockfile, matched by exact file name
    # A reason code, not a credential (bandit S105 false positive). The file is never opened.
    SECRET_NAME = "secret_name"  # noqa: S105
    UNMAPPED_TYPE = "unmapped_type"  # no language for its file name or extension
    BINARY = "binary"  # NUL byte in the first bytes read
    UNREADABLE = "unreadable"  # could not be opened or read (permissions, I/O error)
    NOT_REGULAR_FILE = "not_regular_file"  # symlink, FIFO, socket, device, directory
