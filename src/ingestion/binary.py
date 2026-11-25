"""Bounded binary probe (Phase 1, Step 3): read at most the first ``head_bytes`` of a file.

A NUL byte in that window marks the file as binary (rag_phase1.md Step 3). Unlike the
doc's sketch, it never reads the whole file, and it reports a file it cannot read as
``unreadable``, not ``binary``: an I/O error says nothing about the content.

Safety: the path is checked with ``lstat`` and opened with ``O_NOFOLLOW`` and
``O_NONBLOCK``, so a symlink is never followed and a FIFO or device is never opened (and
cannot block). Per-file I/O errors become a verdict; nothing is raised for them.

Known limits: UTF-16 and UTF-32 text contains NUL bytes and is reported as binary; a NUL
byte after the window is not seen (it is valid UTF-8, so Step 4's decoding will not
reject it either).
"""

from __future__ import annotations

import os
import stat
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

from src.ingestion.reasons import DiscardReason

_O_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK: Final = getattr(os, "O_NONBLOCK", 0)
_O_CLOEXEC: Final = getattr(os, "O_CLOEXEC", 0)


@unique
class ProbeVerdict(StrEnum):
    """What the first bytes of a path say about it."""

    TEXT = "text"
    BINARY = "binary"
    UNREADABLE = "unreadable"
    NOT_REGULAR_FILE = "not_regular_file"

    @property
    def discard_reason(self) -> DiscardReason | None:
        """The matching :class:`DiscardReason`, or None for text."""
        return None if self is ProbeVerdict.TEXT else DiscardReason(self.value)


def probe_head(path: Path, head_bytes: int) -> ProbeVerdict:
    """Classify ``path`` by reading at most ``head_bytes`` bytes from its start."""
    if type(head_bytes) is not int or head_bytes <= 0:
        raise ValueError(f"head_bytes must be a positive integer, got {head_bytes!r}.")
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return ProbeVerdict.NOT_REGULAR_FILE  # symlink, FIFO, socket, device, directory
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
    except OSError:
        # Missing, permission denied, or swapped for a symlink after lstat (ELOOP).
        return ProbeVerdict.UNREADABLE
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return ProbeVerdict.NOT_REGULAR_FILE
        head = b""
        while len(head) < head_bytes:
            chunk = os.read(fd, head_bytes - len(head))
            if not chunk:
                break
            head += chunk
    except OSError:
        return ProbeVerdict.UNREADABLE
    finally:
        os.close(fd)
    return ProbeVerdict.BINARY if b"\0" in head else ProbeVerdict.TEXT
