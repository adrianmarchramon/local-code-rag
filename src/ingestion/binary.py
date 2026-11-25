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


def open_regular_file(path: Path) -> int | ProbeVerdict:
    """Open a regular file read-only, never following a symlink and never blocking.

    Returns the file descriptor (the caller closes it), or the verdict that explains why
    the file was not opened: ``not_regular_file`` (symlink, FIFO, socket, device,
    directory) or ``unreadable`` (missing, permission denied, swapped for a symlink after
    ``lstat``). Shared by the probe and Step 4's content reader.
    """
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return ProbeVerdict.NOT_REGULAR_FILE  # never opened
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
    except OSError:
        return ProbeVerdict.UNREADABLE
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        os.close(fd)
        return ProbeVerdict.UNREADABLE
    if not regular:
        os.close(fd)
        return ProbeVerdict.NOT_REGULAR_FILE
    return fd


def read_at_most(fd: int, limit: int) -> bytes:
    """Read up to ``limit`` bytes, completing short reads; raises OSError on I/O errors."""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def probe_head(path: Path, head_bytes: int) -> ProbeVerdict:
    """Classify ``path`` by reading at most ``head_bytes`` bytes from its start."""
    if type(head_bytes) is not int or head_bytes <= 0:
        raise ValueError(f"head_bytes must be a positive integer, got {head_bytes!r}.")
    opened = open_regular_file(path)
    if isinstance(opened, ProbeVerdict):
        return opened
    try:
        head = read_at_most(opened, head_bytes)
    except OSError:
        return ProbeVerdict.UNREADABLE
    finally:
        os.close(opened)
    return ProbeVerdict.BINARY if b"\0" in head else ProbeVerdict.TEXT
