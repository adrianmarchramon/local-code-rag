"""Bounded reading and strict decoding of one candidate file (Phase 1, Step 4; decision D5).

Two steps, cheapest first:

* :func:`check_metadata` uses ``lstat`` only (nothing is opened): ``symlink``,
  ``not_regular_file``, ``too_large`` or ``unreadable``;
* :func:`read_text_file` opens the file without following symlinks or blocking (the
  helper shared with the binary probe), reads at most ``max_bytes + 1`` bytes, so a file
  that grew after ``lstat`` is still caught as ``too_large``, and then:

  - decodes strict UTF-8, tolerating one leading byte order mark; anything else is
    ``undecodable`` (no fallback encoding, nothing silently replaced);
  - rejects a NUL character anywhere in the decoded text as ``binary`` (NUL is valid
    UTF-8, and the probe only looks at the first KiB);
  - normalizes ``\\r\\n`` and lone ``\\r`` to ``\\n`` (D5): line numbers in later phases
    are counted over this text;
  - rejects empty or whitespace-only text as ``empty``;
  - hashes the bytes read from disk (sha256), which is both ``content_hash`` and the
    deduplication key.

I/O failures on a single file are verdicts, never exceptions: only ``OSError`` and
``UnicodeDecodeError`` are caught.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.binary import ProbeVerdict, open_regular_file, read_at_most
from src.ingestion.reasons import DiscardReason


@dataclass(frozen=True, slots=True)
class FileContent:
    """What a document is built from."""

    text: str  # decoded, BOM stripped, line endings normalized to "\n"
    size_bytes: int  # bytes on disk (equal to the bytes read)
    content_hash: str  # sha256 hex digest of the bytes on disk


@dataclass(frozen=True, slots=True)
class ContentResult:
    """Either the file's content or the reason it was left out, never both."""

    content: FileContent | None = None
    reason: DiscardReason | None = None

    def __post_init__(self) -> None:
        if (self.content is None) == (self.reason is None):
            raise ValueError("A ContentResult has exactly one of content or reason.")


def check_metadata(path: Path, max_bytes: int) -> DiscardReason | None:
    """Reject a path from its ``lstat`` alone, before opening it; None means "go on"."""
    try:
        info = os.lstat(path)
    except OSError:
        return DiscardReason.UNREADABLE
    if stat.S_ISLNK(info.st_mode):
        return DiscardReason.SYMLINK
    if not stat.S_ISREG(info.st_mode):
        return DiscardReason.NOT_REGULAR_FILE
    if info.st_size > max_bytes:
        return DiscardReason.TOO_LARGE
    return None


def normalize_newlines(text: str) -> str:
    """``\\r\\n`` and lone ``\\r`` become ``\\n`` (decision D5)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_file(path: Path, max_bytes: int) -> ContentResult:
    """Read and decode one file under the D5 contract; see the module docstring."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError(f"max_bytes must be a positive integer, got {max_bytes!r}.")
    opened = open_regular_file(path)
    if isinstance(opened, ProbeVerdict):
        return ContentResult(reason=opened.discard_reason)
    try:
        data = read_at_most(opened, max_bytes + 1)
    except OSError:
        return ContentResult(reason=DiscardReason.UNREADABLE)
    finally:
        os.close(opened)
    if len(data) > max_bytes:
        return ContentResult(reason=DiscardReason.TOO_LARGE)
    try:
        decoded = data.decode("utf-8-sig")  # strict; strips one leading BOM
    except UnicodeDecodeError:
        return ContentResult(reason=DiscardReason.UNDECODABLE)
    if "\0" in decoded:
        return ContentResult(reason=DiscardReason.BINARY)
    text = normalize_newlines(decoded)
    if not text.strip():
        return ContentResult(reason=DiscardReason.EMPTY)
    return ContentResult(
        content=FileContent(
            text=text, size_bytes=len(data), content_hash=hashlib.sha256(data).hexdigest()
        )
    )
