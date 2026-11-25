"""Tests for Step 4 content reading: D5 text contract, size cap, races, verdicts."""

import errno
import hashlib
import os
from pathlib import Path

import pytest

from src.ingestion.content import (
    ContentResult,
    FileContent,
    check_metadata,
    normalize_newlines,
    read_text_file,
)
from src.ingestion.reasons import DiscardReason

CAP = 64


def _file(tmp_path: Path, content: bytes, name: str = "a.py") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _read(path: Path, cap: int = CAP) -> ContentResult:
    return read_text_file(path, cap)


def _text(path: Path, cap: int = CAP) -> str:
    content = _read(path, cap).content
    assert content is not None
    return content.text


# --- decoding and the D5 text contract -----------------------------------------------------------


def test_utf8_text_is_kept_exactly(tmp_path: Path) -> None:
    raw = "def café():\n    return 'ñ'\n".encode()

    result = _read(_file(tmp_path, raw))

    assert result == ContentResult(
        content=FileContent(
            text=raw.decode(), size_bytes=len(raw), content_hash=hashlib.sha256(raw).hexdigest()
        )
    )


def test_leading_bom_is_stripped_but_counted_in_size_and_hash(tmp_path: Path) -> None:
    raw = b"\xef\xbb\xbfx = 1\n"

    content = _read(_file(tmp_path, raw)).content

    assert content is not None
    assert content.text == "x = 1\n"
    assert content.size_bytes == len(raw)
    assert content.content_hash == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        (b"a\r\nb\r\n", "a\nb\n"),  # CRLF
        (b"a\rb\r", "a\nb\n"),  # lone CR (classic Mac)
        (b"a\r\nb\rc\n", "a\nb\nc\n"),  # mixed
        (b"a\r\r\nb", "a\n\nb"),  # CR followed by CRLF: two line breaks
    ],
)
def test_line_endings_are_normalized(tmp_path: Path, raw: bytes, text: str) -> None:
    assert _text(_file(tmp_path, raw)) == text


def test_bom_and_crlf_keep_the_on_disk_size(tmp_path: Path) -> None:
    raw = b"\xef\xbb\xbfa\r\nb\r\n"

    content = _read(_file(tmp_path, raw)).content

    assert content is not None
    assert (content.text, content.size_bytes) == ("a\nb\n", len(raw))
    assert len(content.text) != content.size_bytes  # documented trade-off of D5


def test_other_separators_are_left_alone(tmp_path: Path) -> None:
    # Only \r is normalized; form feed and U+2028 are ordinary characters here.
    text = "a\x0cb" + chr(0x2028) + "c\n"  # form feed and U+2028 LINE SEPARATOR
    assert _text(_file(tmp_path, text.encode())) == text


def test_normalize_newlines_helper() -> None:
    assert normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"


# --- discards -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"\xff\xfe\x00", b"caf\xe9\n", b"\xc3\x28", "x".encode("utf-16")],
    ids=["utf16-bom", "latin1", "truncated-sequence", "utf16"],
)
def test_non_utf8_is_undecodable(tmp_path: Path, raw: bytes) -> None:
    assert _read(_file(tmp_path, raw)).reason is DiscardReason.UNDECODABLE


def test_nul_after_the_first_kib_is_binary(tmp_path: Path) -> None:
    raw = b"a" * 2000 + b"\x00" + b"b\n"  # valid UTF-8; the probe would not see it

    assert _read(_file(tmp_path, raw), cap=4096).reason is DiscardReason.BINARY


@pytest.mark.parametrize("raw", [b"", b"   ", b" \n\t\r\n", b"\xef\xbb\xbf", b"\xef\xbb\xbf \n"])
def test_empty_or_whitespace_only_is_empty(tmp_path: Path, raw: bytes) -> None:
    assert _read(_file(tmp_path, raw)).reason is DiscardReason.EMPTY


def test_cap_boundary(tmp_path: Path) -> None:
    at_cap = _file(tmp_path, b"x" * CAP, "at.py")
    over_cap = _file(tmp_path, b"x" * (CAP + 1), "over.py")

    assert _read(at_cap).content is not None  # max_file_bytes itself is accepted
    assert _read(over_cap).reason is DiscardReason.TOO_LARGE


def test_file_that_grows_after_lstat_is_too_large(tmp_path: Path) -> None:
    path = _file(tmp_path, b"x = 1\n")
    assert check_metadata(path, CAP) is None  # small when checked

    with path.open("ab") as handle:  # ... grows before it is read
        handle.write(b"#" * CAP)

    assert _read(path).reason is DiscardReason.TOO_LARGE


def test_file_replaced_by_a_symlink_is_not_followed(tmp_path: Path) -> None:
    target = _file(tmp_path, b"secret = 1\n", "target.py")
    path = _file(tmp_path, b"x = 1\n")
    assert check_metadata(path, CAP) is None
    path.unlink()
    path.symlink_to(target)

    assert _read(path).reason is DiscardReason.NOT_REGULAR_FILE
    assert check_metadata(path, CAP) is DiscardReason.SYMLINK


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_permission_denied_is_unreadable(tmp_path: Path) -> None:
    path = _file(tmp_path, b"x = 1\n")
    path.chmod(0)
    try:
        assert _read(path).reason is DiscardReason.UNREADABLE
    finally:
        path.chmod(0o600)


def test_read_error_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _file(tmp_path, b"x = 1\n")

    def failing_read(_fd: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "read", failing_read)

    assert _read(path).reason is DiscardReason.UNREADABLE


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    assert _read(tmp_path / "gone.py").reason is DiscardReason.UNREADABLE


def test_never_reads_more_than_cap_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _file(tmp_path, b"x = 1\n" * 1_000_000)  # about 6 MB
    returned: list[int] = []
    real_read = os.read

    def counting_read(fd: int, size: int) -> bytes:
        data = real_read(fd, size)
        returned.append(len(data))
        return data

    monkeypatch.setattr(os, "read", counting_read)

    assert _read(path, cap=1024).reason is DiscardReason.TOO_LARGE
    assert sum(returned) <= 1025


# --- check_metadata (lstat only) ----------------------------------------------------------------


def test_check_metadata_verdicts(tmp_path: Path) -> None:
    small = _file(tmp_path, b"x\n", "small.py")
    big = _file(tmp_path, b"x" * (CAP + 1), "big.py")
    (tmp_path / "link.py").symlink_to(small)
    os.mkfifo(tmp_path / "pipe.py")

    assert check_metadata(small, CAP) is None
    assert check_metadata(big, CAP) is DiscardReason.TOO_LARGE
    assert check_metadata(tmp_path / "link.py", CAP) is DiscardReason.SYMLINK
    assert check_metadata(tmp_path / "pipe.py", CAP) is DiscardReason.NOT_REGULAR_FILE
    assert check_metadata(tmp_path / "gone.py", CAP) is DiscardReason.UNREADABLE


def test_check_metadata_never_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _file(tmp_path, b"x\n")

    def forbidden(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("check_metadata opened the file")

    monkeypatch.setattr(os, "open", forbidden)

    assert check_metadata(path, CAP) is None


# --- contract -----------------------------------------------------------------------------------


def test_result_has_exactly_one_outcome() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ContentResult()


@pytest.mark.parametrize("cap", [0, -1, True])
def test_invalid_cap_is_rejected(tmp_path: Path, cap: int) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        read_text_file(_file(tmp_path, b"x\n"), cap)
