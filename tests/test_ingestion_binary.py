"""Tests for the Step 3 binary probe: bounded reads, verdicts, symlinks and special files."""

import errno
import os
import threading
from pathlib import Path

import pytest

from src.ingestion.binary import ProbeVerdict, probe_head
from src.ingestion.reasons import DiscardReason

HEAD = 1024
# Smallest valid PNG header: signature + IHDR chunk (length field starts with NUL bytes).
PNG_HEADER = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489")


def _file(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_plain_text_is_text(tmp_path: Path) -> None:
    path = _file(tmp_path, "a.py", b"def hello():\n    return 'hi'\n")

    assert probe_head(path, HEAD) is ProbeVerdict.TEXT


@pytest.mark.parametrize(("position", "expected"), [(0, True), (1023, True), (1024, False)])
def test_nul_byte_is_only_seen_inside_the_first_kib(
    tmp_path: Path, position: int, expected: bool
) -> None:
    content = bytearray(b"a" * 4096)
    content[position] = 0
    path = _file(tmp_path, "a.py", bytes(content))

    verdict = probe_head(path, HEAD)

    assert (verdict is ProbeVerdict.BINARY) is expected  # 1024 is past the window


def test_png_is_binary(tmp_path: Path) -> None:
    assert probe_head(_file(tmp_path, "logo.png", PNG_HEADER), HEAD) is ProbeVerdict.BINARY


def test_utf16_text_is_reported_as_binary(tmp_path: Path) -> None:
    # Documented limitation: UTF-16 encodes ASCII with NUL bytes.
    path = _file(tmp_path, "notes.md", "hello".encode("utf-16-le"))

    assert probe_head(path, HEAD) is ProbeVerdict.BINARY


def test_empty_file_is_text(tmp_path: Path) -> None:
    # Empty files are a Step 4 decision (reason `empty`), not a binary one.
    assert probe_head(_file(tmp_path, "__init__.py", b""), HEAD) is ProbeVerdict.TEXT


def test_read_is_bounded_to_head_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _file(tmp_path, "big.py", b"x = 1\n" * 1_000_000)  # about 6 MB, no NUL
    returned: list[int] = []
    real_read = os.read

    def counting_read(fd: int, size: int) -> bytes:
        assert size <= HEAD
        data = real_read(fd, size)
        returned.append(len(data))
        return data

    monkeypatch.setattr(os, "read", counting_read)

    assert probe_head(path, HEAD) is ProbeVerdict.TEXT
    assert sum(returned) <= HEAD


def test_short_reads_are_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"a" * 1000 + b"\0" + b"a" * 100
    path = _file(tmp_path, "a.bin", content)
    real_read = os.read
    monkeypatch.setattr(os, "read", lambda fd, size: real_read(fd, min(size, 7)))

    assert probe_head(path, HEAD) is ProbeVerdict.BINARY


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    assert probe_head(tmp_path / "gone.py", HEAD) is ProbeVerdict.UNREADABLE


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_permission_denied_is_unreadable_not_binary(tmp_path: Path) -> None:
    path = _file(tmp_path, "locked.py", b"print('x')\n")
    path.chmod(0)
    try:
        assert probe_head(path, HEAD) is ProbeVerdict.UNREADABLE
    finally:
        path.chmod(0o600)


def test_open_failure_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _file(tmp_path, "a.py", b"x\n")

    def denied(*_args: object, **_kwargs: object) -> int:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(os, "open", denied)

    assert probe_head(path, HEAD) is ProbeVerdict.UNREADABLE


def test_read_error_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _file(tmp_path, "a.py", b"x\n")

    def failing_read(_fd: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "read", failing_read)

    assert probe_head(path, HEAD) is ProbeVerdict.UNREADABLE


@pytest.mark.parametrize("target_inside", [True, False])
def test_symlinks_are_never_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_inside: bool
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = _file(repo if target_inside else tmp_path, "target.py", b"secret = 1\n")
    link = repo / "link.py"
    link.symlink_to(target)
    opened: list[str] = []
    real_open = os.open

    def recording_open(path: str | os.PathLike[str], flags: int, *args: int) -> int:
        opened.append(os.fspath(path))
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", recording_open)

    assert probe_head(link, HEAD) is ProbeVerdict.NOT_REGULAR_FILE
    assert opened == []  # neither the link nor its target was opened


def test_dangling_symlink_is_not_a_regular_file(tmp_path: Path) -> None:
    link = tmp_path / "dangling.py"
    link.symlink_to(tmp_path / "nowhere.py")

    assert probe_head(link, HEAD) is ProbeVerdict.NOT_REGULAR_FILE


def test_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    (tmp_path / "pkg.py").mkdir()

    assert probe_head(tmp_path / "pkg.py", HEAD) is ProbeVerdict.NOT_REGULAR_FILE


def test_fifo_does_not_block(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.py"
    os.mkfifo(fifo)
    result: list[ProbeVerdict] = []

    worker = threading.Thread(target=lambda: result.append(probe_head(fifo, HEAD)), daemon=True)
    worker.start()
    worker.join(5)

    assert not worker.is_alive(), "probe_head blocked on a FIFO"
    assert result == [ProbeVerdict.NOT_REGULAR_FILE]


@pytest.mark.parametrize("head_bytes", [0, -1, True])
def test_invalid_head_size_is_rejected(tmp_path: Path, head_bytes: int) -> None:
    with pytest.raises(ValueError, match="head_bytes"):
        probe_head(_file(tmp_path, "a.py", b"x"), head_bytes)


def test_verdicts_map_to_discard_reasons() -> None:
    assert ProbeVerdict.TEXT.discard_reason is None
    assert ProbeVerdict.BINARY.discard_reason is DiscardReason.BINARY
    assert ProbeVerdict.UNREADABLE.discard_reason is DiscardReason.UNREADABLE
    assert ProbeVerdict.NOT_REGULAR_FILE.discard_reason is DiscardReason.NOT_REGULAR_FILE
