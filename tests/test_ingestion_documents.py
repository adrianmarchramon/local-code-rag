"""Tests for Step 5: metadata contract, doc_id, hidden keys and what the models see."""

import hashlib
import logging
import shutil
import warnings
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from src.ingestion.content import FileContent, read_text_file
from src.ingestion.documents import (
    HIDDEN_METADATA_KEYS,
    METADATA_KEYS,
    VISIBLE_METADATA_KEYS,
    build_document,
    count_lines,
)

CAP = 1024 * 1024


def _content(text: str, raw: bytes | None = None) -> FileContent:
    data = text.encode() if raw is None else raw
    return FileContent(
        text=text, size_bytes=len(data), content_hash=hashlib.sha256(data).hexdigest()
    )


def _doc(rel_path: str = "src/app.py", text: str = "x = 1\n", language: str = "python") -> Document:
    return build_document(rel_path, language, _content(text))


def _from_disk(root: Path, rel_path: str, raw: bytes, language: str = "python") -> Document:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    content = read_text_file(path, CAP).content
    assert content is not None
    return build_document(rel_path, language, content)


# --- the metadata contract ----------------------------------------------------------------------


def test_metadata_has_exactly_the_six_keys_in_order() -> None:
    assert tuple(_doc().metadata) == METADATA_KEYS
    assert set(VISIBLE_METADATA_KEYS) | set(HIDDEN_METADATA_KEYS) == set(METADATA_KEYS)
    assert not set(VISIBLE_METADATA_KEYS) & set(HIDDEN_METADATA_KEYS)


def test_metadata_values_and_types(tmp_path: Path) -> None:
    raw = b"\xef\xbb\xbfdef f():\r\n    return 1\r\n"

    doc = _from_disk(tmp_path, "src/pkg/mod.py", raw)

    assert doc.metadata == {
        "file_path": "src/pkg/mod.py",
        "language": "python",
        "size_bytes": len(raw),  # on-disk bytes, BOM and CRLF included
        "num_lines": 2,
        "content_hash": hashlib.sha256(raw).hexdigest(),
        "directory": "src/pkg",
    }
    assert [type(doc.metadata[k]) for k in METADATA_KEYS] == [str, str, int, int, str, str]
    assert (tmp_path / "src/pkg/mod.py").stat().st_size == doc.metadata["size_bytes"]


def test_directory_is_empty_at_the_root() -> None:
    assert _doc("README.md", "# Title\n", "markdown").metadata["directory"] == ""


def test_text_is_exactly_the_normalized_file(tmp_path: Path) -> None:
    doc = _from_disk(tmp_path, "a.py", b"\xef\xbb\xbfa\r\nb\rc\n")

    assert doc.text == "a\nb\nc\n"
    assert "﻿" not in doc.text
    assert "\0" not in doc.text
    assert "\r" not in doc.text


# --- num_lines (D5) -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lines"),
    [
        ("", 0),
        ("a", 1),
        ("a\n", 1),
        ("a\nb", 2),
        ("a\nb\n", 2),
        ("\n", 1),
        ("a\n\nb\n", 3),
        ("a\n\n\n", 3),
        ("a\x0cb\n", 1),  # form feed is not a line break here (unlike str.splitlines)
    ],
)
def test_count_lines(text: str, lines: int) -> None:
    assert count_lines(text) == lines


@pytest.mark.parametrize("k", range(1, 25))
@pytest.mark.parametrize("trailing_newline", [True, False])
def test_k_lines_count_as_k(k: int, trailing_newline: bool) -> None:
    text = "\n".join(f"line {i}" for i in range(k)) + ("\n" if trailing_newline else "")

    assert count_lines(text) == k
    assert _doc(text=text).metadata["num_lines"] == k


def test_crlf_lines_count_once(tmp_path: Path) -> None:
    assert _from_disk(tmp_path, "a.py", b"a\r\nb\r\n").metadata["num_lines"] == 2


# --- doc_id (D6) --------------------------------------------------------------------------------


def test_doc_id_is_the_posix_path() -> None:
    doc = _doc("src/app.py")

    assert doc.doc_id == "src/app.py"
    assert doc.id_ == "src/app.py"


def test_doc_id_is_unique_per_path_and_independent_of_content() -> None:
    first = _doc("src/a.py", "x = 1\n")
    edited = _doc("src/a.py", "x = 2\n")
    other = _doc("src/b.py", "x = 1\n")

    assert first.doc_id == edited.doc_id  # stable when the content changes
    assert first.doc_id != other.doc_id  # unique per path, even with equal content


def test_doc_id_does_not_depend_on_the_root(tmp_path: Path) -> None:
    original = tmp_path / "one"
    doc_a = _from_disk(original, "src/a.py", b"x = 1\n")
    copy = tmp_path / "elsewhere" / "two"
    shutil.copytree(original, copy)
    content = read_text_file(copy / "src" / "a.py", CAP).content
    assert content is not None

    doc_b = build_document("src/a.py", "python", content)

    assert (doc_a.doc_id, doc_a.metadata) == (doc_b.doc_id, doc_b.metadata)


def test_no_deprecated_api_is_used(caplog: pytest.LogCaptureFixture) -> None:
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        _doc()

    assert caught == []
    assert caplog.records == []  # Document logs a warning for deprecated arguments


# --- what the embedder and the LLM see ----------------------------------------------------------


@pytest.mark.parametrize("mode", [MetadataMode.EMBED, MetadataMode.LLM])
def test_models_see_only_file_path_and_language(mode: MetadataMode) -> None:
    doc = _doc("src/app.py", "x = 1\n")

    seen = doc.get_content(metadata_mode=mode)

    assert seen == "file_path: src/app.py\nlanguage: python\n\nx = 1"
    for key in HIDDEN_METADATA_KEYS:
        assert f"{key}:" not in seen
    assert str(doc.metadata["content_hash"]) not in seen


def test_without_metadata_the_content_is_the_text() -> None:
    doc = _doc(text="x = 1\n\n")

    assert doc.get_content(metadata_mode=MetadataMode.NONE) == "x = 1\n\n"


def test_excluded_lists_are_not_shared_between_documents() -> None:
    first, second = _doc("a.py"), _doc("b.py")

    first.excluded_embed_metadata_keys.append("file_path")
    first.excluded_llm_metadata_keys.clear()

    assert second.excluded_embed_metadata_keys == list(HIDDEN_METADATA_KEYS)
    assert second.excluded_llm_metadata_keys == list(HIDDEN_METADATA_KEYS)


def test_vector_store_keys_do_not_overwrite_ours() -> None:
    doc = _doc("src/app.py")

    stored = node_to_metadata_dict(doc, remove_text=True, flat_metadata=True)

    assert {key: stored[key] for key in METADATA_KEYS} == doc.metadata
    assert stored["doc_id"] == "None"  # a Document has no ref_doc_id; chunks will


def test_visible_metadata_budget_for_the_longest_corpus_path() -> None:
    # Measured: the longest of the 88 corpus paths has 65 characters. Phase 2's splitter
    # subtracts this string (in tokens) from every chunk's size.
    longest = "docs/decisions/0025-phase-5-reproducibility-and-key-test-scope.md"
    doc = _doc(longest, "# t\n", "markdown")

    visible = doc.get_metadata_str(mode=MetadataMode.EMBED)

    assert visible == f"file_path: {longest}\nlanguage: markdown"
    assert len(visible) == len("file_path: ") + 65 + len("\nlanguage: markdown") == 95


# --- input validation ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", ["", "/abs.py", "./a.py", "a/../b.py", "a\\b.py", "a//b.py"])
def test_invalid_paths_are_rejected(rel_path: str) -> None:
    with pytest.raises(ValueError, match="file_path"):
        _doc(rel_path)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="without text"):
        _doc(text="")
