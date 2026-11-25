"""Step 5: build one LlamaIndex ``Document`` per file, with the Q6 metadata contract.

Contract with later phases (decisions Q6, D5, D6):

* ``doc_id`` (``id_``) is the repository-relative POSIX path: deterministic, unique within
  a repository, independent of the file's content and of where the repository lives.
* ``text`` is the decoded file with line endings normalized to ``\\n`` (D5). Phase 2
  counts lines over this text.
* Metadata keys, in this order: ``file_path`` (str), ``language`` (str), ``size_bytes``
  (int, bytes on disk), ``num_lines`` (int, editor-style), ``content_hash`` (str, sha256
  hex of the bytes on disk) and ``directory`` (str, POSIX parent, ``""`` at the root).
* Only ``file_path`` and ``language`` reach the embedding model and the LLM; the other
  four are listed in fresh ``excluded_embed_metadata_keys`` and
  ``excluded_llm_metadata_keys`` lists on every document. Every visible character costs
  tokens in every chunk, and a 64-character hash would carry no meaning for retrieval.

None of the six keys collides with what vector stores add (``doc_id``, ``document_id``,
``ref_doc_id``, ``_node_content``, ``_node_type``), and ``content_hash`` is unrelated to
LlamaIndex's own ``Document.hash``.
"""

from __future__ import annotations

from typing import Final

from llama_index.core import Document

from src.ingestion.content import FileContent

METADATA_KEYS: Final = (
    "file_path",
    "language",
    "size_bytes",
    "num_lines",
    "content_hash",
    "directory",
)
VISIBLE_METADATA_KEYS: Final = ("file_path", "language")
HIDDEN_METADATA_KEYS: Final = ("size_bytes", "num_lines", "content_hash", "directory")


def count_lines(text: str) -> int:
    """Editor-style line count (D5): newlines, plus one for a last line without one.

    ``"a\\nb\\n"`` and ``"a\\nb"`` both have 2 lines; ``""`` has 0. Unlike
    ``str.splitlines``, only ``\\n`` separates lines, matching how Phase 2 counts them.
    """
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _check_rel_path(rel_path: str) -> None:
    parts = rel_path.split("/")
    if (
        not rel_path
        or rel_path.startswith("/")
        or "\\" in rel_path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(
            f"file_path must be a POSIX path relative to the repository root, got {rel_path!r}."
        )


def build_document(rel_path: str, language: str, content: FileContent) -> Document:
    """Build the ``Document`` for one file; see the module docstring for the contract."""
    _check_rel_path(rel_path)
    if not content.text:
        # Document(text="") would silently carry no text; empty files are discarded earlier.
        raise ValueError(f"Refusing to build a document without text for {rel_path!r}.")
    metadata: dict[str, str | int] = {
        "file_path": rel_path,
        "language": language,
        "size_bytes": content.size_bytes,
        "num_lines": count_lines(content.text),
        "content_hash": content.content_hash,
        "directory": rel_path.rpartition("/")[0],
    }
    return Document(
        id_=rel_path,
        text=content.text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(HIDDEN_METADATA_KEYS),
        excluded_llm_metadata_keys=list(HIDDEN_METADATA_KEYS),
    )
