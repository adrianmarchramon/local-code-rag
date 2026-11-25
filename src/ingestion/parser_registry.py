"""Which tree-sitter grammar segments which language label: data, loaded lazily.

Adding a language is a data change here (one :class:`GrammarSpec` plus its wheel in
``uv.lock``); the segmenter reads node types and field names from the spec and never
names a language. Labels are not grammar names: each spec says which module provides the
grammar (decision P5: only ``python`` today; every other label has no parser).

Importing this module imports no grammar and builds nothing. :func:`load_language`
imports the grammar module on first use and caches the ``Language`` for the process;
:func:`new_parser` builds a fresh ``Parser`` per call, so no parser state is shared.
Grammars are compiled into their wheels (decision P1): nothing is ever downloaded.
"""

from __future__ import annotations

import functools
import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import tree_sitter

from src.errors import ParserUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GrammarSpec:
    """Everything the segmenter needs to know about one grammar, as plain data."""

    label: str  # the loader's language label
    grammar_module: str  # importable module providing the compiled grammar
    language_function: str  # function of that module returning the TSLanguage pointer
    function_types: frozenset[str]
    class_types: frozenset[str]
    decorated_types: frozenset[str]  # a wrapper holding decorators plus a definition
    comment_types: frozenset[str]
    block_types: frozenset[str]  # statement lists (bodies of definitions and clauses)
    clause_types: frozenset[str]  # elif / else / except / finally / case ...
    name_field: str  # field holding a definition's name
    definition_field: str  # field of a decorated wrapper holding the definition

    @property
    def node_types(self) -> frozenset[str]:
        """Every node type this spec refers to (checked against the grammar in tests)."""
        return (
            self.function_types
            | self.class_types
            | self.decorated_types
            | self.comment_types
            | self.block_types
            | self.clause_types
        )


# tree-sitter-python 0.25.0 node types, checked against the installed grammar in tests.
PYTHON: Final = GrammarSpec(
    label="python",
    grammar_module="tree_sitter_python",
    language_function="language",
    function_types=frozenset({"function_definition"}),
    class_types=frozenset({"class_definition"}),
    decorated_types=frozenset({"decorated_definition"}),
    comment_types=frozenset({"comment"}),
    block_types=frozenset({"block"}),
    clause_types=frozenset(
        {"elif_clause", "else_clause", "except_clause", "finally_clause", "case_clause"}
    ),
    name_field="name",
    definition_field="definition",
)

GRAMMARS: Final[Mapping[str, GrammarSpec]] = MappingProxyType({PYTHON.label: PYTHON})


def grammar_for(label: str) -> GrammarSpec | None:
    """The grammar for a loader label, or None when the label has no parser (a value)."""
    return GRAMMARS.get(label)


@functools.cache
def _load(grammar_module: str, language_function: str) -> tree_sitter.Language:
    try:
        module = importlib.import_module(grammar_module)
        pointer = getattr(module, language_function)()
        language = tree_sitter.Language(pointer)
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise ParserUnavailableError(
            f"The tree-sitter grammar {grammar_module!r} could not be loaded ({error}). "
            "Grammars ship inside their wheels and are never downloaded: run `make install` "
            "so that .venv matches uv.lock."
        ) from error
    logger.debug(
        "chunking.grammar_loaded",
        extra={"grammar": grammar_module, "abi_version": language.abi_version},
    )
    return language


def load_language(spec: GrammarSpec) -> tree_sitter.Language:
    """The compiled grammar of ``spec``, loaded on first use and cached per process."""
    return _load(spec.grammar_module, spec.language_function)


def new_parser(spec: GrammarSpec) -> tree_sitter.Parser:
    """A fresh parser for ``spec`` (parsers are cheap; they are not shared)."""
    return tree_sitter.Parser(load_language(spec))
