"""Phase 2, Steps 1-2: coherence invariants, the parser registry and the segmenter.

Synthetic texts only. Every segmentation case is checked against all invariants I1-I8
(tests/chunking_invariants.py) and against the structure the agreed policies imply
(docs/decisions.md section 10). Small caps keep the over-the-cap cases readable.
"""

import ast
import builtins
import dataclasses
import hashlib
import io
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from chunk_manifest import (
    ChunkManifestError,
    build_manifest,
    compare_manifests,
    document_digest,
    format_manifest,
    parse_manifest,
)
from chunk_validation import ReviewPick, review_sample, split_definitions, tree_snapshot, validate
from chunking_invariants import (
    Violation,
    check_all,
    check_bounds,
    check_cap,
    check_coverage,
    check_definitions_whole,
    check_deterministic,
    check_forced,
    check_forced_where_needed,
    check_line_boundaries,
    check_order,
    check_shared,
    check_units_whole,
    config_top_regions,
    definition_regions,
    fence_regions,
    markdown_section_regions,
    naive_char_split,
    naive_line_split,
)
from llama_index.core import Document
from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from src.errors import (
    ChunkingError,
    InvalidChunkingRulesError,
    InvalidDocumentError,
    ParserUnavailableError,
)
from src.ingestion import chunker, parser_registry, segmenter
from src.ingestion.chunk_report import NodeTally, percentile
from src.ingestion.chunker import (
    DEFAULT_MAX_CHUNK_CHARS,
    ChunkingRules,
    ChunkResult,
    chunk_documents,
    chunk_documents_with_report,
)
from src.ingestion.documents import HIDDEN_METADATA_KEYS
from src.ingestion.loader import ExitCode, load_repository_with_report
from src.ingestion.nodes import (
    HIDDEN_NODE_KEYS,
    NODE_ID_NAMESPACE,
    build_nodes,
    node_id,
    render_symbol,
)
from src.ingestion.parser_registry import GRAMMARS, PYTHON, GrammarSpec, grammar_for, load_language
from src.ingestion.rules import IngestionRules
from src.ingestion.segmenter import segment
from src.ingestion.spans import ParseStatus, Segmentation, Span, SpanKind
from src.ingestion.strategies import (
    LABEL_STRATEGIES,
    SYMBOL_STYLES,
    FallbackReason,
    Strategy,
    SymbolStyle,
    SyntacticSegmenter,
    fallback_reason,
    segmenter_for,
    strategy_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAP = DEFAULT_MAX_CHUNK_CHARS
# Corpus digest of the synthetic repository below at cap 120 (test_the_synthetic_digest_is_stable).
GOLDEN_SYNTHETIC_DIGEST = "2878752d66cd75d29a5a8cf1cc02156e52c171af3da03079489f320b4363bb0f"

Shape = tuple[str, str, bool]  # (kind, dotted name path, forced)

# rag_phase2.md Step 1: the function a naive splitter cuts in two.
DOC_STEP1_EXAMPLE = '''import time

import jwt


def validar_token(token: str, clave_secreta: str) -> bool:
    """Verifica que un token JWT sea válido y no haya expirado."""
    try:
        payload = jwt.decode(token, clave_secreta, algorithms=["HS256"])
        return payload["exp"] > time.time()
    except jwt.InvalidTokenError:
        return False
'''


def _segment(text: str, cap: int = CAP) -> Segmentation:
    return segment(text, PYTHON, max_chars=cap)


def _violations(text: str, cap: int = CAP) -> list[Violation]:
    return check_all(text, _segment(text, cap), cap, lambda: _segment(text, cap))


def _shape(text: str, cap: int = CAP) -> list[Shape]:
    return [(s.kind.value, ".".join(s.name_path), s.forced) for s in _segment(text, cap).spans]


def _texts(text: str, cap: int = CAP) -> list[str]:
    return [text[s.start : s.end] for s in _segment(text, cap).spans]


def _lines(text: str, span: Span) -> tuple[int, int]:
    return text.count("\n", 0, span.start) + 1, text.count("\n", 0, span.end) + 1


# --- parameters --------------------------------------------------------------------------------


def test_default_cap_is_the_agreed_1500_characters() -> None:
    assert ChunkingRules().max_chunk_chars == 1500 == DEFAULT_MAX_CHUNK_CHARS


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1500", None])
def test_invalid_cap_is_rejected(value: object) -> None:
    with pytest.raises(InvalidChunkingRulesError, match="positive integer"):
        ChunkingRules(max_chunk_chars=value)  # type: ignore[arg-type]
    with pytest.raises(InvalidChunkingRulesError):
        segment("x = 1\n", PYTHON, max_chars=value)  # type: ignore[arg-type]


def test_rules_are_frozen() -> None:
    rules = ChunkingRules()
    with pytest.raises(AttributeError):
        rules.max_chunk_chars = 10  # type: ignore[misc]


# --- parser registry ---------------------------------------------------------------------------


def test_python_is_the_only_registered_grammar() -> None:
    assert grammar_for("python") is PYTHON
    assert PYTHON.grammar_module == "tree_sitter_python"  # a label is not a grammar name


@pytest.mark.parametrize("label", sorted(IngestionRules().languages - {"python"}))
def test_labels_without_a_grammar_return_no_parser_as_a_value(label: str) -> None:
    assert grammar_for(label) is None


def test_every_declared_node_type_and_field_exists_in_the_grammar() -> None:
    language = load_language(PYTHON)

    assert {t for t in PYTHON.node_types if language.id_for_node_kind(t, True) is None} == set()
    assert language.field_id_for_name(PYTHON.name_field) is not None
    assert language.field_id_for_name(PYTHON.definition_field) is not None


def test_language_is_cached_and_parsers_are_not_shared() -> None:
    assert load_language(PYTHON) is load_language(PYTHON)
    assert parser_registry.new_parser(PYTHON) is not parser_registry.new_parser(PYTHON)


@pytest.mark.parametrize(
    ("module", "function"),
    [("tree_sitter_not_installed", "language"), ("tree_sitter_python", "no_such_function")],
)
def test_a_grammar_that_cannot_load_is_a_typed_actionable_error(module: str, function: str) -> None:
    broken = dataclasses.replace(PYTHON, grammar_module=module, language_function=function)

    with pytest.raises(ParserUnavailableError, match="make install") as raised:
        load_language(broken)

    assert raised.value.__cause__ is not None


_IMPORT_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import src.ingestion.parser_registry
    import src.ingestion.segmenter
    print(json.dumps({"grammar_loaded": "tree_sitter_python" in sys.modules}))
    """
)


def test_importing_the_registry_does_not_load_the_grammar(tmp_path: Path) -> None:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LC_ALL": "C.UTF-8"}

    result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
        [sys.executable, "-I", "-B", "-c", _IMPORT_CHILD, str(REPO_ROOT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"grammar_loaded": False}


# --- span model --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "path"),
    [(5, 5, ()), (-1, 3, ()), (4, 2, ()), (0, 3, ("",)), (0, 3, ["f"])],
)
def test_invalid_spans_are_rejected(start: int, end: int, path: object) -> None:
    with pytest.raises(ChunkingError):
        Span(start, end, SpanKind.FUNCTION, False, path)  # type: ignore[arg-type]


def test_segmentation_status_must_match_its_error_count() -> None:
    with pytest.raises(ChunkingError):
        Segmentation((), ParseStatus.CLEAN, error_regions=1)
    with pytest.raises(ChunkingError):
        Segmentation((), ParseStatus.HAS_ERRORS, error_regions=0)


# --- Step 1: every checker detects its violation ------------------------------------------------

TWO_FUNCTIONS = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
ALPHA = (0, TWO_FUNCTIONS.index("\n\n\n"))
BETA = (TWO_FUNCTIONS.index("def beta"), len(TWO_FUNCTIONS) - 1)


def _span(start: int, end: int, forced: bool = False) -> Span:
    return Span(start, end, SpanKind.MODULE_BLOCK, forced, ())


def _codes(violations: list[Violation]) -> set[str]:
    return {v.code for v in violations}


def test_the_correct_spans_satisfy_every_checker() -> None:
    spans = (_span(*ALPHA), _span(*BETA))
    run: Callable[[], Segmentation] = lambda: Segmentation(spans, ParseStatus.CLEAN)  # noqa: E731

    assert check_all(TWO_FUNCTIONS, run(), CAP, run) == []


def test_i1_detects_out_of_bounds_and_trailing_whitespace() -> None:
    text = "x = 1\n"

    assert _codes(check_bounds(text, [_span(0, 7)])) == {"I1"}
    assert _codes(check_bounds(text, [_span(0, 6)])) == {"I1"}


def test_i2_detects_overlap() -> None:
    assert _codes(check_order([_span(0, 10), _span(5, 20)])) == {"I2"}


def test_i3_detects_uncovered_and_doubly_covered_text() -> None:
    assert _codes(check_coverage(TWO_FUNCTIONS, [_span(*ALPHA)])) == {"I3"}
    assert _codes(check_coverage(TWO_FUNCTIONS, [_span(*ALPHA), _span(0, 3), _span(*BETA)])) == {
        "I3"
    }


def test_i4_detects_mid_line_boundaries_even_when_forced_within_a_short_line() -> None:
    text = "x = 1\ny = 2\n"

    assert _codes(check_line_boundaries(text, [_span(2, 5)], CAP)) == {"I4"}
    assert _codes(check_line_boundaries(text, [_span(0, 3)], CAP)) == {"I4"}
    assert _codes(check_line_boundaries(text, [_span(4, 5, forced=True)], CAP)) == {"I4"}
    assert check_line_boundaries(text, [_span(4, 5, forced=True)], 3) == []  # line > cap


def test_i5_detects_a_fitting_function_split_in_two() -> None:
    middle = TWO_FUNCTIONS.index("    return 1")
    spans = [_span(0, middle - 1), _span(middle, ALPHA[1]), _span(*BETA)]

    assert _codes(check_definitions_whole(TWO_FUNCTIONS, spans, CAP)) == {"I5"}


def test_i6_detects_a_span_over_the_cap() -> None:
    assert _codes(check_cap(TWO_FUNCTIONS, [_span(*ALPHA)], 5)) == {"I6"}


def test_i7_detects_unforced_splits_and_needless_forcing() -> None:
    middle = TWO_FUNCTIONS.index("    return 1")
    split = [_span(0, middle - 1), _span(middle, ALPHA[1]), _span(*BETA)]

    assert _codes(check_forced(TWO_FUNCTIONS, split, 15)) == {"I7"}  # alpha does not fit
    needless = [_span(*ALPHA, forced=True), _span(*BETA)]
    assert _codes(check_forced(TWO_FUNCTIONS, needless, CAP)) == {"I7"}  # everything fits


def test_i8_detects_non_determinism() -> None:
    calls: list[int] = []

    def run() -> Segmentation:
        calls.append(1)
        return Segmentation((_span(0, len(calls)),), ParseStatus.CLEAN)

    assert _codes(check_deterministic(run)) == {"I8"}


def test_the_oracle_does_not_apply_to_invalid_python() -> None:
    assert definition_regions("def broken(:\n    pass\n") is None


# --- Step 1: blind splitting breaks coherence, the segmenter does not ---------------------------


def test_naive_four_line_split_cuts_validar_token_in_two() -> None:
    naive = naive_line_split(DOC_STEP1_EXAMPLE, 4)

    violations = check_definitions_whole(DOC_STEP1_EXAMPLE, naive, CAP)
    assert [v.code for v in violations] == ["I5"]
    assert violations[0].detail.startswith("'validar_token' fits")


def test_naive_character_split_cuts_validar_token_mid_line() -> None:
    naive = naive_char_split(DOC_STEP1_EXAMPLE, 120)

    assert "I5" in _codes(check_definitions_whole(DOC_STEP1_EXAMPLE, naive, CAP))
    assert "I4" in _codes(check_line_boundaries(DOC_STEP1_EXAMPLE, naive, CAP))


def test_the_segmenter_keeps_validar_token_whole() -> None:
    assert _violations(DOC_STEP1_EXAMPLE) == []
    assert _shape(DOC_STEP1_EXAMPLE) == [
        ("module_block", "", False),
        ("function", "validar_token", False),
    ]
    assert _texts(DOC_STEP1_EXAMPLE)[1].startswith("def validar_token(")
    assert _texts(DOC_STEP1_EXAMPLE)[1].endswith("return False")


def test_the_doc_test_function_stays_whole_is_vacuous() -> None:
    # rag_phase2.md Step 7's assertions, verbatim in meaning.
    def doc_assertions(contents: list[str]) -> bool:
        return any("def alpha" in c and "return 1" in c for c in contents) and any(
            "def beta" in c and "return 2" in c for c in contents
        )

    assert doc_assertions([TWO_FUNCTIONS])  # a "chunker" returning the whole file passes
    assert doc_assertions(_texts(TWO_FUNCTIONS))
    # What it should have asserted: one span per function.
    assert _shape(TWO_FUNCTIONS) == [("function", "alpha", False), ("function", "beta", False)]


# --- Step 2: segmentation cases -----------------------------------------------------------------

DECORATED = """@cache
def single():
    return 1


@first
# a comment between decorators
@second
def stacked():
    return 2


@route(
    "/path",
    methods=["GET"],
)
def multiline():
    return 3
"""

ASYNC = "async def fetch(client):\n    return await client.get()\n"

CLASS_FITS = """class Point:
    \"\"\"A point.\"\"\"

    x: int = 0

    def norm(self):
        return abs(self.x)
"""

BIG_CLASS = """class Service(Base):
    \"\"\"Docstring of the service.\"\"\"

    name: str = "svc"
    retries = 3

    def start(self):
        return self.name

    timeout = 10
    delay = 2

    def stop(self):
        return None

    trailing = True
"""

CLASS_WITHOUT_METHODS = "class Settings:\n" + "".join(
    f"    option_{i}: int = {i}\n" for i in range(12)
)

NESTED_CLASS = """class Outer:
    \"\"\"Outer doc.\"\"\"

    def first(self):
        return 1

    class Small:
        flag = True

    class Large:
        \"\"\"Large inner class doc.\"\"\"

        def inner_method(self):
            value = 1
            return value
"""

HUGE_FUNCTION = """def process(items):
    \"\"\"Process every item.\"\"\"
    total = 0
    try:
        for item in items:
            if item > 0:
                total += item
            elif item < 0:
                total -= item
            else:
                continue
    except ValueError as error:
        raise RuntimeError("bad item") from error
    finally:
        cleanup()
    with open_resource() as resource:
        resource.write(total)
    return total
"""

HUGE_STATEMENT = (
    "import os\n\nTABLE = {\n"
    + "".join(f"    'key_{i}': {i},\n" for i in range(15))
    + "}\n\nVERSION = 1\n"
)

HUGE_LINE = "TOTAL = " + " + ".join(f"value_{i}" for i in range(40)) + "\n"

STATEMENT_IN_FUNCTION = (
    "def build():\n    mapping = {\n"
    + "".join(f"        'k{i}': {i},\n" for i in range(12))
    + "    }\n    return mapping\n"
)

MODULE_CODE = """\"\"\"Module docstring.\"\"\"

import os
import sys

LIMIT = 10


def main():
    return LIMIT


if __name__ == "__main__":
    sys.exit(main())
"""

DEFS_IN_COMPOUND = """if TYPE_CHECKING:
    def hinted() -> None: ...
try:
    import fast
except ImportError:
    def fast() -> None: ...
"""

COMMENTS = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
def first():
    return 1


# Section banner, separated by a blank line.

# Attached directly.
def second():
    x = 1  # end-of-line comment
    return x
    # indented comment at the end of the block


CONSTANT = 1
# trailing module comment
"""

SYNTAX_ERRORS = {
    "unclosed_paren": "def ok():\n    return 1\n\nx = (1,\n\ndef later():\n    return 3\n",
    "bad_parameters": (
        "def ok():\n    return 1\n\ndef broken(:\n    return 2\n\ndef ok2():\n    return 3\n"
    ),
    "missing_colon": (
        "def ok():\n    return 1\n\nclass Broken\n    pass\n\ndef ok2():\n    return 3\n"
    ),
}

NON_ASCII = 'def saludo():\n    """Añade un saludo: ñandú, café, 🎉."""\n    return "¡hola!"\n'

REPEATED = "def twin():\n    return 1\n\n\ndef twin():\n    return 1\n"

NO_TRAILING_NEWLINE = "def last():\n    return 1"

LINE_SEPARATOR = chr(0x2028)  # written as an escape: RUF001 forbids the literal
SEPARATORS = (
    "def f():\n    s = 'a"
    + LINE_SEPARATOR
    + "b'\n"
    + "    # form\x0cfeed\n    return s\n\x0c\ndef g():\n    pass\n"
)

CASES: dict[str, tuple[str, int]] = {
    "validar_token": (DOC_STEP1_EXAMPLE, CAP),
    "small_functions": (TWO_FUNCTIONS, CAP),
    "decorated": (DECORATED, CAP),
    "async": (ASYNC, CAP),
    "class_fits": (CLASS_FITS, CAP),
    "big_class": (BIG_CLASS, 120),
    "class_without_methods": (CLASS_WITHOUT_METHODS, 100),
    "nested_class": (NESTED_CLASS, 130),
    "huge_function": (HUGE_FUNCTION, 160),
    "huge_statement": (HUGE_STATEMENT, 100),
    "huge_line": (HUGE_LINE, 80),
    "statement_in_function": (STATEMENT_IN_FUNCTION, 90),
    "module_code": (MODULE_CODE, CAP),
    "defs_in_compound": (DEFS_IN_COMPOUND, CAP),
    "comments": (COMMENTS, CAP),
    "empty": ("", CAP),
    "whitespace_only": ("  \n\t\n\n", CAP),
    "comment_only": ("# one\n# two\n", CAP),
    "non_ascii": (NON_ASCII, CAP),
    "repeated": (REPEATED, CAP),
    "no_trailing_newline": (NO_TRAILING_NEWLINE, CAP),
    "separators": (SEPARATORS, CAP),
    **{f"syntax_{name}": (text, CAP) for name, text in SYNTAX_ERRORS.items()},
    "stray_indent": ("def ok():\n    return 1\n\n   y = 2\ndef ok2():\n    return 3\n", CAP),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_case_satisfies_every_invariant(case: str) -> None:
    text, cap = CASES[case]

    assert _violations(text, cap) == []


def test_small_functions_are_one_span_each_and_never_merged() -> None:
    assert [(_lines(TWO_FUNCTIONS, s)) for s in _segment(TWO_FUNCTIONS).spans] == [(1, 2), (5, 6)]


def test_decorators_belong_to_their_definition() -> None:
    texts = _texts(DECORATED)

    assert _shape(DECORATED) == [
        ("function", "single", False),
        ("function", "stacked", False),
        ("function", "multiline", False),
    ]
    assert texts[0].startswith("@cache\n")
    assert texts[1].startswith("@first\n# a comment between decorators\n@second\n")
    assert texts[2].startswith("@route(\n")


def test_async_definition_is_a_function() -> None:
    assert _shape(ASYNC) == [("function", "fetch", False)]


def test_a_class_that_fits_is_one_span() -> None:
    assert _shape(CLASS_FITS) == [("class", "Point", False)]


def test_a_big_class_splits_into_header_methods_and_class_blocks() -> None:
    assert _shape(BIG_CLASS, 120) == [
        ("class_header", "Service", False),
        ("method", "Service.start", False),
        ("class_block", "Service", False),
        ("method", "Service.stop", False),
        ("class_block", "Service", False),
    ]
    texts = _texts(BIG_CLASS, 120)
    assert texts[0].endswith("retries = 3")
    assert texts[2] == "    timeout = 10\n    delay = 2"
    assert texts[4] == "    trailing = True"


def test_a_big_class_without_methods_is_a_forced_header_split() -> None:
    shape = _shape(CLASS_WITHOUT_METHODS, 100)

    assert len(shape) > 1
    assert set(shape) == {("class_header", "Settings", True)}


def test_nested_classes_recurse_like_methods() -> None:
    assert _shape(NESTED_CLASS, 130) == [
        ("class_header", "Outer", False),
        ("method", "Outer.first", False),
        ("class", "Outer.Small", False),
        ("class_header", "Outer.Large", False),
        ("method", "Outer.Large.inner_method", False),
    ]


def test_a_huge_function_is_split_at_statements_keeping_its_header_with_the_first_part() -> None:
    spans = _segment(HUGE_FUNCTION, 160).spans
    texts = _texts(HUGE_FUNCTION, 160)

    assert len(spans) > 1
    assert {(s.kind, s.name_path, s.forced) for s in spans} == {
        (SpanKind.FUNCTION, ("process",), True)
    }
    assert texts[0].startswith('def process(items):\n    """Process every item."""')
    assert all(t.strip() for t in texts)


def test_a_huge_statement_steps_down_to_lines_and_only_its_pieces_are_forced() -> None:
    shape = _shape(HUGE_STATEMENT, 100)
    texts = _texts(HUGE_STATEMENT, 100)

    assert shape[0] == ("module_block", "", False)
    assert texts[0] == "import os"
    assert shape[-1] == ("module_block", "", False)
    assert texts[-1] == "VERSION = 1"
    assert {f for _, _, f in shape[1:-1]} == {True}
    assert texts[1].startswith("TABLE = {")


def test_a_huge_line_is_split_into_characters_at_spaces() -> None:
    spans = _segment(HUGE_LINE, 80).spans
    texts = _texts(HUGE_LINE, 80)

    assert len(spans) > 1
    assert all(s.forced for s in spans)
    assert all(len(t) <= 80 and not t[0].isspace() and not t[-1].isspace() for t in texts)
    assert " ".join(texts) == HUGE_LINE.strip()  # cuts happen at single spaces


def test_a_huge_statement_inside_a_function_is_split_into_forced_lines() -> None:
    shape = _shape(STATEMENT_IN_FUNCTION, 90)

    assert len(shape) > 1
    assert set(shape) == {("function", "build", True)}


@pytest.mark.parametrize("extra", [0, 1])
def test_exactly_the_cap_fits_and_one_more_character_does_not(extra: int) -> None:
    text = "def edge():\n    return 'abcdefghij'\n"
    cap = len(text.rstrip()) - extra

    shape = _shape(text, cap)

    assert _violations(text, cap) == []
    assert shape == (
        [("function", "edge", False)] if extra == 0 else [("function", "edge", True)] * 2
    )


def test_module_code_forms_blocks_around_definitions() -> None:
    texts = _texts(MODULE_CODE)

    assert _shape(MODULE_CODE) == [
        ("module_block", "", False),
        ("function", "main", False),
        ("module_block", "", False),
    ]
    assert texts[0].endswith("LIMIT = 10")
    assert texts[2].startswith('if __name__ == "__main__":')


def test_definitions_inside_module_level_compound_statements_are_module_code() -> None:
    assert _shape(DEFS_IN_COMPOUND) == [("module_block", "", False)]


def test_comment_policies() -> None:
    texts = _texts(COMMENTS)

    assert _shape(COMMENTS) == [
        ("module_block", "", False),
        ("function", "first", False),
        ("function", "second", False),
        ("module_block", "", False),
    ]
    assert texts[0] == "#!/usr/bin/env python\n# -*- coding: utf-8 -*-"  # never attached
    assert texts[1].startswith("def first():")
    assert texts[2].startswith("# Section banner")  # detached runs join the definition
    assert "x = 1  # end-of-line comment" in texts[2]
    assert texts[2].endswith("# indented comment at the end of the block")
    assert texts[3] == "CONSTANT = 1\n# trailing module comment"


def test_leading_comments_that_would_push_a_fitting_definition_over_the_cap_are_split_off() -> None:
    text = "# " + "x" * 15 + "\ndef short():\n    return 1\n"
    cap = len("def short():\n    return 1")  # the comment fits alone, not with the definition

    assert _shape(text, cap) == [("module_block", "", False), ("function", "short", False)]
    assert _violations(text, cap) == []


@pytest.mark.parametrize("text", ["", "  \n\t\n\n"])
def test_empty_and_whitespace_only_texts_have_no_spans(text: str) -> None:
    assert _segment(text) == Segmentation((), ParseStatus.CLEAN)


def test_comment_only_text_is_one_module_block() -> None:
    assert _shape("# one\n# two\n") == [("module_block", "", False)]


@pytest.mark.parametrize("name", sorted(SYNTAX_ERRORS))
def test_syntax_errors_are_a_status_and_healthy_definitions_survive(name: str) -> None:
    text = SYNTAX_ERRORS[name]
    result = _segment(text)

    assert result.status is ParseStatus.HAS_ERRORS
    assert result.error_regions == 1
    assert definition_regions(text) is None  # the ast oracle does not apply here
    assert result.spans[0].kind is SpanKind.FUNCTION
    assert result.spans[0].name_path == ("ok",)
    assert any(s.kind is SpanKind.MODULE_BLOCK for s in result.spans)


def test_a_stray_indent_is_not_flagged_by_tree_sitter_although_python_rejects_it() -> None:
    text = CASES["stray_indent"][0]

    assert _segment(text).status is ParseStatus.CLEAN
    with pytest.raises(SyntaxError):
        ast.parse(text)


def test_the_cap_counts_characters_not_bytes() -> None:
    cap = len(NON_ASCII.rstrip())

    assert len(NON_ASCII.rstrip().encode("utf-8")) > cap
    assert _shape(NON_ASCII, cap) == [("function", "saludo", False)]


def test_repeated_identical_definitions_keep_their_own_offsets() -> None:
    first, second = _segment(REPEATED).spans

    assert (first.start, second.start) == (0, REPEATED.index("def twin", 1))
    assert REPEATED[first.start : first.end] == REPEATED[second.start : second.end]


def test_text_without_a_trailing_newline_ends_its_last_span_at_the_end() -> None:
    assert _segment(NO_TRAILING_NEWLINE).spans[-1].end == len(NO_TRAILING_NEWLINE)


def test_form_feed_and_line_separator_are_not_line_breaks() -> None:
    spans = _segment(SEPARATORS).spans

    assert _shape(SEPARATORS) == [("function", "f", False), ("function", "g", False)]
    assert _lines(SEPARATORS, spans[0]) == (1, 4)
    assert _lines(SEPARATORS, spans[1]) == (6, 7)


# --- Step 2: purity --------------------------------------------------------------------------


def test_the_whole_file_is_parsed_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = parser_registry.new_parser

    class CountingParser:
        def __init__(self, spec: GrammarSpec) -> None:
            self._parser = real(spec)

        def parse(self, source: bytes) -> object:
            calls.append(len(source))
            return self._parser.parse(source)

    monkeypatch.setattr(segmenter, "new_parser", CountingParser)

    segment(HUGE_FUNCTION + BIG_CLASS, PYTHON, max_chars=100)

    assert calls == [len((HUGE_FUNCTION + BIG_CLASS).encode("utf-8"))]


def test_segmenting_opens_no_files(monkeypatch: pytest.MonkeyPatch) -> None:
    load_language(PYTHON)  # the grammar is loaded once per process, outside this check

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the segmenter opened a file")

    monkeypatch.setattr(builtins, "open", forbidden)

    assert _segment(BIG_CLASS, 120).status is ParseStatus.CLEAN


def test_logs_counts_never_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="src.ingestion.segmenter"):
        _segment(DOC_STEP1_EXAMPLE)

    (record,) = [r for r in caplog.records if r.getMessage() == "chunking.segmented"]
    assert record.__dict__["spans"] == 2
    assert record.__dict__["status"] == "clean"
    assert "validar_token" not in json.dumps({k: str(v) for k, v in record.__dict__.items()})


# --- Step 3: language dispatch ------------------------------------------------------------------


def test_every_loader_label_has_an_explicit_strategy() -> None:
    # A new loader label fails here until its strategy is decided (data in strategies.py).
    assert IngestionRules().languages == set(GRAMMARS) | set(LABEL_STRATEGIES)
    assert not set(GRAMMARS) & set(LABEL_STRATEGIES)  # the grammar data is not duplicated


def test_the_strategy_map_is_the_agreed_one() -> None:
    assert {label: strategy_for(label).value for label in sorted(IngestionRules().languages)} == {
        "c": "text",
        "cpp": "text",
        "csharp": "text",
        "dockerfile": "config",
        "go": "text",
        "java": "text",
        "javascript": "text",
        "makefile": "config",
        "markdown": "markdown",
        "php": "text",
        "python": "syntactic",
        "ruby": "text",
        "rust": "text",
        "toml": "config",
        "typescript": "text",
        "yaml": "config",
    }


def test_an_unknown_label_gets_generic_text() -> None:
    assert strategy_for("cobol") is Strategy.TEXT


def test_every_non_syntactic_strategy_carries_no_parser() -> None:
    assert {s: fallback_reason(s) for s in Strategy} == {
        Strategy.SYNTACTIC: None,
        Strategy.MARKDOWN: FallbackReason.NO_PARSER,
        Strategy.CONFIG: FallbackReason.NO_PARSER,
        Strategy.TEXT: FallbackReason.NO_PARSER,
    }


def test_the_syntactic_segmenter_adapts_segment_without_changing_it() -> None:
    adapted = SyntacticSegmenter(PYTHON)

    assert adapted(BIG_CLASS, max_chars=120) == segment(BIG_CLASS, PYTHON, max_chars=120)


def test_every_strategy_has_a_symbol_style() -> None:
    assert set(SYMBOL_STYLES) == set(Strategy)
    assert SYMBOL_STYLES[Strategy.SYNTACTIC].separator == "."
    assert SYMBOL_STYLES[Strategy.MARKDOWN].max_chars == 120


# --- Step 4: from spans to nodes ----------------------------------------------------------------

DOCUMENT_KEYS = ["file_path", "language", "size_bytes", "num_lines", "content_hash", "directory"]


def _document(text: str, path: str = "src/app.py", language: str = "python") -> Document:
    return Document(
        id_=path,
        text=text,
        metadata={
            "file_path": path,
            "language": language,
            "size_bytes": len(text.encode("utf-8")),
            "num_lines": text.count("\n") + (0 if text.endswith("\n") else 1),
            "content_hash": "0" * 64,
            "directory": path.rpartition("/")[0],
        },
        excluded_embed_metadata_keys=list(HIDDEN_METADATA_KEYS),
        excluded_llm_metadata_keys=list(HIDDEN_METADATA_KEYS),
    )


def _nodes(text: str, cap: int = CAP, path: str = "src/app.py") -> tuple[TextNode, ...]:
    document = _document(text, path)
    return build_nodes(document, _segment(text, cap), SYMBOL_STYLES[Strategy.SYNTACTIC])


def _symbols(text: str, cap: int = CAP) -> list[str | None]:
    return [n.metadata.get("symbol") for n in _nodes(text, cap)]


def test_node_ids_are_uuid5_of_the_fixed_namespace() -> None:
    # Golden value: a refactor must not silently change node IDs.
    assert node_id("src/app.py", 0, 21) == "7f3e8e16-5ecf-57b4-b2ce-77cdfffd50a4"
    assert uuid.UUID("41a05dbb-07dd-4b83-840e-8c25648f281a") == NODE_ID_NAMESPACE


def test_node_metadata_keys_types_and_order() -> None:
    function, module = _nodes("def f():\n    return 1\n\n\nx = 1\n")

    assert list(function.metadata) == [*DOCUMENT_KEYS, "symbol", "start_line", "end_line"]
    assert list(module.metadata) == [*DOCUMENT_KEYS, "start_line", "end_line"]  # no symbol
    assert {k: type(v).__name__ for k, v in function.metadata.items()} == {
        "file_path": "str",
        "language": "str",
        "size_bytes": "int",
        "num_lines": "int",
        "content_hash": "str",
        "directory": "str",
        "symbol": "str",
        "start_line": "int",
        "end_line": "int",
    }


@pytest.mark.parametrize(
    ("text", "cap", "expected"),
    [
        (TWO_FUNCTIONS, CAP, ["alpha", "beta"]),
        (BIG_CLASS, 120, ["Service", "Service.start", "Service", "Service.stop", "Service"]),
        (
            NESTED_CLASS,
            130,
            ["Outer", "Outer.first", "Outer.Small", "Outer.Large", "Outer.Large.inner_method"],
        ),
        (HUGE_FUNCTION, 160, ["process"] * len(_segment(HUGE_FUNCTION, 160).spans)),
        (MODULE_CODE, CAP, [None, "main", None]),
        (SYNTAX_ERRORS["unclosed_paren"], CAP, ["ok", None]),
    ],
    ids=["functions", "class_header_and_blocks", "nested", "forced_fragments", "module", "error"],
)
def test_symbol_comes_from_the_name_path(text: str, cap: int, expected: list[str | None]) -> None:
    assert _symbols(text, cap) == expected


@pytest.mark.parametrize(
    ("path", "limit", "expected"),
    [
        ((), 120, None),
        (("Title", "Decision"), 120, "Title > Decision"),
        (("A" * 50, "B" * 50, "C" * 50), 120, "… > " + "B" * 50 + " > " + "C" * 50),
        (("A" * 50, "X" * 200), 120, "…" + "X" * 119),
    ],
)
def test_bounded_symbols_keep_the_most_specific_end(
    path: tuple[str, ...], limit: int, expected: str | None
) -> None:
    symbol = render_symbol(path, SymbolStyle(" > ", max_chars=limit))

    assert symbol == expected
    assert symbol is None or len(symbol) <= limit


@pytest.mark.parametrize("case", sorted(CASES))
def test_node_text_offsets_and_lines_match_an_independent_oracle(case: str) -> None:
    text, cap = CASES[case]

    for node, span in zip(_nodes(text, cap), _segment(text, cap).spans, strict=True):
        assert node.text == text[span.start : span.end]
        assert (node.start_char_idx, node.end_char_idx) == (span.start, span.end)
        assert node.metadata["start_line"] == text[: span.start].count("\n") + 1
        assert node.metadata["end_line"] == text[: span.end - 1].count("\n") + 1


def test_first_and_last_lines_without_trailing_newline_and_repeated_functions() -> None:
    assert [(n.metadata["start_line"], n.metadata["end_line"]) for n in _nodes(REPEATED)] == [
        (1, 2),
        (5, 6),
    ]
    (last,) = _nodes(NO_TRAILING_NEWLINE)
    assert (last.metadata["start_line"], last.metadata["end_line"]) == (1, 2)
    assert last.end_char_idx == len(NO_TRAILING_NEWLINE)
    (accents,) = _nodes(NON_ASCII)
    assert (accents.metadata["start_line"], accents.metadata["end_line"]) == (1, 3)


def test_views_show_only_file_path_language_and_symbol() -> None:
    (node,) = _nodes("def f():\n    return 1\n")
    head = "file_path: src/app.py\nlanguage: python\nsymbol: f\n\n"

    assert node.get_content(metadata_mode=MetadataMode.EMBED) == head + node.text
    assert node.get_content(metadata_mode=MetadataMode.LLM) == head + node.text
    assert node.get_content(metadata_mode=MetadataMode.NONE) == node.text
    assert node.excluded_embed_metadata_keys == list(HIDDEN_NODE_KEYS)
    assert node.excluded_embed_metadata_keys is not node.excluded_llm_metadata_keys


def test_only_the_source_relationship_points_to_the_document() -> None:
    first, second = _nodes(TWO_FUNCTIONS)

    assert set(first.relationships) == {NodeRelationship.SOURCE}
    assert first.ref_doc_id == second.ref_doc_id == "src/app.py"
    assert first.metadata is not second.metadata
    assert first.excluded_embed_metadata_keys is not second.excluded_embed_metadata_keys


def test_vector_store_round_trip_keeps_hash_id_offsets_and_key_order() -> None:
    for node in _nodes(BIG_CLASS, 120):
        stored = node_to_metadata_dict(node, remove_text=True, flat_metadata=True)
        back = metadata_dict_to_node(stored, text=node.text)

        assert (back.node_id, back.hash) == (node.node_id, node.hash)
        assert list(back.metadata) == list(node.metadata)
        assert isinstance(back, TextNode)
        assert (back.start_char_idx, back.end_char_idx) == (node.start_char_idx, node.end_char_idx)
        assert {"_node_content", "_node_type", "document_id", "doc_id", "ref_doc_id"} <= set(stored)


def test_flat_metadata_check_rejects_lists_but_accepts_none() -> None:
    # Installed llama-index-core 0.14.26 behaviour, recorded for Phases 3-4: None passes the
    # flat check (so the contract omits keys instead of storing None).
    node = TextNode(text="x", metadata={"symbol": None})
    assert "symbol" in node_to_metadata_dict(node, flat_metadata=True)
    with pytest.raises(ValueError, match="must be one of"):
        node_to_metadata_dict(TextNode(text="x", metadata={"tags": ["a"]}), flat_metadata=True)


def test_identical_pieces_on_the_same_line_share_a_hash_but_not_an_id() -> None:
    # Known risk for Phase 4: fusion deduplicates on hash (text + metadata), so two forced
    # pieces with the same text on the same line collapse there. P4 adds no key to avoid it.
    text = "VALUES = " + " ".join(["same"] * 60) + "\n"
    nodes = [n for n in _nodes(text, 40) if n.text == "same same same same same same same same"]

    assert len(nodes) >= 2
    assert nodes[0].hash == nodes[1].hash
    assert nodes[0].node_id != nodes[1].node_id


# --- Step 5: fallbacks --------------------------------------------------------------------------

MD_MIXED = (
    "---\ntitle: front matter\n---\nIntro paragraph.\n\n# Title\n\n## Empty\n\n## Usage\n\n"
    "Run it:\n\n```bash\n# not a heading\nmake run\n```\n\nSetext heading\n--------------\n\n"
    "Text under setext.\n\n---\n\nAfter a thematic break.\n"
)
MD_BIG_SECTION = "## Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(6))
MD_FENCES = (
    "# Fences\n\nIntro.\n\n~~~python\n# comment, not a heading\nx = 1\n~~~\n\n"
    "- item\n\n    ```\n    # still code\n    ```\n\n````\n```\n# inside a longer fence\n````\n"
    "\n## After\n\nDone.\n"
)
MD_UNCLOSED = "# Start\n\nText.\n\n```\n# never closed\n## also code\n"
MD_EMPTY_HEADING = "#\n\nBody of an empty heading.\n\n## Named\n\nMore.\n"
MD_NON_ASCII = "# Título\n\nÑandú y café: 🎉.\n"
YAML_CI = (
    "# CI config\nname: ci\n\non:\n  push:\n    branches: [main]\n\njobs:\n"
    "  lint:\n    runs-on: ubuntu\n    steps:\n      - run: make lint\n"
    "  test:\n    runs-on: ubuntu\n    steps:\n      - name: tests\n        run: |\n"
    "          make test\n\n          make check\n"
    "  build:\n    runs-on: ubuntu\n    steps:\n      - run: docker build .\n"
)
YAML_SCALAR = "script: |\n" + "".join(
    f"  echo line {i}\n" + ("\n" if i % 3 == 2 else "") for i in range(12)
)
YAML_DOCUMENTS = "---\na: 1\n---\nb: 2\n...\n"
TOML_STRING = (
    '[project]\nname = "x"\ndescription = """\n[not a table]\n"""\n\n'
    '[tool.ruff]\nline-length = 100\nselect = [\n  "E",\n  "F",\n]\n'
)
TOML_BIG_TABLE = "[deps]\n" + "\n".join(f'pkg{i} = "{i}.0"\n' for i in range(10))
DOCKERFILE = (
    "# syntax=docker/dockerfile:1\nFROM python:3.11 AS build\nRUN apt-get update && \\\n"
    "    apt-get install -y gcc\nRUN <<EOF\necho one\nFROM is not an instruction here\nEOF\n\n"
    'FROM python:3.11-slim\nCOPY --from=build /app /app\nCMD ["python"]\n'
)
MAKEFILE = (
    ".PHONY: all\n\n# build everything\nall: build test\n\nbuild:\n\tgcc -o app main.c\n\n"
    "\tstrip app\n\ndefine HELP\nUsage: make\n\nTargets: all\nendef\n\n"
    "ifeq ($(CI),1)\nFLAGS = -O2 \\\n  -g\nelse\nFLAGS = -O0\nendif\n"
)
TEXT_JS = "function a() {\n  return 1;\n}\n\nfunction b() {\n  return 2;\n}\n"
LONG_WORDS = " ".join(f"w{i}" for i in range(400))

FALLBACK_CASES: dict[str, tuple[str, str, int]] = {
    "md_mixed": ("markdown", MD_MIXED, 200),
    "md_big_section": ("markdown", MD_BIG_SECTION, 200),
    "md_fences": ("markdown", MD_FENCES, 120),
    "md_unclosed": ("markdown", MD_UNCLOSED, CAP),
    "md_no_headings": ("markdown", "Just text.\n\nMore text.\n", CAP),
    "md_empty_heading": ("markdown", MD_EMPTY_HEADING, CAP),
    "md_non_ascii": ("markdown", MD_NON_ASCII, CAP),
    "md_long_line": ("markdown", "# T\n\n" + LONG_WORDS + "\n", 300),
    "yaml_ci": ("yaml", YAML_CI, 200),
    "yaml_scalar": ("yaml", YAML_SCALAR, 60),
    "yaml_documents": ("yaml", YAML_DOCUMENTS, CAP),
    "toml_string": ("toml", TOML_STRING, 60),
    "toml_big_table": ("toml", TOML_BIG_TABLE, 60),
    "dockerfile": ("dockerfile", DOCKERFILE, 120),
    "makefile": ("makefile", MAKEFILE, 80),
    "text_js": ("javascript", TEXT_JS, 40),
    "text_long_line": ("go", LONG_WORDS + "\n", 300),
    "text_unknown_label": ("cobol", TEXT_JS, CAP),
}


def _fallback(label: str, text: str, cap: int) -> Segmentation:
    return segmenter_for(label)[1](text, max_chars=cap)


def _natural_units(label: str, text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(units that must stay whole when they fit, units whose size can justify forcing)."""
    if label == "markdown":
        fences = fence_regions(text)
        return fences, fences + markdown_section_regions(text)
    blocks = config_top_regions(text, label)
    return blocks, blocks


def _fallback_shape(label: str, text: str, cap: int) -> list[tuple[str, str, bool]]:
    result = _fallback(label, text, cap)
    return [(s.kind.value, " > ".join(s.name_path), s.forced) for s in result.spans]


def _fallback_texts(label: str, text: str, cap: int) -> list[str]:
    return [text[s.start : s.end] for s in _fallback(label, text, cap).spans]


@pytest.mark.parametrize("case", sorted(FALLBACK_CASES))
def test_every_fallback_case_satisfies_its_invariants(case: str) -> None:
    label, text, cap = FALLBACK_CASES[case]
    result = _fallback(label, text, cap)
    whole, sized = _natural_units(label, text)

    violations = (
        check_shared(text, result, cap, lambda: _fallback(label, text, cap))
        + check_units_whole(text, result.spans, whole, cap)
        + check_forced_where_needed(text, result.spans, sized, cap)
    )

    assert violations == []
    assert result.status is ParseStatus.CLEAN


def test_fallback_checkers_detect_their_violations() -> None:
    text = "```\na\nb\n```\n"
    split = [_span(0, 5), _span(6, 11)]

    assert _codes(check_units_whole(text, split, fence_regions(text), CAP)) == {"F1"}
    assert _codes(check_forced_where_needed(text, [_span(0, 11, forced=True)], [], CAP)) == {"F2"}


def test_markdown_sections_follow_headings_outside_fences() -> None:
    assert _fallback_shape("markdown", MD_MIXED, 200) == [
        ("section", "", False),  # front matter and the preamble
        ("section", "Title > Usage", False),  # two heading-only sections joined it
        ("section", "Title > Setext heading", False),  # '---' after a blank line is a rule
    ]
    texts = _fallback_texts("markdown", MD_MIXED, 200)
    assert texts[1].startswith("# Title\n\n## Empty\n\n## Usage")
    assert "# not a heading" in texts[1]


def test_a_big_section_is_split_at_paragraphs_keeping_its_heading_with_the_first() -> None:
    shape = _fallback_shape("markdown", MD_BIG_SECTION, 200)
    texts = _fallback_texts("markdown", MD_BIG_SECTION, 200)

    assert len(shape) > 1
    assert set(shape) == {("section", "Big", True)}
    assert texts[0].startswith("## Big\n\nParagraph 0")


def test_fences_hide_headings_and_a_fitting_fence_is_never_split() -> None:
    paths = [p for _, p, _ in _fallback_shape("markdown", MD_FENCES, 120)]

    assert paths[-1] == "Fences > After"  # "## After" sits under "# Fences"
    assert set(paths) == {"Fences", "Fences > After"}  # no '#' inside a fence opened a section


def test_an_unclosed_fence_runs_to_the_end() -> None:
    assert _fallback_shape("markdown", MD_UNCLOSED, CAP) == [("section", "Start", False)]


def test_text_without_headings_is_one_preamble_section() -> None:
    assert _fallback_shape("markdown", "Just text.\n\nMore text.\n", CAP) == [
        ("section", "", False)
    ]


def test_an_empty_heading_adds_nothing_to_the_path() -> None:
    assert _fallback_shape("markdown", MD_EMPTY_HEADING, CAP) == [
        ("section", "", False),
        ("section", "Named", False),
    ]


def test_yaml_steps_down_indentation_levels_keeping_jobs_whole() -> None:
    texts = _fallback_texts("yaml", YAML_CI, 200)

    assert _fallback_shape("yaml", YAML_CI, 200) == [
        ("config_block", "", False),
        ("config_block", "", True),
        ("config_block", "", True),
    ]
    assert texts[0].startswith("# CI config\nname: ci")  # the comment stays with its block
    assert texts[1].startswith("jobs:\n  lint:")  # the parent key keeps its first child
    lines = YAML_CI.split("\n")
    after_jobs = lines.index("jobs:") + 1
    starts = [
        i
        for i, line in enumerate(lines)
        if i >= after_jobs and line.startswith("  ") and line[2] != " "
    ]
    ends = [*starts[1:], len(lines)]
    jobs = ["\n".join(lines[a:b]).rstrip() for a, b in zip(starts, ends, strict=True)]
    assert [job.split(":")[0].strip() for job in jobs] == ["lint", "test", "build"]
    assert min(len(job) for job in jobs) > 40  # whole jobs, not their first line
    for job in jobs:  # every job fits, so each one is whole in exactly one span
        assert sum(job in t for t in texts) == 1


def test_a_yaml_block_scalar_splits_at_blank_lines_not_at_its_text() -> None:
    texts = _fallback_texts("yaml", YAML_SCALAR, 60)

    assert len(texts) > 1
    assert all(not t.strip().startswith("echo line 1\n") for t in texts[1:])


def test_yaml_documents_are_top_level_blocks() -> None:
    assert _fallback_texts("yaml", YAML_DOCUMENTS, 8) == ["---\na: 1", "---\nb: 2", "..."]


def test_a_bracket_inside_a_toml_string_is_not_a_table() -> None:
    assert _fallback_texts("toml", TOML_STRING, 60) == [
        TOML_STRING[: TOML_STRING.index("\n\n[tool")],
        TOML_STRING[TOML_STRING.index("[tool") :].rstrip(),
    ]


def test_an_oversized_toml_table_splits_between_statements() -> None:
    shape = _fallback_shape("toml", TOML_BIG_TABLE, 60)

    assert len(shape) > 1
    assert {f for _, _, f in shape} == {True}


def test_dockerfile_instructions_keep_continuations_and_heredocs_and_stages_never_merge() -> None:
    texts = _fallback_texts("dockerfile", DOCKERFILE, 1000)

    assert len(texts) == 2  # one chunk per stage although both fit together
    assert texts[0].startswith("# syntax=docker/dockerfile:1\nFROM python:3.11 AS build")
    assert texts[0].endswith("FROM is not an instruction here\nEOF")
    assert texts[1].startswith("FROM python:3.11-slim")


def test_makefile_rules_keep_recipes_defines_and_conditionals() -> None:
    texts = _fallback_texts("makefile", MAKEFILE, 1000)
    assert texts == [MAKEFILE.rstrip()]
    whole = config_top_regions(MAKEFILE, "makefile")
    assert check_units_whole(MAKEFILE, _fallback("makefile", MAKEFILE, 80).spans, whole, 80) == []


def test_generic_text_merges_blank_line_blocks_up_to_the_cap() -> None:
    assert _fallback_texts("javascript", TEXT_JS, 40) == [
        "function a() {\n  return 1;\n}",
        "function b() {\n  return 2;\n}",
    ]
    assert _fallback_texts("javascript", TEXT_JS, CAP) == [TEXT_JS.rstrip()]


@pytest.mark.parametrize("label", ["markdown", "yaml", "toml", "dockerfile", "makefile", "c"])
@pytest.mark.parametrize("extra", [0, 1])
def test_fallbacks_exact_cap_fits_and_one_more_character_does_not(label: str, extra: int) -> None:
    text = "alpha beta\ngamma delta\n"
    cap = len(text.rstrip()) - extra

    result = _fallback(label, text, cap)

    assert check_shared(text, result, cap, lambda: _fallback(label, text, cap)) == []
    assert len(result.spans) == (1 if extra == 0 else 2)


@pytest.mark.parametrize("label", ["markdown", "yaml", "toml", "dockerfile", "makefile", "rust"])
@pytest.mark.parametrize("text", ["", " \n\t\n"])
def test_fallbacks_give_no_spans_for_blank_text(label: str, text: str) -> None:
    assert _fallback(label, text, CAP).spans == ()


@pytest.mark.parametrize(
    "text",
    [
        "x" * 100_000 + "\n",
        ("#" * 5000 + "\n") * 20,
        ("`" * 5000 + "\n") * 20,
        ("- " * 10_000 + "\n") * 5,
        ("[" * 5000 + "\n") * 20,
    ],
    ids=["long_line", "hashes", "backticks", "dashes", "brackets"],
)
@pytest.mark.parametrize("label", ["markdown", "yaml", "toml", "dockerfile", "makefile", "c"])
def test_pathological_inputs_finish_quickly(label: str, text: str) -> None:
    started = time.perf_counter()
    result = _fallback(label, text, CAP)

    assert time.perf_counter() - started < 2.0
    assert check_cap(text, result.spans, CAP) == []
    assert check_coverage(text, result.spans) == []


# --- Step 6: the complete chunker ---------------------------------------------------------------

RULES_FILE = REPO_ROOT / "config" / "ingest.ignore"


def _repo_documents() -> list[Document]:
    """Interleaved languages, in the order the loader would never produce (on purpose)."""
    return [
        _document(TWO_FUNCTIONS, "src/a.py"),
        _document(MD_MIXED, "README.md", "markdown"),
        _document(YAML_CI, "ci.yml", "yaml"),
        _document(TEXT_JS, "web/app.js", "javascript"),
        _document(BIG_CLASS, "src/service.py"),
    ]


def test_all_strategies_run_and_output_keeps_input_order_then_position() -> None:
    result = chunk_documents_with_report(_repo_documents())
    paths = [n.metadata["file_path"] for n in result.nodes]

    assert list(dict.fromkeys(paths)) == [
        "src/a.py",
        "README.md",
        "ci.yml",
        "web/app.js",
        "src/service.py",
    ]
    assert set(result.report.by_strategy) == {"syntactic", "markdown", "config", "text"}
    for path in set(paths):
        starts = [n.start_char_idx or 0 for n in result.nodes if n.metadata["file_path"] == path]
        assert starts == sorted(starts)


def test_the_report_is_consistent_with_the_nodes() -> None:
    result = chunk_documents_with_report(
        _repo_documents(), rules=ChunkingRules(max_chunk_chars=120)
    )
    report = result.report
    sizes = [len(n.text) for n in result.nodes]

    assert report.documents == 5
    assert report.nodes == len(result.nodes) == len(result.spans)
    assert sum(t.nodes for t in report.by_language.values()) == report.nodes
    assert sum(t.nodes for t in report.by_strategy.values()) == report.nodes
    assert sum(report.by_kind.values()) == report.nodes
    assert (
        report.forced == sum(report.forced_by_kind.values()) == sum(s.forced for s in result.spans)
    )
    assert dict(report.fallback_reasons) == {"no_parser": 3}
    assert (report.size.p50, report.size.p90, report.size.max) == (
        percentile(sizes, 50),
        percentile(sizes, 90),
        max(sizes),
    )
    assert report.max_embed_chars == max(
        len(n.get_content(metadata_mode=MetadataMode.EMBED)) for n in result.nodes
    )
    assert (
        report
        == chunk_documents_with_report(
            _repo_documents(), rules=ChunkingRules(max_chunk_chars=120)
        ).report
    )
    assert json.loads(json.dumps(report.to_dict()))["nodes"] == report.nodes
    assert report.render_text().startswith("Chunked: 5 documents -> ")


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        ([], 50, 0),
        ([7], 50, 7),
        ([1, 2], 50, 1),
        (list(range(1, 11)), 50, 5),
        (list(range(1, 11)), 90, 9),
    ],
)
def test_percentiles_use_the_nearest_rank_method(
    values: list[int], percent: int, expected: int
) -> None:
    assert percentile(values, percent) == expected


def test_no_documents_give_no_nodes_and_an_empty_report() -> None:
    result = chunk_documents_with_report([])

    assert result.nodes == ()
    assert (result.report.documents, result.report.nodes, result.report.size.max) == (0, 0, 0)


def test_a_whitespace_only_document_gives_no_nodes_and_is_counted() -> None:
    result = chunk_documents_with_report([_document("  \n\t\n", "blank.py")])

    assert result.nodes == ()
    assert result.report.empty_documents == ("blank.py",)


def _without(key: str) -> Document:
    document = _document("x = 1\n", "a.py")
    document.metadata.pop(key)
    return document


def _with(key: str, value: object) -> Document:
    document = _document("x = 1\n", "a.py")
    document.metadata[key] = value
    return document


@pytest.mark.parametrize(
    "documents",
    [
        [_without("content_hash")],
        [_with("size_bytes", True)],  # a bool is not an int here
        [_with("num_lines", "1")],
        [_with("file_path", "b.py")],  # doc_id != file_path
        [_document("x = 1\n", "a.py"), _document("y = 2\n", "a.py")],  # duplicate doc_id
        [Document(text="x = 1\n", metadata=_document("x = 1\n", "a.py").metadata)],  # no id_
    ],
    ids=["missing_key", "bool_for_int", "str_for_int", "doc_id_mismatch", "duplicate", "no_id"],
)
def test_invalid_documents_are_a_typed_error(documents: list[Document]) -> None:
    with pytest.raises(InvalidDocumentError):
        chunk_documents(documents)


def test_chunk_documents_is_a_thin_wrapper() -> None:
    documents = _repo_documents()
    nodes = chunk_documents(documents)

    assert isinstance(nodes, list)
    assert [n.node_id for n in nodes] == [
        n.node_id for n in chunk_documents_with_report(documents).nodes
    ]


def test_a_file_whose_strategy_fails_degrades_to_generic_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing(text: str, *, max_chars: int) -> Segmentation:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(chunker, "segmenter_for", lambda label: (Strategy.SYNTACTIC, failing))

    with caplog.at_level(logging.WARNING, logger="src.ingestion.chunker"):
        result = chunk_documents_with_report([_document(TWO_FUNCTIONS, "src/a.py")])

    assert result.report.degraded_files == ("src/a.py",)
    assert dict(result.report.fallback_reasons) == {"segmentation_failed": 1}
    assert dict(result.report.by_strategy) == {"text": NodeTally(1, 1)}
    (record,) = [r for r in caplog.records if r.getMessage() == "chunking.degraded"]
    assert record.levelno == logging.WARNING
    assert "src/a.py" not in json.dumps({k: str(v) for k, v in record.__dict__.items()})


def test_an_unavailable_parser_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(text: str, *, max_chars: int) -> Segmentation:
        raise ParserUnavailableError("grammar missing")

    monkeypatch.setattr(chunker, "segmenter_for", lambda label: (Strategy.SYNTACTIC, unavailable))

    with pytest.raises(ParserUnavailableError):
        chunk_documents([_document(TWO_FUNCTIONS, "src/a.py")])


def test_the_run_logs_counts_never_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        chunk_documents(_repo_documents())

    (done,) = [r for r in caplog.records if r.getMessage() == "chunking.done"]
    assert done.levelno == logging.INFO
    assert done.__dict__["nodes"] > 0
    rendered = json.dumps([{k: str(v) for k, v in r.__dict__.items()} for r in caplog.records])
    for secret in ("def alpha", "Service", "make lint", "function a", "Title"):
        assert secret not in rendered


def test_no_module_in_src_imports_the_chunker() -> None:
    importers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src").rglob("*.py")
        if path.name != "chunker.py"
        and (
            "ingestion.chunker" in path.read_text(encoding="utf-8")
            or "import chunker" in path.read_text(encoding="utf-8")
        )
    ]
    assert importers == []


# --- Step 6: CLI ----------------------------------------------------------------------------


def _write_repo(root: Path) -> Path:
    for rel, content in {
        "src/a.py": TWO_FUNCTIONS,
        "README.md": MD_MIXED,
        "ci.yml": YAML_CI,
        "web/app.js": TEXT_JS,
        "src/escape.py": "x = '\\x1b[31m'  # \x1b[31mred\x1b[0m\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _cli(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = chunker.main([*argv, "--rules-file", str(RULES_FILE)], stdout=out)
    return code, out.getvalue()


def test_cli_text_report(tmp_path: Path) -> None:
    code, output = _cli(["--repo", str(_write_repo(tmp_path))])

    assert code == 0
    assert output.startswith("Chunked: 5 documents -> ")
    assert output.endswith("RESULT: OK (exit 0 OK)\n")


def test_cli_json_is_the_report(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)

    code, output = _cli(["--repo", str(repo), "--json"])

    documents = load_repository_with_report(repo, rules_file=RULES_FILE).documents
    assert code == 0
    assert json.loads(output) == chunk_documents_with_report(documents).report.to_dict()


def test_cli_list_has_one_line_per_node(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path)

    code, output = _cli(["--repo", str(repo), "--list"])

    documents = load_repository_with_report(repo, rules_file=RULES_FILE).documents
    lines = output.splitlines()
    assert code == 0
    assert len(lines) == len(chunk_documents(documents))
    assert all(len(line.split("\t")) == 6 for line in lines)


def test_cli_show_escapes_control_characters(tmp_path: Path) -> None:
    code, output = _cli(["--repo", str(_write_repo(tmp_path)), "--show", "src/escape.py"])

    assert code == 0
    assert "\x1b" not in output
    assert "\\x1b[31mred" in output


def test_cli_show_rejects_a_path_that_is_not_a_loaded_document(tmp_path: Path) -> None:
    code, output = _cli(["--repo", str(_write_repo(tmp_path)), "--show", "../etc/passwd"])

    assert code == chunker.USAGE_ERROR == 2
    assert "is not a loaded document" in output


def test_cli_exit_codes_for_missing_root_and_empty_repository(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("{}\n", encoding="utf-8")

    assert _cli(["--repo", str(tmp_path / "missing")])[0] == ExitCode.INGESTION
    assert _cli(["--repo", str(tmp_path)])[0] == ExitCode.NO_DOCUMENTS


def test_cli_maps_invalid_configuration_to_exit_10(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPO_PATH", "relative/path")
    monkeypatch.delitem(sys.modules, "src.config")

    out = io.StringIO()
    code = chunker.main([], stdout=out)

    assert code == ExitCode.CONFIGURATION
    assert "REPO_PATH" in out.getvalue()


def test_cli_blocks_and_reports_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = chunker.chunk_documents_with_report

    def leaky(*args: object, **kwargs: object) -> object:
        socket.getaddrinfo("example.com", 443)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(chunker, "chunk_documents_with_report", leaky)

    assert _cli(["--repo", str(_write_repo(tmp_path))])[0] == ExitCode.EXTERNAL_NETWORK


_DETERMINISM_CHILD = textwrap.dedent(
    """
    import hashlib, json, sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_repository
    documents = load_repository(Path(sys.argv[2]), rules_file=Path(sys.argv[3]))
    print(json.dumps([
        [n.node_id, hashlib.sha256(n.text.encode()).hexdigest(), list(n.metadata.items()), n.hash]
        for n in chunk_documents(documents)
    ]))
    """
)


def test_nodes_are_identical_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path / "repo")
    outputs = []
    for seed in ("0", "4242"):
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "PYTHONHASHSEED": seed}
        result = subprocess.run(  # noqa: S603 (this interpreter, a constant script)
            [
                sys.executable,
                "-B",
                "-c",
                _DETERMINISM_CHILD,
                str(REPO_ROOT),
                str(repo),
                str(RULES_FILE),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])) > 5


def test_the_module_runs_as_a_script(tmp_path: Path) -> None:
    repo = _write_repo(tmp_path / "repo")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}

    result = subprocess.run(  # noqa: S603 (this interpreter, constant arguments)
        [
            sys.executable,
            "-m",
            "src.ingestion.chunker",
            "--repo",
            str(repo),
            "--json",
            "--rules-file",
            str(RULES_FILE),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["documents"] == 5


# --- Step 5: regressions found on the corpus (synthetic reproductions) ------------------------


def test_comments_that_would_push_a_fitting_config_block_over_the_cap_get_their_own_chunk() -> None:
    # Found on the corpus: a long comment run attached to the next key forced a split of a
    # key block that fits. Same rule as for Python definitions (Gate 1 policies).
    comments = "".join(f"# note {i}: " + "x" * 30 + "\n" for i in range(6))
    text = "first: 1\n\n" + comments + "second:\n  a: 1\n  b: 2\n"
    cap = 120
    result = _fallback("yaml", text, cap)
    whole, sized = _natural_units("yaml", text)

    assert check_units_whole(text, result.spans, whole, cap) == []
    assert check_forced_where_needed(text, result.spans, sized, cap) == []
    texts = _fallback_texts("yaml", text, cap)
    assert texts[-1] == "second:\n  a: 1\n  b: 2"  # the key block is whole and alone
    assert not any(s.forced for s in result.spans if text[s.start : s.end].startswith("second"))


def test_comments_between_config_blocks_belong_to_the_next_block_for_the_oracle_too() -> None:
    text = "FROM a\nRUN one\n\n# about the next one\n# second line\nRUN two\n"

    regions = config_top_regions(text, "dockerfile")

    assert [text[a:b] for a, b in regions] == [
        "FROM a",
        "RUN one",
        "# about the next one\n# second line\nRUN two",
    ]


# --- Step 7 (DoD #4): the doc's three tests, adapted and non-vacuous ---------------------------

STEP7_FUNCTIONS = (
    "import functools\n\n\n@functools.cache\ndef decorated(x):\n    return x * 2\n\n\n"
    "async def fetch(client):\n    return await client.get()\n\n\n"
    "def neighbour_over_the_cap():\n    total = 0\n"
    + "".join(f"    total += {i}  # step {i}\n" for i in range(12))
    + "    return total\n\n\ndef tail():\n    return 1\n"
)
STEP7_CAP = 200

Piece = tuple[int, int, str | None, bool]  # (start, end, symbol, forced)


def _function_problems(text: str, pieces: Sequence[Piece], cap: int) -> list[str]:
    """Why ``pieces`` break "a function stays whole"; [] when every fitting function is
    exactly one piece with its complete source and no piece mixes two functions."""
    regions = [r for r in definition_regions(text) or [] if r.is_function]
    problems = []
    for region in regions:
        touching = [p for p in pieces if p[0] < region.end and region.start < p[1]]
        if region.size <= cap:
            exact = [p for p in touching if (p[0], p[1]) == (region.start, region.end)]
            if len(exact) != 1 or len(touching) != 1:
                problems.append(f"{region.name}: not exactly one piece with its whole source")
            elif exact[0][2] != region.name:
                problems.append(f"{region.name}: wrong symbol {exact[0][2]!r}")
        elif len(touching) < 2 or not all(p[3] and p[2] == region.name for p in touching):
            problems.append(f"{region.name}: split without forced pieces named after it")
    for start, end, _, _ in pieces:
        if sum(1 for r in regions if start < r.end and r.start < end) > 1:
            problems.append(f"piece [{start}, {end}) mixes two functions")
    return problems


def _node_pieces(nodes: Sequence[BaseNode], spans: Sequence[Span]) -> list[Piece]:
    return [
        (s.start, s.end, n.metadata.get("symbol"), s.forced)
        for n, s in zip(nodes, spans, strict=True)
    ]


def test_each_function_is_one_node_with_its_whole_source() -> None:
    result = chunk_documents_with_report(
        [_document(STEP7_FUNCTIONS, "src/step7.py")], rules=ChunkingRules(STEP7_CAP)
    )
    by_symbol: dict[str, list[str]] = {}
    for node in result.nodes:
        by_symbol.setdefault(str(node.metadata.get("symbol")), []).append(node.text)

    assert (
        _function_problems(STEP7_FUNCTIONS, _node_pieces(result.nodes, result.spans), STEP7_CAP)
        == []
    )
    assert by_symbol["decorated"] == ["@functools.cache\ndef decorated(x):\n    return x * 2"]
    assert by_symbol["fetch"] == ["async def fetch(client):\n    return await client.get()"]
    assert by_symbol["tail"] == ["def tail():\n    return 1"]
    assert len(by_symbol["neighbour_over_the_cap"]) > 1  # the one that does not fit


def test_the_function_assertions_fail_for_a_whole_file_and_a_blind_chunker() -> None:
    whole_file = [(0, len(STEP7_FUNCTIONS.rstrip()), None, False)]
    blind = [(s.start, s.end, None, False) for s in naive_line_split(STEP7_FUNCTIONS, 4)]

    assert _function_problems(STEP7_FUNCTIONS, whole_file, STEP7_CAP) != []
    assert _function_problems(STEP7_FUNCTIONS, blind, STEP7_CAP) != []


def _symbols_of(documents: list[Document], cap: int = CAP) -> list[str | None]:
    nodes = chunk_documents(documents, rules=ChunkingRules(cap))
    return [n.metadata.get("symbol") for n in nodes]


def test_symbols_on_nodes_for_every_kind_of_unit() -> None:
    assert _symbols_of([_document(TWO_FUNCTIONS, "a.py")]) == ["alpha", "beta"]
    assert _symbols_of([_document(CLASS_FITS, "a.py")]) == ["Point"]  # a whole class
    assert "Service.start" in _symbols_of([_document(BIG_CLASS, "a.py")], 120)
    assert _symbols_of([_document(NESTED_CLASS, "a.py")], 130)[-1] == "Outer.Large.inner_method"
    assert _symbols_of([_document(MD_MIXED, "README.md", "markdown")], 200) == [
        None,  # the preamble has no heading
        "Title > Usage",
        "Title > Setext heading",
    ]


@pytest.mark.parametrize(
    ("path", "language", "text"),
    [("a.py", "python", MODULE_CODE), ("ci.yml", "yaml", YAML_CI), ("a.go", "go", TEXT_JS)],
)
def test_nodes_without_a_name_have_no_symbol_key(path: str, language: str, text: str) -> None:
    nodes = chunk_documents([_document(text, path, language)], rules=ChunkingRules(200))
    unnamed = [n for n in nodes if "symbol" not in n.metadata]

    assert unnamed  # module code, configuration blocks and generic text
    assert all(None not in n.metadata.values() for n in nodes)


def test_metadata_is_inherited_exactly_and_enriched_with_exact_lines() -> None:
    document = _document(STEP7_FUNCTIONS, "src/step7.py")

    nodes = chunk_documents([document], rules=ChunkingRules(STEP7_CAP))
    fetch = next(n for n in nodes if n.metadata.get("symbol") == "fetch")

    assert list(fetch.metadata.items()) == [
        ("file_path", "src/step7.py"),
        ("language", "python"),
        ("size_bytes", len(STEP7_FUNCTIONS.encode("utf-8"))),
        ("num_lines", STEP7_FUNCTIONS.count("\n")),
        ("content_hash", "0" * 64),
        ("directory", "src"),
        ("symbol", "fetch"),
        ("start_line", 9),
        ("end_line", 10),
    ]
    assert STEP7_FUNCTIONS.split("\n")[8] == "async def fetch(client):"  # line 9, 1-based
    assert STEP7_FUNCTIONS.split("\n")[9] == "    return await client.get()"


# --- Step 7 (P6): manifest and content-free validation, tested without the corpus ------------


def _synthetic_run() -> tuple[list[Document], ChunkResult]:
    documents = _repo_documents()
    return documents, chunk_documents_with_report(documents, rules=ChunkingRules(120))


def _synthetic_manifest() -> dict[str, object]:
    _, result = _synthetic_run()
    return build_manifest("a" * 40, "https://example.invalid/repo.git", result, ChunkingRules(120))


def test_the_document_digest_follows_its_written_definition() -> None:
    _, result = _synthetic_run()
    nodes = [n for n in result.nodes if n.metadata["file_path"] == "src/a.py"]
    lines = "\n".join(f"{n.node_id}:{n.hash}" for n in nodes)

    assert document_digest(nodes) == hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_the_synthetic_digest_is_stable() -> None:
    # Golden value: chunking, IDs or hashes changing would change it.
    digest = _synthetic_manifest()["digest"]

    assert isinstance(digest, dict)
    assert digest["corpus"] == GOLDEN_SYNTHETIC_DIGEST


def test_the_manifest_round_trips_through_its_canonical_text() -> None:
    manifest = _synthetic_manifest()
    text = format_manifest(manifest)

    assert parse_manifest(text.encode("utf-8")) == manifest
    assert text.endswith("}\n")
    assert "\r" not in text


def _broken(change: Callable[[dict[str, object]], object]) -> bytes:
    manifest = json.loads(format_manifest(_synthetic_manifest()))
    change(manifest)
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _row(manifest: dict[str, object]) -> dict[str, object]:
    rows = manifest["documents"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    return row


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"\xef\xbb\xbf{}\n", "without BOM"),
        (b"{}\r\n", "LF line endings"),
        (b"{}", "ending with a newline"),
        (b"{not json\n", "not valid UTF-8 JSON"),
        (b"{}\n", "top-level keys"),
        (_broken(lambda m: m.update(format="other/1")), "unknown format"),
        (_broken(lambda m: _row(m).update(nodes=-1)), "invalid counts"),
        (_broken(lambda m: _row(m).update(strategy="magic")), "invalid path or strategy"),
        (_broken(lambda m: _row(m).update(digest="xyz")), "invalid digest"),
        (_broken(lambda m: _row(m).pop("forced")), "exactly the keys"),
        (_broken(lambda m: m.update(documents=[])), "non-empty list"),
        (_broken(lambda m: _row(m).update(digest="0" * 64)), "corpus digest does not match"),
    ],
    ids=[
        "bom",
        "crlf",
        "no_newline",
        "json",
        "keys",
        "format",
        "counts",
        "strategy",
        "digest",
        "row_keys",
        "empty",
        "corpus_digest",
    ],
)
def test_the_manifest_reader_is_strict_and_actionable(data: bytes, match: str) -> None:
    with pytest.raises(ChunkManifestError, match=match) as raised:
        parse_manifest(data)

    assert "Regenerate it with" in str(raised.value)


def test_a_non_canonical_manifest_is_rejected() -> None:
    compact = json.dumps(_synthetic_manifest()) + "\n"

    with pytest.raises(ChunkManifestError, match="canonical form"):
        parse_manifest(compact.encode("utf-8"))


def test_manifest_comparison_names_every_difference_without_content() -> None:
    expected = _synthetic_manifest()
    actual = json.loads(format_manifest(expected))
    rows = actual["documents"]
    moved = rows.pop(0)
    rows.append({**moved, "path": "new/file.py"})
    rows[0]["forced"] += 1
    actual["versions"]["tree-sitter"] = "9.9.9"
    actual["report"]["nodes"] += 1

    problems = compare_manifests(expected, actual)

    assert "missing from the run: src/a.py" in problems
    assert "not in the manifest: new/file.py" in problems
    assert any(p.startswith("README.md: forced differs") for p in problems)
    assert any(p.startswith("versions differs") for p in problems)
    assert any(p.startswith("report.nodes differs") for p in problems)
    assert compare_manifests(expected, expected) == []


def test_manifest_comparison_detects_a_reordering() -> None:
    expected = _synthetic_manifest()
    actual = json.loads(format_manifest(expected))
    actual["documents"].reverse()

    assert compare_manifests(expected, actual) == ["document order differs"]


def test_the_validator_accepts_a_correct_run_and_counts_split_definitions() -> None:
    documents, result = _synthetic_run()

    assert validate(documents, result, 120) == []
    counts = split_definitions(documents, result, 120)
    assert counts.split_that_fit == 0
    assert counts.split == counts.over_cap


def _tampered(change: Callable[[TextNode], None]) -> list[str]:
    documents, result = _synthetic_run()
    node = result.nodes[0]
    change(node)
    return sorted({f.code for f in validate(documents, result, 120)})


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda n: setattr(n, "text", n.text + " "), "slice"),
        (lambda n: setattr(n, "start_char_idx", 1), "offsets"),
        (lambda n: n.metadata.update(start_line=99), "lines"),
        (lambda n: setattr(n, "metadata", dict(reversed(list(n.metadata.items())))), "keys"),
        (lambda n: n.metadata.update(size_bytes="12"), "type:size_bytes"),
        (lambda n: n.metadata.update(language="go"), "inherited"),
        (lambda n: n.excluded_embed_metadata_keys.remove("start_line"), "visible"),
        (lambda n: setattr(n, "id_", "not-the-uuid5"), "id"),
        (lambda n: n.relationships.clear(), "relationship"),
    ],
    ids=["text", "offsets", "lines", "key_order", "type", "inherited", "visible", "id", "rel"],
)
def test_the_validator_detects_each_kind_of_tampering(
    change: Callable[[TextNode], None], code: str
) -> None:
    assert code in _tampered(change)


def test_the_validator_detects_duplicates_and_degraded_files() -> None:
    documents, result = _synthetic_run()
    duplicated = dataclasses.replace(
        result,
        nodes=(*result.nodes, result.nodes[0].model_copy(deep=True)),
        spans=(*result.spans, result.spans[0]),
    )
    degraded = dataclasses.replace(
        result, report=dataclasses.replace(result.report, degraded_files=("src/a.py",))
    )

    assert {"duplicate_id", "duplicate_hash", "order"} <= {
        f.code for f in validate(documents, duplicated, 120)
    }
    assert {f.code for f in validate(documents, degraded, 120)} == {"degraded"}


def test_findings_never_carry_text_or_symbols() -> None:
    documents, result = _synthetic_run()
    result.nodes[0].text += " SENTINEL_TEXT_7f1"

    findings = validate(documents, result, 120)

    assert findings
    assert "SENTINEL" not in repr(findings)
    assert "alpha" not in repr(findings)  # the symbol of the tampered node


def test_the_review_sample_is_deterministic_and_has_no_duplicates() -> None:
    documents = [
        _document(HUGE_FUNCTION, "src/big.py"),
        _document(BIG_CLASS, "src/service.py"),
        _document(MODULE_CODE, "src/main.py"),
        _document(MD_BIG_SECTION, "docs/big.md", "markdown"),
        _document(MD_FENCES, "docs/fences.md", "markdown"),
        _document(MD_MIXED, "README.md", "markdown"),
        _document(DOCKERFILE, "Dockerfile", "dockerfile"),
        _document(MAKEFILE, "Makefile", "makefile"),
        _document(TOML_STRING, "pyproject.toml", "toml"),
        _document(YAML_CI, "ci.yml", "yaml"),
    ]
    result = chunk_documents_with_report(documents, rules=ChunkingRules(120))

    picks = review_sample(documents, result)

    assert picks == review_sample(documents, result)
    assert len({p.path for p in picks}) == len(picks)
    assert {p.path for p in picks} == {d.doc_id for d in documents}
    assert any(p.path == "README.md" for p in picks)  # chosen once, by its first criterion
    assert ReviewPick("src/service.py", "split class (class_header)") in picks


def test_the_tree_snapshot_sees_a_change(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    before = tree_snapshot(tmp_path)
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")

    assert tree_snapshot(tmp_path) != before


_SENTINEL_TEST = textwrap.dedent(
    """
    import json
    from pathlib import Path

    from chunk_validation import validate
    from llama_index.core import Document
    from src.ingestion.chunker import ChunkingRules, chunk_documents_with_report

    def test_failure_is_content_free():
        data = json.loads((Path(__file__).parent / "doc.json").read_text())
        document = Document(id_=data["path"], text=data["text"], metadata=data["metadata"])
        result = chunk_documents_with_report([document], rules=ChunkingRules(1500))
        result.nodes[0].text += data["text"]
        findings = validate([document], result, 1500)
        assert findings == []
    """
)


def test_a_failing_validation_prints_no_content(tmp_path: Path) -> None:
    sentinel = "SENTINEL_CONTENT_c41f"
    text = f"def {sentinel}_function():\n    return '{sentinel}'\n"
    document = _document(text, "src/s.py")
    (tmp_path / "doc.json").write_text(
        json.dumps({"path": "src/s.py", "text": text, "metadata": dict(document.metadata)}),
        encoding="utf-8",
    )
    (tmp_path / "test_sentinel.py").write_text(_SENTINEL_TEST, encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'tests'}",
    }

    result = subprocess.run(  # this interpreter, a constant test file
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_sentinel.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 1  # the forced failure happened
    assert "Finding(code=" in result.stdout  # the records are shown...
    assert sentinel not in result.stdout + result.stderr  # ...but never the content


# --- Step 7: defect found by the second-codebase smoke test (approved fix) ---------------------

BANNER_THEN_BIG_FUNCTION = (
    "# section banner "
    + "-" * 40
    + "\n\n\n@parametrize(\n"
    + "".join(f"    ({i}, {i}),\n" for i in range(12))
    + ")\ndef f(a, b):\n    return a + b\n"
)
BANNER_THEN_BIG_CLASS = (  # a banner that fits alone, not together with the class line
    "# section banner "
    + "-" * 120
    + "\n\n\nclass Config:\n"
    + "".join(f"    option_{i}: int = {i}\n" for i in range(12))
    + "\n    def method(self):\n        return 1\n"
)


@pytest.mark.parametrize(
    ("text", "kind"),
    [(BANNER_THEN_BIG_FUNCTION, "function"), (BANNER_THEN_BIG_CLASS, "class_header")],
    ids=["function", "class_header"],
)
def test_leading_comments_that_cannot_join_a_split_definition_get_their_own_span(
    text: str, kind: str
) -> None:
    cap = 150
    spans = _segment(text, cap).spans

    assert _violations(text, cap) == []
    assert (spans[0].kind.value, spans[0].forced, spans[0].name_path) == ("module_block", False, ())
    assert text[spans[0].start : spans[0].end].startswith("# section banner")
    assert spans[1].kind.value == kind
    assert text[spans[1].start : spans[1].end].startswith(("@parametrize(", "class Config:"))
