"""Phase 2, Steps 1-2: coherence invariants, the parser registry and the segmenter.

Synthetic texts only. Every segmentation case is checked against all invariants I1-I8
(tests/chunking_invariants.py) and against the structure the agreed policies imply
(docs/decisions.md section 10). Small caps keep the over-the-cap cases readable.
"""

import ast
import builtins
import dataclasses
import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
from chunking_invariants import (
    Violation,
    check_all,
    check_bounds,
    check_cap,
    check_coverage,
    check_definitions_whole,
    check_deterministic,
    check_forced,
    check_line_boundaries,
    check_order,
    definition_regions,
    naive_char_split,
    naive_line_split,
)

from src.errors import ChunkingError, InvalidChunkingRulesError, ParserUnavailableError
from src.ingestion import parser_registry, segmenter
from src.ingestion.chunker import DEFAULT_MAX_CHUNK_CHARS, ChunkingRules
from src.ingestion.parser_registry import PYTHON, GrammarSpec, grammar_for, load_language
from src.ingestion.rules import IngestionRules
from src.ingestion.segmenter import segment
from src.ingestion.spans import ParseStatus, Segmentation, Span, SpanKind

REPO_ROOT = Path(__file__).resolve().parents[1]
CAP = DEFAULT_MAX_CHUNK_CHARS

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
