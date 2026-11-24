"""Drift guard: ollama/Modelfile and the Makefile must agree with src/config.py.

Neither make nor a Modelfile can import Python, so both repeat a few literals from
src/config.py (the single source of truth); these tests keep those copies honest.
"""

import re
from pathlib import Path

import pytest

from src import config
from src.config import load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELFILE = REPO_ROOT / "ollama" / "Modelfile"
MAKEFILE = REPO_ROOT / "Makefile"


def _modelfile_instructions() -> list[tuple[str, str]]:
    instructions: list[tuple[str, str]] = []
    for raw_line in MODELFILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, argument = line.partition(" ")
        instructions.append((keyword.upper(), " ".join(argument.split())))
    return instructions


def _make_variable(name: str) -> str:
    pattern = rf"^{re.escape(name)}\s*:?=\s*(\S+)\s*$"
    match = re.search(pattern, MAKEFILE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert match is not None, f"{name} is not defined in the Makefile"
    return match.group(1)


def test_modelfile_only_sets_the_base_model_and_num_ctx() -> None:
    assert _modelfile_instructions() == [
        ("FROM", config.CODE_BASE_MODEL),
        ("PARAMETER", f"num_ctx {config.DERIVED_MODEL_NUM_CTX}"),
    ]


def test_modelfile_num_ctx_is_the_default_llm_context_window() -> None:
    assert load_settings({}).llm_context_window == config.DERIVED_MODEL_NUM_CTX


def test_default_code_model_is_the_derived_model() -> None:
    assert load_settings({}).code_model == config.CODE_DERIVED_MODEL


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("PROVISION_CODE_BASE", config.CODE_BASE_MODEL),
        ("PROVISION_CODE_RAG", config.CODE_DERIVED_MODEL),
        ("PROVISION_EMBEDDING", config.DEFAULT_EMBEDDING_MODEL),
        ("PROVISION_MODELFILE", MODELFILE.relative_to(REPO_ROOT).as_posix()),
    ],
)
def test_makefile_provisioning_names_match_config(variable: str, expected: str) -> None:
    assert _make_variable(variable) == expected
