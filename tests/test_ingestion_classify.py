"""Tests for Step 3 name classification: lockfiles, secret-like names, language.

Parametrized over the IngestionRules defaults, which tests/test_ingestion_rules.py pins to
the agreed lists; here the point is that classification applies every entry.
"""

import os
from pathlib import Path

import pytest

from src.ingestion.classify import NameClassification, classify_name
from src.ingestion.reasons import DiscardReason
from src.ingestion.rules import IngestionRules

RULES = IngestionRules()
EXTENSIONS = sorted(RULES.language_by_extension.items())
LOCKFILES = sorted(RULES.lockfile_names)


def _reason(path: str) -> DiscardReason | None:
    return classify_name(path, RULES).reason


def _language(path: str) -> str | None:
    return classify_name(path, RULES).language


# --- language -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("extension", "language"), EXTENSIONS)
def test_every_mapped_extension_gets_its_language(extension: str, language: str) -> None:
    assert classify_name(f"src/module{extension}", RULES) == NameClassification(language)


@pytest.mark.parametrize(("extension", "language"), EXTENSIONS)
def test_extensions_are_case_insensitive(extension: str, language: str) -> None:
    assert _language(f"MODULE{extension.upper()}") == language


def test_mixed_case_extension() -> None:
    assert _language("docs/Guide.Md") == "markdown"


@pytest.mark.parametrize(
    ("path", "language"),
    [("Dockerfile", "dockerfile"), ("docker/Dockerfile", "dockerfile"), ("Makefile", "makefile")],
)
def test_dockerfile_and_makefile_by_exact_name(path: str, language: str) -> None:
    assert _language(path) == language


@pytest.mark.parametrize("path", ["dockerfile", "makefile", "DOCKERFILE", "Dockerfile.dev"])
def test_file_names_are_matched_exactly_and_case_sensitively(path: str) -> None:
    # Decision Q3: exact names only; lowercase variants and suffixed names are not mapped.
    assert _reason(path) == DiscardReason.UNMAPPED_TYPE


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/a.test.py", "python"),  # only the last suffix counts
        ("dist.tar.gz", None),
        ("archive.py.bak", None),
    ],
)
def test_only_the_last_suffix_counts(path: str, expected: str | None) -> None:
    assert _language(path) == expected


@pytest.mark.parametrize("path", [".gitignore", ".dockerignore", ".python-version", "sub/.gitkeep"])
def test_dotfiles_have_no_extension(path: str) -> None:
    assert _reason(path) == DiscardReason.UNMAPPED_TYPE


@pytest.mark.parametrize(
    "path",
    [
        "data.json",
        "requirements.txt",
        "notebooks/01_exploration.ipynb",
        "other.lock",  # a .lock that is not one of the nine lockfile names
        "deploy/model/MLmodel",
        "docs/images/architecture.png",
        "docs/whitepaper.pdf",
        "docs/videos/intro.mp4",
        "model.pkl",
        "preprocessor.joblib",
        "LICENSE",
        "data/raw/creditcard.csv.dvc",
    ],
)
def test_unmapped_types_are_discarded(path: str) -> None:
    assert classify_name(path, RULES) == NameClassification(reason=DiscardReason.UNMAPPED_TYPE)


# --- lockfiles ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", LOCKFILES)
def test_every_lockfile_is_discarded(name: str) -> None:
    assert classify_name(f"sub/{name}", RULES) == NameClassification(reason=DiscardReason.LOCKFILE)


def test_lockfile_wins_over_a_mapped_extension() -> None:
    assert _reason("pnpm-lock.yaml") == DiscardReason.LOCKFILE  # .yaml is mapped


def test_lockfile_names_are_exact() -> None:
    assert _reason("UV.LOCK") == DiscardReason.UNMAPPED_TYPE


# --- secret-like names (decision D4) -------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".ENV",
        "docker/.env.production",
        ".env.local",
        "id_rsa",
        "id_rsa.pub",  # a public key, but the glob is deliberately broad
        "certs/server.pem",
        "private.key",
        "config/secrets.yaml",  # mapped extension: the secret check runs first
        "Credentials.py",  # false positive by design: a legitimate module would be skipped
        "aws_credentials",
        "credentials.example",  # template exceptions apply to the .env family only
    ],
)
def test_secret_like_names_are_discarded(path: str) -> None:
    assert classify_name(path, RULES) == NameClassification(reason=DiscardReason.SECRET_NAME)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("docker/.env.example", DiscardReason.UNMAPPED_TYPE),
        (".env.sample", DiscardReason.UNMAPPED_TYPE),
        (".env.TEMPLATE", DiscardReason.UNMAPPED_TYPE),
        ("docs/decisions/0025-key-test-scope.md", None),  # "key" in the name, not *.key
        ("keyboard.py", None),
        ("environment.yaml", None),
    ],
)
def test_names_that_are_not_secrets(path: str, expected: DiscardReason | None) -> None:
    assert _reason(path) == expected


# --- contract -----------------------------------------------------------------------------------


def test_classification_has_exactly_one_outcome() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        NameClassification()
    with pytest.raises(ValueError, match="exactly one"):
        NameClassification("python", DiscardReason.BINARY)


@pytest.mark.parametrize("path", ["", "dir/"])
def test_paths_without_a_file_name_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="file"):
        classify_name(path, RULES)


def test_classification_never_touches_the_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("classify_name touched the filesystem")

    for name in ("stat", "lstat", "open"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("stat", "exists", "is_file", "open"):
        monkeypatch.setattr(Path, name, forbidden)

    assert classify_name("src/app.py", RULES).language == "python"
    assert classify_name(".env", RULES).reason == DiscardReason.SECRET_NAME
