"""Phase 3, Step 6: golden-query loader, metrics, evaluator, baseline and review dump.

Hand-computed metric cases; a scripted retriever for the evaluator; the CLI over a fake
index. The golden texts here are synthetic; no query text may appear in errors, logs,
outputs or the baseline.
"""

from __future__ import annotations

import json
import logging
import stat
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pytest
from fake_embedding import FakeEmbedding

from src import config
from src.errors import GoldenQueriesError, InvalidQueryError
from src.evaluation import retrieval as ev
from src.evaluation.retrieval import (
    EvaluationResult,
    GoldenQuery,
    RankedChunk,
    evaluate,
    first_relevant_rank,
    golden_sha256,
    hit_at,
    mrr,
    parse_golden,
    summarize,
    write_review,
)
from src.indexing import loading
from src.indexing.build_index import build_index_with_report
from src.indexing.manifest import IndexManifest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "eval" / "retrieval_golden.example.json"
SECRET = "SENTINEL_GOLDEN_QUERY_TEXT_51aa"  # noqa: S105 (a sentinel, not a password)


def _golden(queries: list[dict[str, object]], version: int = 1) -> bytes:
    return json.dumps({"version": version, "queries": queries}).encode("utf-8")


def _query(qid: str = "q1", **changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": qid,
        "intent": "i1",
        "lang": "en",
        "kind": "concept",
        "query": f"{SECRET} how is it done?",
        "expected_files": ["src/a.py"],
    }
    base.update(changes)
    return base


# --- the strict loader -------------------------------------------------------------------------


def test_the_versioned_example_is_valid() -> None:
    queries = parse_golden(EXAMPLE.read_bytes())

    assert [q.id for q in queries] == ["ex01", "ex02", "ex03"]
    assert {q.lang for q in queries} == {"es", "en"}
    assert queries[2].expected_files == ("src/rows.py", "docs/rows.md")


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"not json", "not UTF-8 JSON"),
        (b"[]", 'keys "version", "queries"'),
        (_golden([_query()], version=2), "version must be 1"),
        (_golden([]), "non-empty list"),
        (_golden([_query(extra=1)]), r"query 'q1': unknown keys \['extra'\]"),
        (_golden([{k: v for k, v in _query().items() if k != "kind"}]), r"missing keys \['kind'\]"),
        (_golden([_query(), _query()]), "query 'q1': id is repeated"),
        (_golden([_query(id="bad id!")]), r"query #1: id must match"),
        (_golden([_query(lang="fr")]), "lang must be one of"),
        (_golden([_query(kind="other")]), "kind must be one of"),
        (_golden([_query(query="  ")]), "query must be a non-empty string"),
        (_golden([_query(expected_files=[])]), "expected_files must be a non-empty list"),
        (_golden([_query(expected_files=["a", "a"])]), "distinct"),
    ],
    ids=[
        "json",
        "shape",
        "version",
        "empty",
        "unknown",
        "missing",
        "dup",
        "id",
        "lang",
        "kind",
        "query",
        "no_files",
        "repeated_file",
    ],
)
def test_every_format_error_names_the_query_never_its_text(data: bytes, match: str) -> None:
    with pytest.raises(GoldenQueriesError, match=match) as raised:
        parse_golden(data)

    assert SECRET not in str(raised.value)


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("query", "REPLACE_WITH_SOME_SYMBOL"),
        ("query", "  <your question>  "),
        ("query", "TODO: write the question"),
        ("query", "FIXME"),
        ("intent", "<intent>"),
    ],
    ids=["replace_with", "angle", "todo", "fixme", "intent"],
)
def test_template_placeholders_are_refused_by_id_and_field(field: str, text: str) -> None:
    data = _golden([_query(**{field: text})])

    with pytest.raises(
        GoldenQueriesError, match=rf"query 'q1': {field} looks like a template placeholder"
    ) as raised:
        parse_golden(data)

    assert text.strip() not in str(raised.value)


def test_real_questions_that_mention_a_marker_are_accepted() -> None:
    data = _golden(
        [
            _query("q1", query="what does List<T> mean here?"),
            _query("q2", query="where are the TODO comments?"),
            _query("q3", query="what does replace_with_default do?"),
        ]
    )

    assert [q.id for q in parse_golden(data)] == ["q1", "q2", "q3"]


def test_expected_files_must_be_in_the_index() -> None:
    data = _golden([_query(expected_files=["src/a.py", "src/missing.py"])])

    with pytest.raises(GoldenQueriesError, match=r"query 'q1': expected_files\[1\] are not files"):
        parse_golden(data, known_files={"src/a.py"})
    assert parse_golden(data, known_files={"src/a.py", "src/missing.py"})


def test_the_queries_digest_is_the_sha256_of_the_file_bytes() -> None:
    data = EXAMPLE.read_bytes()

    assert golden_sha256(data) == __import__("hashlib").sha256(data).hexdigest()


# --- metrics, computed by hand -----------------------------------------------------------------


def test_first_relevant_rank_with_several_expected_files_and_repeats() -> None:
    ranked = ["x.py", "x.py", "b.md", "a.py", "a.py"]

    assert first_relevant_rank(ranked, {"a.py", "b.md"}) == 3
    assert first_relevant_rank(ranked, {"zzz.py"}) is None
    assert first_relevant_rank([], {"a.py"}) is None


def test_hit_at_k_and_mrr_by_hand() -> None:
    ranks: list[int | None] = [1, None, 3, 21]

    assert hit_at(ranks, 1) == 0.25
    assert hit_at(ranks, 3) == 0.5
    assert hit_at(ranks, 20) == 0.5  # rank 21 is beyond the depth
    assert mrr(ranks) == pytest.approx((1 + 1 / 3) / 4)
    assert summarize(ranks) == {
        "n": 4,
        "hit@1": 0.25,
        "hit@3": 0.5,
        "hit@5": 0.5,
        "hit@10": 0.5,
        "hit@20": 0.5,
        "mrr@20": 0.3333,
    }


def test_no_hits_and_no_queries() -> None:
    assert summarize([None, None])["mrr@20"] == 0.0
    assert hit_at([], 5) == 0.0
    assert mrr([]) == 0.0


# --- the evaluator, with a scripted retriever ----------------------------------------------------


class _Scripted:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings
        self.seen: list[str] = []

    @property
    def name(self) -> str:
        return "scripted"

    def rank(self, query: str, depth: int) -> Sequence[RankedChunk]:
        self.seen.append(query)
        files = self._rankings[query][:depth]
        return [RankedChunk(f, 1, 2, None, f"n{i}", 0.5, "code") for i, f in enumerate(files)]


def test_evaluate_ranks_breaks_down_and_records_by_id(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    queries = (
        GoldenQuery("q1", "i1", "en", "concept", f"{SECRET} en", ("src/a.py",)),
        GoldenQuery("q2", "i1", "es", "concept", f"{SECRET} es", ("src/a.py",)),
        GoldenQuery("q3", "i2", "en", "identifier", f"{SECRET} id", ("docs/b.md",)),
    )
    retriever = _Scripted(
        {
            f"{SECRET} en": ["src/a.py"],
            f"{SECRET} es": ["x.py", "y.py", "src/a.py"],
            f"{SECRET} id": ["x.py"] * 25,
        }
    )
    languages = {"src/a.py": "python", "docs/b.md": "markdown", "x.py": "python", "y.py": "python"}

    result = evaluate(queries, retriever, languages)

    assert result.ranks() == {"q1": 1, "q2": 3, "q3": None}
    assert result.metrics["hit@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert result.breakdown["lang"]["es"]["mrr@20"] == pytest.approx(1 / 3, abs=1e-4)
    assert result.breakdown["kind"]["identifier"]["hit@20"] == 0.0
    assert result.breakdown["target"] == {
        "code": summarize([1, 3]),
        "docs": summarize([None]),
    }
    assert SECRET not in caplog.text


def _result() -> EvaluationResult:
    queries = (GoldenQuery("q1", "i1", "en", "concept", f"{SECRET} q", ("src/a.py",)),)
    return evaluate(queries, _Scripted({f"{SECRET} q": ["src/a.py"]}), {"src/a.py": "python"})


def test_the_baseline_has_metrics_ranks_and_identity_but_no_query_text(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    record = ev.baseline_record(_result(), "f" * 64, manifest)

    text = json.dumps(record)
    assert SECRET not in text
    assert record["ranks"] == {"q1": 1}
    assert record["queries"] == {"sha256": "f" * 64, "count": 1}
    assert record["index"]["node_digest"] == manifest.node_digest  # type: ignore[index]
    assert record["depth"] == 20


def _manifest(tmp_path: Path) -> IndexManifest:
    repo = tmp_path / "repo-m"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "rules-m").write_text("", encoding="utf-8")
    from src.indexing.vector_store import ChromaIndexStore

    return build_index_with_report(
        repo,
        ChromaIndexStore(tmp_path / "vs-m", "code"),
        FakeEmbedding(),
        rules_file=tmp_path / "rules-m",
    ).manifest


# --- the review dump: owner only, outside both repositories, private ---------------------------


def test_the_review_dump_is_private_and_refused_inside_a_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    for inside in (REPO_ROOT / "eval" / "review", repo / "review"):
        with pytest.raises(InvalidQueryError, match="outside"):
            write_review(_result(), inside, [REPO_ROOT, repo])
    assert not (REPO_ROOT / "eval" / "review").exists()

    path = write_review(_result(), tmp_path / "review", [REPO_ROOT, repo])
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert SECRET in path.read_text(encoding="utf-8")  # the owner's dump does carry the text


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "auth.py").write_text(
        "def verify_token(t):\n    return t == 'ok'\n", encoding="utf-8"
    )
    (repo / "src" / "rows.py").write_text("def read_rows(p):\n    return []\n", encoding="utf-8")
    rules = tmp_path / "rules"
    rules.write_text("", encoding="utf-8")
    from src.indexing.vector_store import ChromaIndexStore

    build_index_with_report(
        repo, ChromaIndexStore(tmp_path / "vs", "code"), FakeEmbedding(), rules_file=rules
    )
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", tmp_path / "vs")
    monkeypatch.setattr(loading, "get_embedding_model", FakeEmbedding)
    golden = tmp_path / "golden.json"
    golden.write_bytes(
        _golden(
            [
                _query("q1", query=f"{SECRET} verify token", expected_files=["src/auth.py"]),
                _query(
                    "q2",
                    lang="es",
                    query=f"{SECRET} leer filas rows",
                    expected_files=["src/rows.py"],
                ),
            ]
        )
    )
    return repo, rules, golden


def _run(argv: list[str]) -> tuple[int, str]:
    out = StringIO()
    return ev.main(argv, stdout=out), out.getvalue()


def test_the_cli_reports_metrics_and_ranks_without_query_text(
    indexed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    repo, rules, golden = indexed
    args = ["--golden", str(golden), "--repo", str(repo), "--rules-file", str(rules)]

    code, output = _run(args)
    code_json, output_json = _run([*args, "--json"])

    assert (code, code_json) == (0, 0)
    assert "hit@1" in output
    assert "q1=" in output
    assert SECRET not in output + output_json
    assert json.loads(output_json)["queries"]["count"] == 2


def test_the_cli_writes_a_baseline_once_and_never_overwrites_it_silently(
    indexed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    repo, rules, golden = indexed
    baseline = tmp_path / "baseline.json"
    args = [
        "--golden",
        str(golden),
        "--repo",
        str(repo),
        "--rules-file",
        str(rules),
        "--write-baseline",
        str(baseline),
    ]

    assert _run(args)[0] == 0
    written = baseline.read_text(encoding="utf-8")
    refused, message = _run(args)

    assert refused == 2
    assert "owner decision" in message
    assert baseline.read_text(encoding="utf-8") == written
    assert SECRET not in written
    assert _run([*args, "--overwrite-baseline"])[0] == 0


def test_the_cli_refuses_bad_golden_files_and_review_dirs_inside_repositories(
    indexed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    repo, rules, golden = indexed
    bad = tmp_path / "bad.json"
    bad.write_bytes(_golden([_query("q9", expected_files=["src/missing.py"])]))
    base = ["--repo", str(repo), "--rules-file", str(rules)]

    code, output = _run(["--golden", str(bad), *base])
    assert code == 2
    assert "q9" in output
    assert SECRET not in output
    assert _run(["--golden", str(tmp_path / "absent.json"), *base])[0] == 2
    assert _run(["--golden", str(golden), *base, "--review-dir", str(repo / "dump")])[0] == 2
    assert not (repo / "dump").exists()
