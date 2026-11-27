"""Phase 3, Step 1, made executable with the real local model (needs Ollama).

"How do I authenticate?" has no word in common with a function that verifies an access
token, yet their vectors are close: the embedding captures intent, not keywords. The
snippets below were written for this test (synthetic, not from any corpus); the token
check never uses the word "authenticate". Cosine similarity is computed here, not by any
library, from the recipe's document and query prefixes.

English: the token snippet must rank first among the distractors. Spanish (decision P3:
nomic-embed-text v1.5 is English-centric): its rank is recorded, not asserted; it is
reported in docs/decisions.md section 13.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from src.indexing.embeddings import get_embedding_model

pytestmark = pytest.mark.integration

RecordProperty = Callable[[str, object], None]  # the `record_property` fixture

# A code snippet that checks a token, not a secret.
TOKEN_CHECK = '''def verify_access_token(token: str, signing_key: bytes) -> dict[str, object]:
    """Check the bearer token's signature and expiry; return its claims."""
    header, payload, signature = token.split(".")
    expected = hmac.new(signing_key, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, urlsafe_b64decode(signature)):
        raise PermissionError("bad signature")
    claims = json.loads(urlsafe_b64decode(payload))
    if claims["exp"] < time.time():
        raise PermissionError("token expired")
    return claims
'''  # noqa: S105
DISTRACTORS = (
    '''def read_rows(path: str) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries, one per row."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))
''',
    '''def retry(call, attempts: int = 5, base_delay: float = 0.5):
    """Call `call` until it succeeds, sleeping exponentially longer after each failure."""
    for attempt in range(attempts):
        try:
            return call()
        except ConnectionError:
            time.sleep(base_delay * 2 ** attempt)
    raise RuntimeError("gave up")
''',
    '''def paginate(items: list, page: int, per_page: int = 20) -> list:
    """Return one page of results, counting pages from 1."""
    start = (page - 1) * per_page
    return items[start : start + per_page]
''',
    '''def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from degrees Celsius to degrees Fahrenheit."""
    return celsius * 9 / 5 + 32
''',
    '''def resize_image(image, max_side: int = 1024):
    """Shrink an image so that its longest side is at most `max_side` pixels."""
    scale = max_side / max(image.width, image.height)
    return image.resize((int(image.width * scale), int(image.height * scale)))
''',
)
QUERIES = {"en": "how do I authenticate?", "es": "¿cómo me autentico?"}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _rank_of_token_check(language: str) -> tuple[int, list[float]]:
    embedder = get_embedding_model()
    embedder.identity()  # the local model is the verified nomic-embed-text v1.5
    documents = embedder.get_text_embedding_batch([TOKEN_CHECK, *DISTRACTORS])
    query = embedder.get_query_embedding(QUERIES[language])
    scores = [_cosine(query, document) for document in documents]
    rank = 1 + sum(score > scores[0] for score in scores[1:])
    return rank, scores


def test_the_token_check_ranks_first_for_an_english_question(
    record_property: RecordProperty,
) -> None:
    rank, scores = _rank_of_token_check("en")

    record_property("en_rank", rank)
    record_property("en_scores", [round(s, 4) for s in scores])
    assert rank == 1, f"token check ranked {rank}; cosines {[round(s, 4) for s in scores]}"


def test_the_rank_for_a_spanish_question_is_recorded(
    record_property: RecordProperty,
) -> None:
    rank, scores = _rank_of_token_check("es")

    record_property("es_rank", rank)
    record_property("es_scores", [round(s, 4) for s in scores])
    assert 1 <= rank <= 1 + len(DISTRACTORS)
