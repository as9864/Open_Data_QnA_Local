from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.faq_cache import FaqMatchCache


class _StubEmbedder:
    def __init__(self, vectors):
        self._vectors = vectors

    def create(self, text):
        vector = self._vectors.get(text)
        if vector is None:
            vector = self._vectors.get("__default__", [1.0])
        return list(vector)


def _make_cache(vectors, **kwargs) -> FaqMatchCache:
    return FaqMatchCache(embedder_factory=lambda: _StubEmbedder(vectors), **kwargs)


def test_faq_cache_static_match_returns_answer():
    vectors = {
        "faq question": [1.0, 0.0],
        "user question": [1.0, 0.0],
    }
    cache = _make_cache(vectors, threshold=0.9, history_threshold=0.8, max_dynamic_entries=10)
    cache.load_static_entries([
        {"question": "faq question", "answer": "cached answer", "question_types": [1, 2]},
    ])

    match = cache.match_question("user question", question_type=1)

    assert match is not None
    assert match.answer == "cached answer"


def test_faq_cache_respects_question_type():
    vectors = {
        "faq question": [1.0, 0.0],
        "user question": [1.0, 0.0],
    }
    cache = _make_cache(vectors, threshold=0.9, history_threshold=0.8, max_dynamic_entries=5)
    cache.load_static_entries([
        {"question": "faq question", "answer": "cached answer", "question_types": [2]},
    ])

    match = cache.match_question("user question", question_type=1)

    assert match is None


def test_faq_cache_history_threshold_allows_similar_match():
    base_vector = [1.0, 0.0]
    similar_vector = [0.9, 0.4358898943540673]
    vectors = {
        "original question": base_vector,
        "follow up question": similar_vector,
    }
    cache = _make_cache(vectors, threshold=0.95, history_threshold=0.85, max_dynamic_entries=5)

    cache.remember("original question", "first answer", question_type=1, chat_id="chat-1")

    # Same chat should satisfy relaxed threshold
    match_same_chat = cache.match_question("follow up question", question_type=1, chat_id="chat-1")
    assert match_same_chat is not None
    assert match_same_chat.answer == "first answer"

    # Different chat should not meet the stricter global threshold
    match_other_chat = cache.match_question("follow up question", question_type=1, chat_id="chat-2")
    assert match_other_chat is None
