from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore

from agents import EmbedderAgent

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached question/answer pair."""

    question: str
    answer: str
    question_types: Optional[Sequence[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    is_dynamic: bool = False


@dataclass
class MatchResult:
    """Result returned when the cache finds a match."""

    question: str
    answer: str
    score: float
    entry: CacheEntry

    @property
    def metadata(self) -> Dict[str, Any]:  # pragma: no cover - convenience
        return self.entry.metadata


def _flatten_embedding(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        if raw and isinstance(raw[0], (float, int)):
            return [float(x) for x in raw]
        if raw and isinstance(raw[0], list):
            return [float(x) for x in raw[0]]
    raise TypeError("Unsupported embedding format returned by EmbedderAgent")


def _normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


class FaqMatchCache:
    """Key-value style cache with embedding-based matching."""

    def __init__(
        self,
        *,
        embedder_factory: Callable[[], EmbedderAgent],
        threshold: float = 0.85,
        history_threshold: float = 0.8,
        max_dynamic_entries: int = 200,
    ) -> None:
        if threshold <= 0 or threshold > 1:
            raise ValueError("threshold must be in (0, 1]")
        if history_threshold <= 0 or history_threshold > 1:
            raise ValueError("history_threshold must be in (0, 1]")

        self._embedder_factory = embedder_factory
        self._threshold = float(threshold)
        self._history_threshold = float(history_threshold)
        self._max_dynamic_entries = max(0, int(max_dynamic_entries))

        self._lock = threading.RLock()
        self._embedder: Optional[EmbedderAgent] = None
        self._static_entries: List[CacheEntry] = []
        self._dynamic_entries: Deque[CacheEntry] = deque()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_static_entries(self, entries: Iterable[Dict[str, Any]]) -> None:
        """Load FAQ-style entries from dictionaries."""

        processed: List[CacheEntry] = []
        for raw in entries:
            question = (raw.get("question") or "").strip()
            answer = (raw.get("answer") or "").strip()
            if not question or not answer:
                continue
            question_types = raw.get("question_types")
            if isinstance(question_types, int):
                question_types = [question_types]
            elif isinstance(question_types, (list, tuple)):
                question_types = [int(q) for q in question_types]
            else:
                question_types = None

            metadata = dict(raw.get("metadata") or {})
            metadata.setdefault("source", "faq")

            processed.append(
                CacheEntry(
                    question=question,
                    answer=answer,
                    question_types=question_types,
                    metadata=metadata,
                    is_dynamic=False,
                )
            )

        with self._lock:
            self._static_entries = processed
            for entry in self._static_entries:
                entry.embedding = None  # lazy compute

        if processed:
            logger.info("Loaded %d static FAQ cache entries", len(processed))

    def load_static_file(self, path: str) -> None:
        """Load static entries from a JSON or YAML file."""

        if not path:
            return
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            logger.warning("FAQ cache file not found: %%s", resolved)
            return

        try:
            if resolved.lower().endswith(('.yml', '.yaml')):
                if yaml is None:
                    raise RuntimeError("pyyaml is required to load YAML FAQ cache files")
                with open(resolved, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or []
            else:
                with open(resolved, "r", encoding="utf-8") as f:
                    data = json.load(f) or []
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to load FAQ cache entries from %s", resolved)
            return

        if not isinstance(data, list):
            logger.error("FAQ cache file must contain a list of entries: %s", resolved)
            return

        self.load_static_entries(data)

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------
    def clear_dynamic(self) -> None:
        with self._lock:
            self._dynamic_entries.clear()

    def remember(
        self,
        question: Optional[str],
        answer: Optional[str],
        *,
        question_type: Optional[int] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return

        entry = CacheEntry(
            question=question,
            answer=answer,
            question_types=[question_type] if question_type else None,
            metadata={
                "chat_id": chat_id,
                "question_type": question_type,
                "source": "history",
            },
            is_dynamic=True,
        )

        with self._lock:
            self._dynamic_entries.append(entry)
            if self._max_dynamic_entries and len(self._dynamic_entries) > self._max_dynamic_entries:
                self._dynamic_entries.popleft()

    # ------------------------------------------------------------------
    def match_question(
        self,
        question: Optional[str],
        question_type: Optional[int] = None,
        chat_id: Optional[str] = None,
    ) -> Optional[MatchResult]:
        question = (question or "").strip()
        if not question:
            return None

        query_embedding = self._encode(question)
        if not query_embedding:
            return None

        best: Optional[MatchResult] = None
        now = time.time()

        for entry in self._iter_entries():
            if not self._supports_question_type(entry, question_type):
                continue
            entry_embedding = self._ensure_embedding(entry)
            if not entry_embedding:
                continue

            score = _cosine_similarity(query_embedding, entry_embedding)
            effective_threshold = self._threshold
            if chat_id and entry.metadata.get("chat_id") == chat_id:
                effective_threshold = min(self._history_threshold, self._threshold)

            if score < effective_threshold:
                continue

            if best is None or score > best.score:
                entry.last_used = now
                best = MatchResult(
                    question=entry.question,
                    answer=entry.answer,
                    score=score,
                    entry=entry,
                )

        if best:
            source = best.entry.metadata.get("source")
            logger.info(
                "FAQ cache hit (source=%s, score=%.3f, question_type=%s)",
                source,
                best.score,
                question_type,
            )
        return best

    # ------------------------------------------------------------------
    def _iter_entries(self) -> Iterable[CacheEntry]:
        with self._lock:
            return list(self._static_entries) + list(self._dynamic_entries)

    def _supports_question_type(self, entry: CacheEntry, question_type: Optional[int]) -> bool:
        if not entry.question_types:
            return True
        if question_type is None:
            return False
        return int(question_type) in {int(q) for q in entry.question_types}

    def _encode(self, text: str) -> List[float]:
        embedder = self._get_embedder()
        if embedder is None:
            return []
        try:
            embedding = embedder.create(text)
        except Exception:  # pragma: no cover - log and fallback
            logger.exception("Failed to create embedding for FAQ cache query")
            return []
        try:
            vector = _flatten_embedding(embedding)
        except Exception:  # pragma: no cover - log and fallback
            logger.exception("Invalid embedding format from EmbedderAgent")
            return []
        return _normalize(vector)

    def _ensure_embedding(self, entry: CacheEntry) -> List[float]:
        if entry.embedding is not None:
            return entry.embedding
        embedder = self._get_embedder()
        if embedder is None:
            return []
        try:
            embedding = embedder.create(entry.question)
            vector = _flatten_embedding(embedding)
        except Exception:  # pragma: no cover - log and fallback
            logger.exception("Failed to create embedding for FAQ cache entry")
            entry.embedding = []
            return []
        entry.embedding = _normalize(vector)
        return entry.embedding

    def _get_embedder(self) -> Optional[EmbedderAgent]:
        embedder = self._embedder
        if embedder is not None:
            return embedder
        with self._lock:
            if self._embedder is not None:
                return self._embedder
            try:
                self._embedder = self._embedder_factory()
            except Exception:  # pragma: no cover - log and fallback
                logger.exception("Failed to initialise FAQ cache embedder")
                self._embedder = None
                return None
        return self._embedder


__all__ = [
    "FaqMatchCache",
    "MatchResult",
]
