"""Service for handling OMOP concept chat requests."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Sequence

try:  # psycopg is optional during unit tests
    import psycopg
    from psycopg import errors
    from psycopg.rows import dict_row
    from pgvector.psycopg import register_vector
    _PSYCOPG_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    psycopg = None  # type: ignore
    errors = None  # type: ignore
    dict_row = None  # type: ignore
    register_vector = None  # type: ignore
    _PSYCOPG_IMPORT_ERROR = exc

from agents.EmbedderAgent import EmbedderAgent
from agents.ResponseAgent_Local import ResponseAgent as ResponseAgentLocal
from utilities import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    LOCAL_PG_CONN,
    PG_CONN_STRING,
    PROMPTS,
    config,
    format_prompt,
)


def _normalize_pg_url(url: str) -> str:
    return (
        (url or "").strip()
        .replace("postgresql+psycopg2://", "postgresql://")
        .replace("postgres+psycopg2://", "postgresql://")
        .replace("postgres://", "postgresql://")
    )


_PG_CONNSTR = _normalize_pg_url(LOCAL_PG_CONN or PG_CONN_STRING or "")
if not _PG_CONNSTR:
    raise RuntimeError("LOCAL_PG_CONN or PG_CONN_STRING must be defined in config.ini")


def _pg_connect() -> "psycopg.Connection":
    if psycopg is None:  # pragma: no cover - exercised in tests via monkeypatch
        raise RuntimeError("psycopg is required for OMOP concept chat") from _PSYCOPG_IMPORT_ERROR
    return psycopg.connect(_PG_CONNSTR)


def _resolve_embedding_model() -> str:
    configured = config.get("OMOP", "CONCEPT_EMBEDDING_MODEL", fallback="").strip()
    if configured:
        return configured
    if EMBEDDING_MODEL_PATH:
        return EMBEDDING_MODEL_PATH
    if EMBEDDING_MODEL:
        return EMBEDDING_MODEL
    return "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _get_embedder() -> EmbedderAgent:
    return EmbedderAgent("local", _resolve_embedding_model())


@lru_cache(maxsize=1)
def _get_agent() -> ResponseAgentLocal:
    model = config.get("OMOP", "CONCEPT_CHAT_MODEL", fallback="").strip()
    if model:
        return ResponseAgentLocal(model=model)
    return ResponseAgentLocal()


def _format_concept_context(concepts: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for concept in concepts:
        header = (
            f"- concept_id {concept['concept_id']} ({concept['concept_name']})"
            f" | vocabulary: {concept.get('vocabulary_id') or 'N/A'}"
            f" | domain: {concept.get('domain_id') or 'N/A'}"
            f" | class: {concept.get('concept_class_id') or 'N/A'}"
        )
        description = (concept.get("description") or "").strip()
        if description:
            lines.append(f"{header}\n  {description}")
        else:
            lines.append(header)
    if not lines:
        return "- 검색된 개념이 없습니다. 기본 지식을 활용하세요."
    return "\n".join(lines)


def _to_float_list(vector: Sequence[float] | Sequence[Any]) -> List[float]:
    return [float(x) for x in vector]


def _query_similar_concepts(query_embedding: Sequence[float], limit: int = 5) -> List[Dict[str, Any]]:
    if psycopg is None or register_vector is None or dict_row is None:
        return []

    if not query_embedding:
        return []

    embedding = _to_float_list(query_embedding)
    if not embedding:
        return []

    try:
        with _pg_connect() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    SELECT
                        concept_id,
                        concept_name,
                        domain_id,
                        vocabulary_id,
                        concept_class_id,
                        description,
                        1 - (embedding <=> %s) AS similarity
                    FROM omop_concept_embeddings
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                    """,
                    (embedding, embedding, limit),
                )
                rows = cur.fetchall()
    except Exception as exc:  # pragma: no cover - defensive fallback
        if psycopg is not None:
            if errors is not None and isinstance(exc, errors.UndefinedTable):
                return []
            if isinstance(exc, psycopg.OperationalError):
                return []
        raise

    concepts: List[Dict[str, Any]] = []
    for row in rows:
        similarity = row.get("similarity")
        if similarity is not None:
            similarity = float(similarity)
        concepts.append(
            {
                "concept_id": row["concept_id"],
                "concept_name": row["concept_name"],
                "domain_id": row.get("domain_id"),
                "vocabulary_id": row.get("vocabulary_id"),
                "concept_class_id": row.get("concept_class_id"),
                "description": row.get("description"),
                "similarity": similarity,
            }
        )
    return concepts


def _extract_embedding(raw_embedding: Any) -> List[float]:
    if isinstance(raw_embedding, list):
        if raw_embedding and isinstance(raw_embedding[0], (float, int)):
            return _to_float_list(raw_embedding)
        if raw_embedding and isinstance(raw_embedding[0], list):
            return _to_float_list(raw_embedding[0])
    raise TypeError("EmbedderAgent.create returned an unexpected embedding format")


async def run(question: str, *, top_k: int = 5) -> Dict[str, Any]:
    """Generate an OMOP concept chat response enriched with retrieved concepts."""

    embedder = _get_embedder()
    raw_embedding = embedder.create(question)
    query_embedding = _extract_embedding(raw_embedding)
    concepts = _query_similar_concepts(query_embedding, limit=top_k)

    prompt_template = PROMPTS["omop_concept_chat"]
    base_prompt = format_prompt(prompt_template, question=question)
    context_block = _format_concept_context(concepts)
    final_prompt = f"{base_prompt}\n\n[Retrieved OMOP Concepts]\n{context_block}"

    agent = _get_agent()
    answer = await asyncio.to_thread(agent.generate_llm_response, final_prompt)

    return {
        "answer": answer,
        "concepts": concepts,
        "prompt": final_prompt,
    }


