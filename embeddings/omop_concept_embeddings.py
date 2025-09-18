"""Utilities for ingesting OMOP concept embeddings into pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, List, Dict, Any, Sequence

import psycopg
from pgvector.psycopg import register_vector

try:  # Optional dependency; EmbedderAgent may yield numpy arrays
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    np = None  # type: ignore

from agents import EmbedderAgent
from dbconnectors import pgconnector
from utilities import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    LOCAL_PG_CONN,
    PG_CONN_STRING,
    config,
)


# ---------------------------------------------------------------------------
# PostgreSQL connection helpers (mirrors embeddings.store_embeddings)
# ---------------------------------------------------------------------------

def _normalize_pg_url(url: str) -> str:
    """Normalise SQLAlchemy-style URLs to libpq format."""

    return (
        (url or "").strip()
        .replace("postgresql+psycopg2://", "postgresql://")
        .replace("postgres+psycopg2://", "postgresql://")
        .replace("postgres://", "postgresql://")
    )


_PG_CONNSTR = _normalize_pg_url(LOCAL_PG_CONN or PG_CONN_STRING or "")
if not _PG_CONNSTR:
    raise RuntimeError("LOCAL_PG_CONN or PG_CONN_STRING must be defined in config.ini")


def _pg_connect() -> psycopg.Connection:
    """Return a psycopg connection using the configured connection string."""

    return psycopg.connect(_PG_CONNSTR)


# ---------------------------------------------------------------------------
# Data modelling helpers
# ---------------------------------------------------------------------------


def _to_list(vector: Sequence[float] | Any) -> List[float]:
    """Coerce embedding vectors into JSON-serialisable lists."""

    if isinstance(vector, list):
        return [float(x) for x in vector]
    if isinstance(vector, tuple):
        return [float(x) for x in vector]
    if np is not None and isinstance(vector, np.ndarray):  # pragma: no branch
        return vector.astype(float).tolist()
    raise TypeError(f"Unsupported embedding vector type: {type(vector)!r}")


@dataclass
class ConceptPayload:
    concept_id: int
    concept_name: str
    domain_id: str | None
    vocabulary_id: str | None
    concept_class_id: str | None
    description: str
    embedding: List[float]

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.concept_id,
            self.concept_name,
            self.domain_id,
            self.vocabulary_id,
            self.concept_class_id,
            self.description,
            self.embedding,
        )


def _resolve_embedding_model() -> str:
    """Pick an embedding model using config fallbacks."""

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
    """Return a cached local embedder instance."""

    return EmbedderAgent("local", _resolve_embedding_model())


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

_CONCEPT_QUERY = """
WITH base AS (
    SELECT
        c.concept_id,
        c.concept_name,
        c.domain_id,
        c.vocabulary_id,
        c.concept_class_id,
        COALESCE(string_agg(DISTINCT cs.concept_synonym_name, '; '), '') AS synonyms,
        COALESCE(
            string_agg(
                DISTINCT CONCAT(r.relationship_name, ': ', c2.concept_name),
                '; '
            ),
            ''
        ) AS relationships
    FROM concept c
    LEFT JOIN concept_synonym cs
        ON cs.concept_id = c.concept_id
    LEFT JOIN concept_relationship cr
        ON cr.concept_id_1 = c.concept_id
        AND cr.invalid_reason IS NULL
    LEFT JOIN relationship r
        ON r.relationship_id = cr.relationship_id
    LEFT JOIN concept c2
        ON c2.concept_id = cr.concept_id_2
    WHERE c.invalid_reason IS NULL
    GROUP BY c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id, c.concept_class_id
)
SELECT * FROM base ORDER BY concept_id
"""


def _fetch_concepts(limit: int | None = None):
    """Retrieve the raw concept metadata as a pandas DataFrame."""

    query = _CONCEPT_QUERY
    if limit:
        query = f"{query}\nLIMIT {int(limit)}"
    return pgconnector.retrieve_df(query)


def _build_description(record: Dict[str, Any]) -> str:
    """Create a compact natural-language payload for a concept."""

    synonyms = (record.get("synonyms") or "").strip()
    relationships = (record.get("relationships") or "").strip()

    parts = [
        f"concept_id {record['concept_id']} — {record['concept_name']}",
        f"Domain: {record.get('domain_id') or 'N/A'}",
        f"Vocabulary: {record.get('vocabulary_id') or 'N/A'}",
        f"Class: {record.get('concept_class_id') or 'N/A'}",
    ]
    if synonyms:
        parts.append(f"Synonyms: {synonyms}")
    if relationships:
        parts.append(f"Relationships: {relationships}")
    return "\n".join(parts)


def _chunk_iter(iterable: Iterable[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    """Yield chunks from *iterable* with the given *size*."""

    bucket: List[Dict[str, Any]] = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


# ---------------------------------------------------------------------------
# Schema preparation and persistence
# ---------------------------------------------------------------------------

DDL_TABLE_TEMPLATE = """
CREATE TABLE IF NOT EXISTS omop_concept_embeddings (
  concept_id      BIGINT PRIMARY KEY,
  concept_name    TEXT NOT NULL,
  domain_id       TEXT,
  vocabulary_id   TEXT,
  concept_class_id TEXT,
  description     TEXT NOT NULL,
  embedding       vector({dim}) NOT NULL
);
"""

IDX_TABLE = """
CREATE INDEX IF NOT EXISTS idx_omop_concept_embeddings_cos
  ON omop_concept_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=64);
"""

SQL_UPSERT = """
INSERT INTO omop_concept_embeddings (
    concept_id,
    concept_name,
    domain_id,
    vocabulary_id,
    concept_class_id,
    description,
    embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (concept_id) DO UPDATE SET
    concept_name = EXCLUDED.concept_name,
    domain_id = EXCLUDED.domain_id,
    vocabulary_id = EXCLUDED.vocabulary_id,
    concept_class_id = EXCLUDED.concept_class_id,
    description = EXCLUDED.description,
    embedding = EXCLUDED.embedding;
"""


def _prepare_schema(dim: int) -> None:
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive")

    ddl = DDL_TABLE_TEMPLATE.format(dim=dim)
    with _pg_connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(ddl)
            cur.execute(IDX_TABLE)
        conn.commit()


def _persist_payloads(payloads: Iterable[ConceptPayload]) -> int:
    inserted = 0
    with _pg_connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            for payload in payloads:
                cur.execute(SQL_UPSERT, payload.to_row())
                inserted += 1
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def ingest_omop_concepts(limit: int | None = None, batch_size: int = 100) -> int:
    """Fetch OMOP concept metadata, embed it, and persist to pgvector."""

    df = _fetch_concepts(limit)
    if df is None or df.empty:
        return 0

    records = df.fillna("").to_dict(orient="records")
    embedder = _get_embedder()

    all_payloads: List[ConceptPayload] = []
    embedding_dim: int | None = None

    for chunk in _chunk_iter(records, max(1, batch_size)):
        descriptions = [_build_description(rec) for rec in chunk]
        embeddings = embedder.create(descriptions)
        if not isinstance(embeddings, list):
            raise TypeError("EmbedderAgent.create must return a list for batch inputs")
        if len(embeddings) != len(chunk):
            raise ValueError("Embedding count mismatch for OMOP concept ingestion")

        for rec, emb in zip(chunk, embeddings):
            emb_list = _to_list(emb)
            embedding_dim = embedding_dim or len(emb_list)
            all_payloads.append(
                ConceptPayload(
                    concept_id=int(rec["concept_id"]),
                    concept_name=str(rec["concept_name"]),
                    domain_id=rec.get("domain_id") or None,
                    vocabulary_id=rec.get("vocabulary_id") or None,
                    concept_class_id=rec.get("concept_class_id") or None,
                    description=_build_description(rec),
                    embedding=emb_list,
                )
            )

    if embedding_dim is None:
        return 0

    _prepare_schema(embedding_dim)
    return _persist_payloads(all_payloads)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    count = ingest_omop_concepts()
    print(f"Ingested {count} OMOP concept embeddings.")
