"""Create the papers_embeddings table and its HNSW index."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
from utilities import PG_VECTOR_CONN


# Normalize various Postgres URI schemes to libpq format

def _normalize_pg_url(url: str) -> str:
    return (
        (url or "")
        .strip()
        .replace("postgresql+psycopg2://", "postgresql://")
        .replace("postgresql+psycopg://", "postgresql://")
        .replace("postgres+psycopg2://", "postgresql://")
        .replace("postgres+psycopg://", "postgresql://")
        .replace("postgres://", "postgresql://")
    )


_CONNSTR = _normalize_pg_url(PG_VECTOR_CONN or "")
if not _CONNSTR:
    raise RuntimeError("PG_VECTOR_CONN (또는 LOCAL_PG_CONN) must be set in config.ini")

DDL_TABLE = """
CREATE TABLE IF NOT EXISTS papers_embeddings (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  abstract TEXT,
  content TEXT NOT NULL,
  metadata JSONB,
  embedding vector(1024)
);
"""

IDX = """
CREATE INDEX IF NOT EXISTS idx_papers_embedding
  ON papers_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=64);
"""


def main() -> None:
    with psycopg.connect(_CONNSTR) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(DDL_TABLE)
            cur.execute(IDX)
        conn.commit()


if __name__ == "__main__":
    main()
