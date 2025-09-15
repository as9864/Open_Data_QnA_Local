from __future__ import annotations

"""Utility script to embed paper documents and store them in Postgres."""

import argparse
import json
from typing import Iterable, Dict, Any, List, Tuple

import psycopg
from pgvector.psycopg import register_vector

from agents import EmbedderAgent
from utilities import LOCAL_PG_CONN, PG_CONN_STRING


# ---------------------------------------------------------------------------
# PostgreSQL connection helpers (borrowed from store_embeddings.py style)
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
    return psycopg.connect(_PG_CONNSTR)


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def _load_documents(path: str) -> List[Dict[str, Any]]:
    """Load a list of document dictionaries from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of document objects")
    return data


def _prepare_records(docs: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, str, str, List[float]]]:
    """Embed documents and return records ready for DB insertion."""
    embedder = EmbedderAgent("local", "BAAI/bge-m3")
    records: List[Tuple[str, str, str, str, List[float]]] = []
    for doc in docs:
        title = doc.get("title")
        abstract = doc.get("abstract")
        content = doc.get("content")
        metadata = json.dumps(doc.get("metadata") or {})
        text = content or " ".join(filter(None, [title, abstract]))
        emb = embedder.create(text)
        records.append((title, abstract, content, metadata, emb))
    return records


def store_papers(records: List[Tuple[str, str, str, str, List[float]]]) -> int:
    """Insert embedded paper records into the `papers_embeddings` table."""
    if not records:
        return 0

    dim = len(records[0][-1])
    with _pg_connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS papers_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({dim})
                );
                """
            )
            cur.executemany(
                """
                INSERT INTO papers_embeddings (title, abstract, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s);
                """,
                records,
            )
        conn.commit()
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Store paper embeddings")
    parser.add_argument("--input", required=True, help="Path to JSON file with papers")
    args = parser.parse_args(argv)

    docs = _load_documents(args.input)
    records = _prepare_records(docs)
    n = store_papers(records)
    print(f"Inserted {n} papers into papers_embeddings")


if __name__ == "__main__":
    main()
