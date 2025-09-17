from __future__ import annotations

"""Utility script to embed paper documents (JSON or PDF) and store them in Postgres."""

import argparse
import json
import os
import sys
import hashlib
from dataclasses import dataclass, asdict
from typing import Iterable, Dict, Any, List, Tuple, Optional

import psycopg
from pgvector.psycopg import register_vector

# --- optional PDF deps (graceful import) ---
_HAS_PYMUPDF = False
_HAS_PDFMINER = False
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    pass

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    _HAS_PDFMINER = True
except Exception:
    pass

from agents import EmbedderAgent
from utilities import LOCAL_PG_CONN, PG_CONN_STRING


# ---------------------------------------------------------------------------
# PostgreSQL connection helpers
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
# Models & helpers
# ---------------------------------------------------------------------------

@dataclass
class PaperDoc:
    title: Optional[str]
    abstract: Optional[str]
    content: Optional[str]
    metadata: Dict[str, Any]

    def to_db_tuple(self, embedding: List[float]) -> Tuple[str, str, str, str, List[float]]:
        return (
            self.title,
            self.abstract,
            self.content,
            json.dumps(self.metadata, ensure_ascii=False),
            embedding,
        )


def _file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extract_text_pymupdf(path: str) -> str:
    # PyMuPDF: 빠르고 레이아웃 무시 텍스트 추출이 깔끔
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)


def _extract_text_pdfminer(path: str) -> str:
    # pdfminer.six: 설치되어 있으면 레이아웃 보존형 추출 가능
    return pdfminer_extract_text(path)


def _extract_text_from_pdf(path: str) -> str:
    # 우선 PyMuPDF → 폴백으로 pdfminer.six
    if _HAS_PYMUPDF:
        try:
            return _extract_text_pymupdf(path)
        except Exception as e:
            # 폴백 시도
            if not _HAS_PDFMINER:
                raise RuntimeError(
                    f"PyMuPDF failed to extract text and pdfminer.six not available: {e}"
                )
    if _HAS_PDFMINER:
        return _extract_text_pdfminer(path)
    raise RuntimeError(
        "No PDF extractor available. Install PyMuPDF (`pip install pymupdf`) "
        "or pdfminer.six (`pip install pdfminer.six`)."
    )


def _pdf_title_and_meta(path: str) -> Tuple[str, Dict[str, Any]]:
    title = os.path.splitext(os.path.basename(path))[0]
    meta: Dict[str, Any] = {
        "source_path": os.path.abspath(path),
        "source_type": "pdf",
        "md5": _file_md5(path),
    }
    # 메타데이터 보강 (가능하면)
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(path)
            info = doc.metadata or {}
            if info.get("title"):
                title = info.get("title")
            meta.update({f"pdfmeta_{k}": v for k, v in info.items() if v})
        except Exception:
            pass
    return title, meta


def _is_json_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".json")


def _is_pdf_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".pdf")


def _gather_pdf_files(input_path: str, recursive: bool = True) -> List[str]:
    files: List[str] = []
    if os.path.isfile(input_path) and _is_pdf_file(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        if recursive:
            for root, _, fnames in os.walk(input_path):
                for fn in fnames:
                    if fn.lower().endswith(".pdf"):
                        files.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(input_path):
                fp = os.path.join(input_path, fn)
                if os.path.isfile(fp) and fn.lower().endswith(".pdf"):
                    files.append(fp)
    return files


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_documents_from_json(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of document objects")
    return data


def _load_documents_from_pdfs(paths: List[str]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for pdf_path in paths:
        title, meta = _pdf_title_and_meta(pdf_path)
        content = _extract_text_from_pdf(pdf_path)
        # 간단 추상(abstract) 추정: 첫 1200자 정도를 abstract로 할당(원한다면 규칙 변경)
        snippet = content.strip().splitlines()
        abstract = None
        if snippet:
            head = "\n".join(snippet[:30])  # 대략 앞 30줄
            abstract = head[:1200]

        docs.append({
            "title": title,
            "abstract": abstract,
            "content": content,
            "metadata": meta,
        })
    return docs


def _load_documents(input_path: str, json_encoding: str = "utf-8", recursive: bool = True) -> List[Dict[str, Any]]:
    """
    Load documents from:
      - JSON file (list[doc])  OR
      - single PDF file        OR
      - directory of PDFs (recursively by default)
    """
    if _is_json_file(input_path):
        return _load_documents_from_json(input_path, encoding=json_encoding)

    pdfs = _gather_pdf_files(input_path, recursive=recursive)
    if not pdfs:
        raise ValueError(
            f"Input '{input_path}' is neither a JSON file nor a PDF file/folder containing PDFs."
        )
    return _load_documents_from_pdfs(pdfs)


# ---------------------------------------------------------------------------
# Embedding & DB
# ---------------------------------------------------------------------------

def _prepare_records(docs: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, str, str, List[float]]]:
    """Embed documents and return records ready for DB insertion."""
    embedder = EmbedderAgent("local", "BAAI/bge-m3")
    records: List[Tuple[str, str, str, str, List[float]]] = []
    for doc in docs:
        title = doc.get("title")
        abstract = doc.get("abstract")
        content = doc.get("content")
        metadata = json.dumps(doc.get("metadata") or {}, ensure_ascii=False)

        # 임베딩 텍스트: content 우선, 없으면 title+abstract
        text = content or " ".join(filter(None, [title, abstract])) or ""
        if not text.strip():
            # 완전 비어있으면 스킵
            print(f"[WARN] Empty text for document titled '{title}'. Skipping.", file=sys.stderr)
            continue

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
    parser = argparse.ArgumentParser(description="Store paper embeddings (JSON or PDF)")
    parser.add_argument("--input", required=True,
                        help="Path to JSON file, a PDF file, or a folder containing PDFs")
    parser.add_argument("--encoding", default="utf-8",
                        help="JSON file encoding (default: utf-8)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Do not search subdirectories when input is a folder")
    args = parser.parse_args(argv)

    docs = _load_documents(args.input, json_encoding=args.encoding, recursive=not args.no_recursive)
    records = _prepare_records(docs)
    n = store_papers(records)
    print(f"Inserted {n} papers into papers_embeddings")


if __name__ == "__main__":
    main()
