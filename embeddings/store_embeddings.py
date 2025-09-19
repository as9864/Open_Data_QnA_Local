# embeddings/store_embedding.py
from __future__ import annotations

import asyncio
from typing import Iterable, Optional, Tuple, List, Dict, Any
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from agents import EmbedderAgent

# 우리 유틸에서 가져옴 (이미 만들어둔 값 사용)
from utilities import PG_VECTOR_CONN, EMBEDDING_MODEL, VECTOR_STORE, PG_LOCAL, LOCAL_USER_GROUPING

# ─────────────────────────────────────────────────────────────
# 연결 문자열 정규화 (sqlalchemy 스타일 → libpq)
# ─────────────────────────────────────────────────────────────
def _normalize_pg_url(url: str) -> str:
    return (url or "").strip()\
        .replace("postgresql+psycopg2://", "postgresql://")\
        .replace("postgres+psycopg2://", "postgresql://")\
        .replace("postgres://", "postgresql://")

_PG_CONNSTR = _normalize_pg_url(PG_VECTOR_CONN or "")
if not _PG_CONNSTR:
    raise RuntimeError("PG_VECTOR_CONN (또는 LOCAL_PG_CONN) 값이 config.ini에 설정되어 있어야 합니다.")

def _pg_connect() -> psycopg.Connection:
    # libpq URL로 접속 (필요시 kwargs 파싱 로직을 추가해도 됨)
    return psycopg.connect(_PG_CONNSTR)

# ─────────────────────────────────────────────────────────────
# 유틸: 차원/리스트 변환
# ─────────────────────────────────────────────────────────────
def _to_list(v: Any) -> List[float]:
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    # 문자열로 저장된 경우 등 방어
    if isinstance(v, (tuple,)):
        return list(v)
    raise TypeError(f"embedding 타입을 리스트로 변환할 수 없습니다: {type(v)}")

def _infer_dim(df) -> int:
    # 첫 행에서 embedding 길이 추론
    for _, row in df.iterrows():
        emb = _to_list(row["embedding"])
        return len(emb)
    raise ValueError("빈 DataFrame 입니다. embedding을 저장할 수 없습니다.")

# ─────────────────────────────────────────────────────────────
# 스키마 준비: 확장/테이블/인덱스/제약조건
# ─────────────────────────────────────────────────────────────
DDL_TABLE = """
CREATE TABLE IF NOT EXISTS table_details_embeddings (
  id BIGSERIAL PRIMARY KEY,
  table_schema TEXT NOT NULL,
  table_name   TEXT NOT NULL,
  user_goupring TEXT NOT NULL,
  source_type TEXT NOT NULL,
  embedding    vector(%s) ,
  content      TEXT,
  UNIQUE (table_schema, table_name)
);
"""

"""CREATE TABLE IF NOT EXISTS table_details_embeddings(
                                    source_type VARCHAR(100) NOT NULL,
                                    user_grouping VARCHAR(100) NOT NULL,
                                    table_schema VARCHAR(1024) NOT NULL,
                                    table_name VARCHAR(1024) NOT NULL,
                                    content TEXT,
                                    embedding vector(768))"""




DDL_COLUMN = """
CREATE TABLE IF NOT EXISTS tablecolumn_details_embeddings (
  id BIGSERIAL PRIMARY KEY,
  source_type TEXT NOT NULL,
  user_grouping TEXT NOT NULL,
  table_schema TEXT NOT NULL,
  table_name   TEXT NOT NULL,
  column_name  TEXT NOT NULL,
  embedding    vector(%s) ,
  content      TEXT NOT NULL,
  UNIQUE (table_schema, table_name, column_name)
);
"""




# 코사인 검색 기준 HNSW 인덱스 (임베딩은 보통 정규화해서 코사인 권장)
IDX_TABLE = """
CREATE INDEX IF NOT EXISTS idx_tbl_emb_hnsw_cos
  ON table_details_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=64);
"""
IDX_COLUMN = """
CREATE INDEX IF NOT EXISTS idx_col_emb_hnsw_cos
  ON tablecolumn_details_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=64);
"""

def _prepare_schema(dim: int) -> None:
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"invalid embedding dimension: {dim}")

    # ⚠️ 타입 차원은 반드시 상수로 인라인해야 함 (파라미터 바인딩 X)
    ddl_table = f"""
    CREATE TABLE IF NOT EXISTS table_details_embeddings (
      id BIGSERIAL PRIMARY KEY,
  source_type TEXT NOT NULL,
  user_grouping TEXT NOT NULL,
      table_schema TEXT NOT NULL,
      table_name   TEXT NOT NULL,
      embedding    vector({dim}) ,
      content      TEXT NOT NULL,
      UNIQUE (user_grouping,table_schema, table_name)
    );
    """

    ddl_column = f"""
    CREATE TABLE IF NOT EXISTS tablecolumn_details_embeddings (
      id BIGSERIAL PRIMARY KEY,
      source_type TEXT NOT NULL,
      user_grouping TEXT NOT NULL,
      table_schema TEXT NOT NULL,
      table_name   TEXT NOT NULL,
      column_name  TEXT NOT NULL,
      embedding    vector({dim}) ,
      content      TEXT NOT NULL,
      UNIQUE (user_grouping,table_schema, table_name, column_name)
    );
    """

    with _pg_connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(ddl_table)   # ← 파라미터 없이 실행
            cur.execute(ddl_column)  # ← 파라미터 없이 실행
            cur.execute(IDX_TABLE)
            cur.execute(IDX_COLUMN)
        conn.commit()

# ─────────────────────────────────────────────────────────────
# 저장 로직 (UPSERT)
# ─────────────────────────────────────────────────────────────
SQL_UPSERT_TABLE = """
INSERT INTO table_details_embeddings
  (source_type, user_grouping, table_schema, table_name, embedding, content)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (user_grouping, table_schema,table_name)
DO UPDATE SET
  source_type = EXCLUDED.source_type,
  table_schema = EXCLUDED.table_schema,
  embedding = EXCLUDED.embedding,
  content = EXCLUDED.content;
"""

SQL_UPSERT_COLUMN = """
INSERT INTO tablecolumn_details_embeddings
  (source_type, user_grouping, table_schema, table_name, column_name, embedding, content)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (user_grouping,table_schema, table_name, column_name)
DO UPDATE SET
  source_type = EXCLUDED.source_type,
  table_schema = EXCLUDED.table_schema,
  embedding = EXCLUDED.embedding,
  content = EXCLUDED.content;
"""











def store_embeddings(
    table_df, column_df,
    replace_schema: Optional[str] = None,
) -> Tuple[int, int]:
    """
    retrieve_embeddings() 가 반환한 두 DataFrame을 pgvector 테이블에 저장.
    - replace_schema: "public" 처럼 넘기면 해당 스키마의 기존 레코드를 먼저 삭제 후 저장
    반환: (저장된 table row 수, column row 수)
    """
    if table_df is None or column_df is None:
        raise ValueError("table_df/column_df 가 None 입니다.")

    # 차원 자동 추론 (둘이 다르면 table_df 우선, 다르면 오류)
    tdim = _infer_dim(table_df) if len(table_df) else None
    cdim = _infer_dim(column_df) if len(column_df) else None
    dim = tdim or cdim
    if dim is None:
        raise ValueError("저장할 embedding 데이터가 없습니다.")
    if tdim and cdim and tdim != cdim:
        raise ValueError(f"table({tdim})/column({cdim}) 임베딩 차원이 다릅니다. 동일 모델로 생성해야 합니다.")

    _prepare_schema(dim)

    # psycopg 연결/등록
    n_table = n_col = 0
    with _pg_connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # 선택: 특정 스키마 교체
            if replace_schema:
                cur.execute("DELETE FROM table_details_embeddings  WHERE table_schema = %s;", (replace_schema,))
                cur.execute("DELETE FROM tablecolumn_details_embeddings WHERE table_schema = %s;", (replace_schema,))

            # table embeddings
            if len(table_df):
                rows = []
                for _, r in table_df.iterrows():
                    emb = _to_list(r["embedding"])
                    rows.append((PG_LOCAL, LOCAL_USER_GROUPING, str(r["table_schema"]), str(r["table_name"]), emb, str(r["content"])))
                cur.executemany(SQL_UPSERT_TABLE, rows)
                n_table = len(rows)

            # column embeddings
            if len(column_df):
                rows = []
                for _, r in column_df.iterrows():
                    emb = _to_list(r["embedding"])
                    # column_name 없을 수도 있다면 방어
                    col_name = str(r.get("column_name", "")) or ""
                    rows.append((PG_LOCAL, LOCAL_USER_GROUPING,str(r["table_schema"]), str(r["table_name"]), col_name, emb, str(r["content"])))
                cur.executemany(SQL_UPSERT_COLUMN, rows)
                n_col = len(rows)

        conn.commit()

    # 선택: 통계 갱신 (플랜 품질 향상)
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("ANALYZE table_details_embeddings;")
            cur.execute("ANALYZE tablecolumn_details_embeddings;")
        conn.commit()

    return n_table, n_col

embedder = EmbedderAgent("local")





# 기존 import/유틸(_pg_connect, register_vector 등) 그대로 활용

async def add_sql_embedding(user_question, generated_sql, database):
    """
    로컬 PostgreSQL(pgvector) 전용.
    - 동일 (user_grouping, example_user_question) 키는 UPSERT
    - 테이블/인덱스/확장은 자동 보장
    """
    # 1) 임베딩 생성
    emb = embedder.create(user_question)
    # numpy/리스트 어느 쪽이든 리스트 변환
    if isinstance(emb, np.ndarray):
        emb = emb.astype(float).tolist()
    elif isinstance(emb, (tuple,)):
        emb = list(emb)
    elif not isinstance(emb, list):
        raise TypeError(f"embedding 타입이 list/ndarray가 아님: {type(emb)}")
    dim = len(emb)

    # 2) SQL 텍스트 정리
    cleaned_sql = (generated_sql or "").replace("\r", " ").replace("\n", " ")

    # 3) 로컬 DB 연결(동기 psycopg 사용)
    with _pg_connect() as conn:
        register_vector(conn)  # pgvector 어댑터 등록
        with conn.cursor() as cur:
            # 확장/테이블/인덱스 보장 (차원은 최초 생성 시점 기준으로 고정)
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS example_prompt_sql_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    user_grouping TEXT NOT NULL,
                    example_user_question TEXT NOT NULL,
                    example_generated_sql TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    UNIQUE (user_grouping, example_user_question)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_example_embed_hnsw_cos
                ON example_prompt_sql_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m=16, ef_construction=64);
            """)

            # UPSERT
            cur.execute(
                """
                INSERT INTO example_prompt_sql_embeddings
                    (user_grouping, example_user_question, example_generated_sql, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_grouping, example_user_question)
                DO UPDATE SET
                    example_generated_sql = EXCLUDED.example_generated_sql,
                    embedding = EXCLUDED.embedding,
                    created_at = now();
                """,
                (database, user_question, cleaned_sql, emb)
            )
        conn.commit()

    return 1


# ─────────────────────────────────────────────────────────────
# 사용 예시
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # retrieve → store end-to-end 테스트
    from embeddings.retrieve_embeddings import retrieve_embeddings
    tdf, cdf = retrieve_embeddings("cloudsql-pg", SCHEMA="public")
    saved = store_embeddings(tdf, cdf, replace_schema="public")
    print("stored:", saved)

    tdf, cdf = retrieve_embeddings("cloudsql-pg", SCHEMA="fhir_to_cdm")
    saved = store_embeddings(tdf, cdf, replace_schema="fhir_to_cdm")

