# embeddings/store_embeddings.py  (이전 코드 형식으로 재작성)
import asyncio
import asyncpg
import numpy as np
import pandas as pd

from typing import Optional

from pgvector.asyncpg import register_vector
from google.cloud.sql.connector import Connector
from google.cloud import bigquery

from agents import EmbedderAgent
from utilities import (
    VECTOR_STORE, PROJECT_ID, PG_INSTANCE, PG_DATABASE, PG_USER, PG_PASSWORD, PG_REGION,
    BQ_OPENDATAQNA_DATASET_NAME, BQ_REGION, EMBEDDING_MODEL
)

embedder = EmbedderAgent("local")

# ---------------- Helpers ----------------
def _as_list(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    raise TypeError(f"Unsupported embedding type: {type(v)}")

def _infer_dim_from_df(df: pd.DataFrame) -> Optional[int]:
    if df is None or df.empty:
        return None
    for _, row in df.iterrows():
        emb = _as_list(row.get("embedding"))
        if emb is not None:
            return len(emb)
    return None

# ---------------- Postgres DDL (이전 필드 구성 유지 + UNIQUE 추가) ----------------
def _ddl_tde(dim: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS table_details_embeddings(
        id BIGSERIAL PRIMARY KEY,
        source_type     VARCHAR(100)  NOT NULL,
        user_grouping   VARCHAR(100)  NOT NULL,
        table_schema    VARCHAR(1024) NOT NULL,
        table_name      VARCHAR(1024) NOT NULL,
        content         TEXT,
        embedding       vector({dim}),
        UNIQUE (user_grouping, table_name)
    );
    """

def _ddl_tcde(dim: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS tablecolumn_details_embeddings(
        id BIGSERIAL PRIMARY KEY,
        source_type     VARCHAR(100)  NOT NULL,
        user_grouping   VARCHAR(100)  NOT NULL,
        table_schema    VARCHAR(1024) NOT NULL,
        table_name      VARCHAR(1024) NOT NULL,
        column_name     VARCHAR(1024) NOT NULL,
        content         TEXT,
        embedding       vector({dim}),
        UNIQUE (user_grouping, table_name, column_name)
    );
    """

def _ddl_example(dim: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS example_prompt_sql_embeddings(
        id BIGSERIAL PRIMARY KEY,
        user_grouping           VARCHAR(1024) NOT NULL,
        example_user_question   TEXT          NOT NULL,
        example_generated_sql   TEXT          NOT NULL,
        embedding               vector({dim}),
        created_at              TIMESTAMPTZ DEFAULT now(),
        UNIQUE (user_grouping, example_user_question)
    );
    """

# ---------------- Main entry (이전 스타일 시그니처) ----------------
async def store_schema_embeddings(
    table_details_embeddings: pd.DataFrame,
    tablecolumn_details_embeddings: pd.DataFrame,
    project_id: str,
    instance_name: str,
    database_name: str,
    schema: str,
    database_user: str,
    database_password: str,
    region: str,
    VECTOR_STORE: str
) -> str:
    """
    Store the vectorised table and column details in the DB/warehouse.
    (이전 코드 스타일 유지: cloudsql-pgvector / bigquery-vector 분기)
    """

    # ---- 임베딩 차원 결정 ----
    dim_t = _infer_dim_from_df(table_details_embeddings)
    dim_c = _infer_dim_from_df(tablecolumn_details_embeddings)
    embedding_dim = dim_t or dim_c or 1024

    if VECTOR_STORE == "cloudsql-pgvector":
        loop = asyncio.get_running_loop()
        async with Connector(loop=loop) as connector:
            conn: asyncpg.Connection = await connector.connect_async(
                f"{project_id}:{region}:{instance_name}",
                "asyncpg",
                user=database_user,
                password=database_password,
                db=database_name,
            )
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await register_vector(conn)

                # DDL (이전 필드 구성 + UNIQUE)
                await conn.execute(_ddl_tde(embedding_dim))
                await conn.execute(_ddl_tcde(embedding_dim))
                await conn.execute(_ddl_example(embedding_dim))

                # 업서트 문 (이전 키 조합과 정합)
                SQL_UPSERT_TDE = """
                INSERT INTO table_details_embeddings
                  (source_type, user_grouping, table_schema, table_name, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_grouping, table_name)
                DO UPDATE SET
                  source_type = EXCLUDED.source_type,
                  table_schema = EXCLUDED.table_schema,
                  content = EXCLUDED.content,
                  embedding = EXCLUDED.embedding;
                """

                SQL_UPSERT_TCDE = """
                INSERT INTO tablecolumn_details_embeddings
                  (source_type, user_grouping, table_schema, table_name, column_name, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (user_grouping, table_name, column_name)
                DO UPDATE SET
                  source_type = EXCLUDED.source_type,
                  table_schema = EXCLUDED.table_schema,
                  content = EXCLUDED.content,
                  embedding = EXCLUDED.embedding;
                """

                # 테이블 임베딩 저장
                if table_details_embeddings is not None and not table_details_embeddings.empty:
                    for _, row in table_details_embeddings.iterrows():
                        emb = _as_list(row.get("embedding"))
                        await conn.execute(
                            SQL_UPSERT_TDE,
                            row["source_type"], row["user_grouping"], row["table_schema"], row["table_name"],
                            row.get("content"), emb
                        )

                # 컬럼 임베딩 저장
                if tablecolumn_details_embeddings is not None and not tablecolumn_details_embeddings.empty:
                    for _, row in tablecolumn_details_embeddings.iterrows():
                        emb = _as_list(row.get("embedding"))
                        await conn.execute(
                            SQL_UPSERT_TCDE,
                            row["source_type"], row["user_grouping"], row["table_schema"], row["table_name"],
                            row["column_name"], row.get("content"), emb
                        )

            finally:
                await conn.close()

        return "Embeddings are stored successfully (Postgres)."

    elif VECTOR_STORE == "bigquery-vector":
        client = bigquery.Client(project=project_id)

        # 테이블 보장 (이전 필드 구성 반영)
        client.query_and_wait(f'''
            CREATE TABLE IF NOT EXISTS `{project_id}.{schema}.table_details_embeddings` (
                source_type STRING NOT NULL,
                user_grouping STRING NOT NULL,
                table_schema STRING NOT NULL,
                table_name STRING NOT NULL,
                content STRING,
                embedding ARRAY<FLOAT64>
            )
        ''')
        client.query_and_wait(f'''
            CREATE TABLE IF NOT EXISTS `{project_id}.{schema}.tablecolumn_details_embeddings` (
                source_type STRING NOT NULL,
                user_grouping STRING NOT NULL,
                table_schema STRING NOT NULL,
                table_name STRING NOT NULL,
                column_name STRING NOT NULL,
                content STRING,
                embedding ARRAY<FLOAT64>
            )
        ''')
        client.query_and_wait(f'''
            CREATE TABLE IF NOT EXISTS `{project_id}.{schema}.example_prompt_sql_embeddings` (
                user_grouping STRING NOT NULL,
                example_user_question STRING NOT NULL,
                example_generated_sql STRING NOT NULL,
                embedding ARRAY<FLOAT64>
            )
        ''')

        # 테이블 임베딩: 키 조합으로 선삭제 → 적재
        if table_details_embeddings is not None and not table_details_embeddings.empty:
            keys = table_details_embeddings[["user_grouping", "table_name"]].apply(tuple, axis=1).tolist()
            if keys:
                where = " OR ".join([f"(user_grouping = '{u}' AND table_name = '{t}')" for (u, t) in keys])
                client.query_and_wait(f'''
                    DELETE FROM `{project_id}.{schema}.table_details_embeddings`
                    WHERE {where}
                ''')
            client.load_table_from_dataframe(
                table_details_embeddings, f'{project_id}.{schema}.table_details_embeddings'
            )

        # 컬럼 임베딩: 키 조합으로 선삭제 → 적재
        if tablecolumn_details_embeddings is not None and not tablecolumn_details_embeddings.empty:
            keys = tablecolumn_details_embeddings[["user_grouping", "table_name", "column_name"]].apply(tuple, axis=1).tolist()
            if keys:
                where = " OR ".join([f"(user_grouping = '{u}' AND table_name = '{t}' AND column_name = '{c}')" for (u, t, c) in keys])
                client.query_and_wait(f'''
                    DELETE FROM `{project_id}.{schema}.tablecolumn_details_embeddings`
                    WHERE {where}
                ''')
            client.load_table_from_dataframe(
                tablecolumn_details_embeddings, f'{project_id}.{schema}.tablecolumn_details_embeddings'
            )

        return "Embeddings are stored successfully (BigQuery)."

    else:
        raise ValueError("Please provide a valid Vector Store ('cloudsql-pgvector' or 'bigquery-vector').")

# ---------------- add_sql_embedding (이전 스타일, Postgres/BigQuery 분기) ----------------
async def add_sql_embedding(user_question: str, generated_sql: str, database: str) -> int:
    emb = embedder.create(user_question)
    emb_list = _as_list(emb)
    cleaned_sql = (generated_sql or "").replace("\r", " ").replace("\n", " ")

    if VECTOR_STORE == "cloudsql-pgvector":
        loop = asyncio.get_running_loop()
        async with Connector(loop=loop) as connector:
            conn: asyncpg.Connection = await connector.connect_async(
                f"{PROJECT_ID}:{PG_REGION}:{PG_INSTANCE}",
                "asyncpg",
                user=PG_USER,
                password=PG_PASSWORD,
                db=PG_DATABASE,
            )
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await register_vector(conn)

                # 테이블 보장 (임베딩 차원은 현재 생성한 벡터 길이 사용)
                dim = len(emb_list) if emb_list is not None else 1024
                await conn.execute(_ddl_example(dim))

                # UPSERT
                await conn.execute("""
                    INSERT INTO example_prompt_sql_embeddings
                      (user_grouping, example_user_question, example_generated_sql, embedding)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_grouping, example_user_question)
                    DO UPDATE SET
                      example_generated_sql = EXCLUDED.example_generated_sql,
                      embedding = EXCLUDED.embedding,
                      created_at = now();
                """, database, user_question, cleaned_sql, emb_list)

            finally:
                await conn.close()
        return 1

    elif VECTOR_STORE == "bigquery-vector":
        client = bigquery.Client(project=PROJECT_ID)
        client.query_and_wait(f'''
            CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{BQ_OPENDATAQNA_DATASET_NAME}.example_prompt_sql_embeddings` (
                user_grouping STRING NOT NULL,
                example_user_question STRING NOT NULL,
                example_generated_sql STRING NOT NULL,
                embedding ARRAY<FLOAT64>
            )
        ''')
        client.query_and_wait(f'''
            DELETE FROM `{PROJECT_ID}.{BQ_OPENDATAQNA_DATASET_NAME}.example_prompt_sql_embeddings`
            WHERE user_grouping = "{database}" AND example_user_question = "{user_question}"
        ''')
        emb_str = "NULL" if emb_list is None else str(emb_list)
        client.query_and_wait(f'''
            INSERT INTO `{PROJECT_ID}.{BQ_OPENDATAQNA_DATASET_NAME}.example_prompt_sql_embeddings`
            (user_grouping, example_user_question, example_generated_sql, embedding)
            VALUES ("{database}", "{user_question}", "{cleaned_sql}", {emb_str})
        ''')
        return 1

    else:
        raise ValueError("Please provide a valid Vector Store ('cloudsql-pgvector' or 'bigquery-vector').")

# ---------------- local test (optional) ----------------
if __name__ == '__main__':
    from retrieve_embeddings import retrieve_embeddings
    # from utilities import PG_SCHEMA
    VECTOR = "cloudsql-pgvector"
    t, c = retrieve_embeddings(VECTOR)
    asyncio.run(store_schema_embeddings(
        t, c, PROJECT_ID, PG_INSTANCE, PG_DATABASE,'public', PG_USER, PG_PASSWORD, PG_REGION, VECTOR_STORE=VECTOR
    ))
