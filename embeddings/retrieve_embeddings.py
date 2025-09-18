import re
import sys
import pandas as pd
from typing import Optional, List, Dict, Any

from dbconnectors import pgconnector, bqconnector
from agents import EmbedderAgent, DescriptionAgent
from utilities import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    DESCRIPTION_MODEL,
    USE_COLUMN_SAMPLES,
)

# ─────────────────────────────────────────────
# Embedder 초기화
# ─────────────────────────────────────────────
def _clean_model_id(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().strip('"').strip("'")
    for tok in (";", "#"):
        if tok in s:
            s = s.split(tok, 1)[0].strip()
    return s

_raw_id = EMBEDDING_MODEL_PATH or EMBEDDING_MODEL or "BAAI/bge-m3"
_model_id = _clean_model_id(_raw_id)

try:
    embedder = EmbedderAgent("local", _model_id)
except TypeError:
    embedder = EmbedderAgent("local")

# ─────────────────────────────────────────────
# 설명 생성기 설정
# ─────────────────────────────────────────────
_desc_model = (DESCRIPTION_MODEL or "").strip().lower()
_use_desc = _desc_model not in ("", "none", "off")

if _use_desc:
    descriptor = DescriptionAgent.DescriptionAgent(DESCRIPTION_MODEL)
else:
    class _NoopDesc:
        def generate_missing_descriptions(self, _src, table_df, col_df):
            return table_df, col_df
    descriptor = _NoopDesc()

# ─────────────────────────────────────────────
# Postgres 스키마 조회
# ─────────────────────────────────────────────
def _pg_table_schema_df(schema: str, table_names=None) -> pd.DataFrame:
    if hasattr(pgconnector, "return_table_schema_sql") and hasattr(pgconnector, "retrieve_df"):
        sql = pgconnector.return_table_schema_sql(schema, table_names=table_names)
        return pgconnector.retrieve_df(sql)
    import psycopg
    from utilities import LOCAL_PG_CONN, PG_CONN_STRING
    connstr = (LOCAL_PG_CONN or PG_CONN_STRING)

    names_filter = ""
    params = [schema]
    if table_names:
        names_filter = " AND c.relname = ANY(%s) "
        params.append(list(table_names) if not isinstance(table_names, list) else table_names)

    sql = f"""
    SELECT
      n.nspname AS table_schema,
      c.relname AS table_name,
      COALESCE(td.description, '') AS table_description,
      array_to_string(array_agg(a.attname ORDER BY a.attnum), ', ') AS table_columns
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_description td ON td.objoid = c.oid AND td.objsubid = 0
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
    WHERE c.relkind IN ('r','p') AND n.nspname = %s
      {names_filter}
    GROUP BY n.nspname, c.relname, td.description
    ORDER BY c.relname;
    """
    with psycopg.connect(connstr) as conn:
        return pd.read_sql_query(sql, conn, params=params)

def _pg_column_schema_df(schema: str, table_names=None) -> pd.DataFrame:
    if hasattr(pgconnector, "return_column_schema_sql") and hasattr(pgconnector, "retrieve_df"):
        sql = pgconnector.return_column_schema_sql(schema, table_names=table_names)
        return pgconnector.retrieve_df(sql)
    import psycopg
    from utilities import LOCAL_PG_CONN, PG_CONN_STRING
    connstr = (LOCAL_PG_CONN or PG_CONN_STRING)

    names_filter = ""
    params = [schema]
    if table_names:
        names_filter = " AND c.relname = ANY(%s) "
        params.append(list(table_names) if not isinstance(table_names, list) else table_names)

    sql = f"""
    SELECT
      n.nspname AS table_schema,
      c.relname AS table_name,
      a.attname  AS column_name,
      format_type(a.atttypid, a.atttypmod) AS data_type,
      COALESCE(cd.description, '') AS column_description,
      CASE WHEN ct.contype = 'p' AND a.attnum = ANY (ct.conkey) THEN 'PRIMARY KEY' ELSE '' END AS column_constraints
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
    LEFT JOIN pg_description cd ON cd.objoid = c.oid AND cd.objsubid = a.attnum
    LEFT JOIN pg_constraint ct ON ct.conrelid = c.oid AND ct.contype = 'p'
    WHERE c.relkind IN ('r','p') AND n.nspname = %s
      {names_filter}
    ORDER BY c.relname, a.attnum;
    """
    with psycopg.connect(connstr) as conn:
        return pd.read_sql_query(sql, conn, params=params)

# ─────────────────────────────────────────────
# BigQuery 스키마 조회 (폴백)
# ─────────────────────────────────────────────
def _bq_table_schema_df(dataset: str, table_names=None) -> pd.DataFrame:
    if hasattr(bqconnector, "return_table_schema_sql") and hasattr(bqconnector, "retrieve_df"):
        sql = bqconnector.return_table_schema_sql(dataset, table_names=table_names)
        return bqconnector.retrieve_df(sql)
    from google.cloud import bigquery
    client = bigquery.Client()
    query = f"""
    SELECT
      table_schema,
      table_name,
      "" AS table_description,
      STRING_AGG(column_name, ', ' ORDER BY ordinal_position) AS table_columns
    FROM `{client.project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    GROUP BY table_schema, table_name
    ORDER BY table_name
    """
    return client.query(query).result().to_dataframe()

def _bq_column_schema_df(dataset: str, table_names=None) -> pd.DataFrame:
    if hasattr(bqconnector, "return_column_schema_sql") and hasattr(bqconnector, "retrieve_df"):
        sql = bqconnector.return_column_schema_sql(dataset, table_names=table_names)
        return bqconnector.retrieve_df(sql)
    from google.cloud import bigquery
    client = bigquery.Client()
    query = f"""
    SELECT
      table_schema,
      table_name,
      column_name,
      data_type,
      '' AS column_description,
      '' AS column_constraints
    FROM `{client.project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    ORDER BY table_name, ordinal_position
    """
    return client.query(query).result().to_dataframe()

# ─────────────────────────────────────────────
# 임베딩 유틸
# ─────────────────────────────────────────────
def get_embedding_chunked(textinput: List[Dict[str, Any]], batch_size: int) -> pd.DataFrame:
    if not textinput:
        return pd.DataFrame(columns=["table_schema", "table_name", "content", "embedding", "column_name"])
    use_embed = hasattr(embedder, "embed")
    use_create = hasattr(embedder, "create")

    for i in range(0, len(textinput), batch_size):
        request = [x["content"] for x in textinput[i:i+batch_size]]
        if use_embed:
            response = embedder.embed(request)
        elif use_create:
            response = embedder.create(request)
        else:
            raise RuntimeError("EmbedderAgent에 embed/create 메서드가 없습니다.")
        for x, e in zip(textinput[i:i+batch_size], response):
            x["embedding"] = e

    return pd.DataFrame(textinput)

# ─────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────
def retrieve_embeddings(SOURCE: str, SCHEMA: str = "public", table_names=None):
    table_details_embeddings = pd.DataFrame(columns=["table_schema", "table_name", "content", "embedding"])
    column_details_embeddings = pd.DataFrame(columns=["table_schema", "table_name", "column_name", "content", "embedding"])

    try:
        if SOURCE == "cloudsql-pg":
            table_desc_df = _pg_table_schema_df(SCHEMA, table_names)
            column_name_df = _pg_column_schema_df(SCHEMA, table_names)

            table_desc_df, column_name_df = descriptor.generate_missing_descriptions(
                SOURCE, table_desc_df, column_name_df
            )

            column_name_df["sample_values"] = None
            if USE_COLUMN_SAMPLES and hasattr(pgconnector, "get_column_samples"):
                column_name_df = pgconnector.get_column_samples(column_name_df)

            # 테이블 임베딩
            table_details_chunked = []
            for _, row_aug in table_desc_df.iterrows():
                table_detailed_description = (
                    f"Table Name: {row_aug['table_name']} | "
                    f"Schema Name: {row_aug['table_schema']} | "
                    f"Table Description - {row_aug.get('table_description','')} | "
                    f"Columns List: [{row_aug.get('table_columns','')}]"
                )
                table_details_chunked.append({
                    "table_schema": row_aug["table_schema"],
                    "table_name": row_aug["table_name"],
                    "content": table_detailed_description,
                })
            table_details_embeddings = get_embedding_chunked(table_details_chunked, 10)

            # 컬럼 임베딩
            column_details_chunked = []
            for _, row_aug in column_name_df.iterrows():
                column_detailed_description = (
                    f"Schema Name:{row_aug['table_schema']} | "
                    f"Table Name: {row_aug['table_name']} | "
                    f"Column Name: {row_aug['table_schema']}.{row_aug['table_name']}.{row_aug['column_name']} "
                    f"(Data type: {row_aug.get('data_type','')}) | "
                    f"(column description: {row_aug.get('column_description','')})(constraints: {row_aug.get('column_constraints','')}) | "
                    f"(Sample Values in the Column: {row_aug.get('sample_values','')})"
                )
                column_details_chunked.append({
                    "table_schema": row_aug["table_schema"],
                    "table_name": row_aug["table_name"],
                    "column_name": row_aug["column_name"],
                    "content": column_detailed_description,
                })
            column_details_embeddings = get_embedding_chunked(column_details_chunked, 10)

        elif SOURCE == "bigquery":
            table_desc_df = _bq_table_schema_df(SCHEMA, table_names)
            column_name_df = _bq_column_schema_df(SCHEMA, table_names)

            table_desc_df, column_name_df = descriptor.generate_missing_descriptions(
                SOURCE, table_desc_df, column_name_df
            )

            column_name_df["sample_values"] = None
            if USE_COLUMN_SAMPLES and hasattr(bqconnector, "get_column_samples"):
                column_name_df = bqconnector.get_column_samples(column_name_df)

            # 테이블 임베딩
            table_details_chunked = []
            for _, row_aug in table_desc_df.iterrows():
                table_detailed_description = (
                    f"Full Table Name : {row_aug['table_schema']}.{row_aug['table_name']} | "
                    f"Table Columns List: [{row_aug.get('table_columns','')}] | "
                    f"Table Description: {row_aug.get('table_description','')}"
                )
                table_details_chunked.append({
                    "table_schema": row_aug["table_schema"],
                    "table_name": row_aug["table_name"],
                    "content": table_detailed_description,
                })
            table_details_embeddings = get_embedding_chunked(table_details_chunked, 10)

            # 컬럼 임베딩
            column_details_chunked = []
            for _, row_aug in column_name_df.iterrows():
                column_detailed_description = (
                    f"Column Name: {row_aug['table_schema']}.{row_aug['table_name']}.{row_aug['column_name']} | "
                    f"Full Table Name : {row_aug['table_schema']}.{row_aug['table_name']} | "
                    f"Data type: {row_aug.get('data_type','')} | "
                    f"Column description: {row_aug.get('column_description','')} | "
                    f"Column Constraints: {row_aug.get('column_constraints','')} | "
                    f"Sample Values in the Column : {row_aug.get('sample_values','')}"
                )
                column_details_chunked.append({
                    "table_schema": row_aug["table_schema"],
                    "table_name": row_aug["table_name"],
                    "column_name": row_aug["column_name"],
                    "content": column_detailed_description,
                })
            column_details_embeddings = get_embedding_chunked(column_details_chunked, 10)

    except Exception as e:
        print(f"[retrieve_embeddings] error: {e}", file=sys.stderr)

    return table_details_embeddings, column_details_embeddings


if __name__ == "__main__":
    SOURCE = "cloudsql-pg"
    t, c = retrieve_embeddings(SOURCE, SCHEMA="public")
    print("t:\n", t.head())
    print("c:\n", c.head())
