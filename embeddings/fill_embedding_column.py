import psycopg
import numpy as np
from pgvector.psycopg import register_vector
from agents import EmbedderAgent  # 네가 쓰는 EmbedderAgent
from utilities import PG_CONN_STRING, EMBEDDING_MODEL, EMBEDDING_MODEL_PATH

MODEL_ID = EMBEDDING_MODEL_PATH or EMBEDDING_MODEL or "BAAI/bge-m3"
embedder = EmbedderAgent("local", MODEL_ID)  # 너의 구현에 맞춰 인자 조정

def to_list(v):
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return v  # agent가 리스트를 준다면 그대로

with psycopg.connect(PG_CONN_STRING) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        # 1) 임베딩 없는 것만 뽑기 (user_grouping별로도 가능)
        cur.execute("""
            SELECT id, user_grouping, example_user_question
            FROM example_prompt_sql_embeddings
            WHERE embedding IS NULL
            ORDER BY id
        """)
        rows = cur.fetchall()

        for _id, grouping, question in rows:
            emb = embedder.create(question)   # → 1024 floats
            emb = to_list(emb)
            cur.execute("""
                UPDATE example_prompt_sql_embeddings
                SET embedding = %s
                WHERE id = %s
            """, (emb, _id))

    conn.commit()
print("Backfill done.")