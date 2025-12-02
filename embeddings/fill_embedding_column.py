import psycopg
import numpy as np
import time
from pgvector.psycopg import register_vector
from agents import EmbedderAgent
from utilities import PG_VECTOR_CONN, EMBEDDING_MODEL, EMBEDDING_MODEL_PATH

MODEL_ID = EMBEDDING_MODEL_PATH or EMBEDDING_MODEL or "BAAI/bge-m3"
embedder = EmbedderAgent("local", MODEL_ID)

# 한 번에 처리할 로우 수 (메모리 및 속도 조절용)
BATCH_SIZE = 1000

def to_list(v):
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return v  # agent가 리스트를 준다면 그대로
def create_embedding_text(row):
    (c_id, c_name, d_id, v_id, c_class_id, c_code) = row
    parts = [
        f"Concept Name: {c_name}" if c_name else "",
        f"Domain: {d_id}" if d_id else "",
        f"Vocabulary: {v_id}" if v_id else "",
        f"Class: {c_class_id}" if c_class_id else "",
        f"Code: {c_code}" if c_code else ""
    ]
    return ", ".join([p for p in parts if p])
    
with psycopg.connect(PG_VECTOR_CONN) as conn:
    register_vector(conn)
    
    total_processed = 0
    start_time = time.time()

    while True:
        with conn.cursor() as cur:
            # 1. 처리되지 않은 데이터 BATCH_SIZE 만큼만 조회
            cur.execute("""
                SELECT 
                    concept_id, concept_name, domain_id, vocabulary_id, 
                    concept_class_id, concept_code
                FROM public.concept
                WHERE embedding IS NULL
                ORDER BY concept_id
                LIMIT %s
            """, (BATCH_SIZE,))
            
            rows = cur.fetchall()
            
            # 더 이상 처리할 데이터가 없으면 종료
            if not rows:
                break
            
            # 2. 배치 단위로 임베딩 생성
            texts_to_embed = [create_embedding_text(row) for row in rows]
            concept_ids = [row[0] for row in rows]
            
            embeddings = embedder.create(texts_to_embed)
            
            # 3. 일괄 업데이트 (executemany 사용으로 속도 향상)
            update_data = [(to_list(emb), c_id) for emb, c_id in zip(embeddings, concept_ids)]
            
            if update_data:
                cur.executemany("""
                    UPDATE public.concept
                    SET embedding = %s
                    WHERE concept_id = %s
                """, update_data)
            
            # 4. 커밋 및 진행상황 출력
            conn.commit()
            total_processed += len(rows)
            elapsed_time = time.time() - start_time
            print(f"Current Batch: {len(rows)} rows / Total Processed: {total_processed} / Elapsed: {elapsed_time:.2f}s")

print("Backfill done.")