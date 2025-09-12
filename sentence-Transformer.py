from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")             # 다국어, 1024차원
# model = SentenceTransformer("BAAI/bge-small-en-v1.5") # 영어 위주, 384차원

print("dim:", model.get_sentence_embedding_dimension())  # bge-m3=1024, small=384
emb = model.encode(["테스트 문장입니다."], normalize_embeddings=True)
print(len(emb[0]))  # 1024 또는 384


e = model.encode("Represent this sentence for searching relevant passages: 테스트", normalize_embeddings=True)
print(len(e))  # 1024가 출력되어야 함
print(e)