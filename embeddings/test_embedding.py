from embeddings.retrieve_embeddings import retrieve_embeddings
t_df, c_df = retrieve_embeddings("cloudsql-pg", SCHEMA="public")


print("tables:", t_df.shape, "\n", t_df.head(2))
print("columns:", c_df.shape, "\n", c_df.head(2))

# 임베딩 길이(첫 행) 확인
print("table emb dim:", len(t_df.iloc[0]["embedding"]))
print("column emb dim:", len(c_df.iloc[0]["embedding"]))