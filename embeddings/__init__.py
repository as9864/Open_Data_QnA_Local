from .retrieve_embeddings import retrieve_embeddings

try:  # Optional dependency: store_embeddings requires psycopg
    from .store_embeddings import store_embeddings as store_schema_embeddings, store_embeddings
except ModuleNotFoundError:  # pragma: no cover - import guard
    def store_embeddings(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("psycopg is required to store embeddings")

    def store_schema_embeddings(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError("psycopg is required to store embeddings")

__all__ = ["retrieve_embeddings", "store_embeddings", "store_schema_embeddings"]