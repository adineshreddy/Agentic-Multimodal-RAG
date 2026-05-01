from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the sentence-transformers embedding model once and cache it."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
