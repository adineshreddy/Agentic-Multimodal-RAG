import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embeddings import get_embedding_model

settings = get_settings()

RAW_COLLECTION_NAME = "rag_documents_raw"
SUMMARY_COLLECTION_NAME = "rag_documents_summary"


def _get_store(collection_name: str) -> Chroma:
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_db_path,
    )


@lru_cache(maxsize=1)
def get_raw_vectorstore() -> Chroma:
    """Return the raw-chunk vectorstore."""
    return _get_store(RAW_COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_summary_vectorstore() -> Chroma:
    """Return the summary-node vectorstore."""
    return _get_store(SUMMARY_COLLECTION_NAME)


def get_vectorstore() -> Chroma:
    """Backward-compatible alias for the raw vectorstore."""
    return get_raw_vectorstore()


def clear_vectorstore_caches() -> None:
    get_raw_vectorstore.cache_clear()
    get_summary_vectorstore.cache_clear()


def _add_documents(vs: Chroma, docs: list[Document], ids: list[str] | None = None) -> list[str]:
    if not docs:
        return []
    ids = vs.add_documents(docs, ids=ids)
    return ids


def add_documents(docs: list[Document], ids: list[str] | None = None) -> list[str]:
    """Backward-compatible alias for raw-chunk insertion."""
    return add_raw_documents(docs, ids=ids)


def add_raw_documents(docs: list[Document], ids: list[str] | None = None) -> list[str]:
    return _add_documents(get_raw_vectorstore(), docs, ids=ids)


def add_summary_documents(docs: list[Document], ids: list[str] | None = None) -> list[str]:
    return _add_documents(get_summary_vectorstore(), docs, ids=ids)


def _where_for_user(user_id: int, **extra: object) -> dict:
    filters = [{"user_id": user_id}]
    for key, value in extra.items():
        if value is None:
            continue
        filters.append({key: value})
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def similarity_search(
    query: str,
    *,
    user_id: int,
    k: int | None = None,
    document_id: str | None = None,
) -> list:
    """Return top-k raw chunks with relevance scores for the given query."""
    vs = get_raw_vectorstore()
    k = k or settings.top_k_results
    return vs.similarity_search_with_relevance_scores(
        query,
        k=k,
        filter=_where_for_user(user_id, document_id=document_id, node_type="raw_chunk"),
    )


def _source_where(source: str, user_id: int) -> dict:
    return _where_for_user(user_id, source=source)


def summary_similarity_search(query: str, *, user_id: int, k: int | None = None) -> list:
    """Return top-k summary nodes with relevance scores for the given query."""
    vs = get_summary_vectorstore()
    k = k or settings.summary_top_k
    return vs.similarity_search_with_relevance_scores(
        query,
        k=k,
        filter=_where_for_user(user_id),
    )


def get_chunks_by_source(source: str, *, user_id: int) -> list:
    """Return ALL chunks for a specific source file, ordered by chunk_index."""
    vs = get_raw_vectorstore()
    collection = vs._collection
    results = collection.get(
        where=_where_for_user(user_id, source=source, node_type="raw_chunk"),
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    paired = []
    for text, meta in zip(docs, metas):
        paired.append(Document(page_content=text, metadata=meta or {}))
    paired.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return paired


def get_chunks_by_document_id(document_id: str, *, user_id: int) -> list[Document]:
    """Return all raw chunks for a given document ID, ordered by chunk_index."""
    vs = get_raw_vectorstore()
    collection = vs._collection
    results = collection.get(
        where=_where_for_user(user_id, document_id=document_id, node_type="raw_chunk"),
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    paired = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(docs, metas)]
    paired.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return paired


def get_document_id_for_source(source: str, *, user_id: int) -> str | None:
    """Resolve a stable document_id for a given source filename."""
    for vs in (get_raw_vectorstore(), get_summary_vectorstore()):
        collection = vs._collection
        results = collection.get(
            where=_where_for_user(user_id, source=source),
            include=["metadatas"],
        )
        for meta in (results.get("metadatas") or []):
            document_id = (meta or {}).get("document_id")
            if isinstance(document_id, str) and document_id.strip():
                return document_id
    return None


def get_summary_nodes_by_document_id(
    document_id: str,
    *,
    user_id: int,
    node_types: list[str] | None = None,
) -> list[Document]:
    """Return summary nodes for a document, optionally filtered by node_type."""
    vs = get_summary_vectorstore()
    collection = vs._collection
    where: dict = _where_for_user(user_id, document_id=document_id)

    if node_types:
        type_filters = [{"node_type": node_type} for node_type in node_types if node_type]
        if type_filters:
            where = {"$and": [where, {"$or": type_filters}]}

    results = collection.get(where=where, include=["documents", "metadatas"])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    paired = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(docs, metas)]
    paired.sort(key=lambda d: d.metadata.get("summary_index", 0))
    return paired


def get_image_summaries_by_document_id(document_id: str, *, user_id: int) -> list[Document]:
    return get_summary_nodes_by_document_id(
        document_id,
        user_id=user_id,
        node_types=["image_summary"],
    )


def list_sources(*, user_id: int) -> list[str]:
    """Return a deduplicated list of source filenames stored in ChromaDB."""
    sources: set[str] = set()
    for vs in (get_raw_vectorstore(), get_summary_vectorstore()):
        collection = vs._collection
        results = collection.get(where=_where_for_user(user_id), include=["metadatas"])
        sources.update(
            meta.get("source", "unknown")
            for meta in (results["metadatas"] or [])
        )
    return sorted(sources)


def _delete_from_store(vs: Chroma, where: dict) -> int:
    collection = vs._collection
    results = collection.get(where=where, include=["metadatas"])
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def delete_source(source: str, *, user_id: int) -> int:
    """Delete all raw and summary nodes belonging to a given source file."""
    return delete_document_family(source=source, user_id=user_id)


def delete_document_family(*, source: str, user_id: int) -> int:
    raw_deleted = _delete_from_store(get_raw_vectorstore(), _source_where(source, user_id))
    summary_deleted = _delete_from_store(get_summary_vectorstore(), _source_where(source, user_id))
    return raw_deleted + summary_deleted
