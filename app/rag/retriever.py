"""
Retrieval layer with query optimization.

Pipeline:
  1. Multi-query expansion — LLM generates 2-3 rephrasings of the user's query
     to capture different angles / phrasings. Each variant is searched against
     ChromaDB, and results are merged + deduplicated.
  2. Cross-encoder re-ranking — a local cross-encoder model (no API) re-scores
     the merged candidates for much higher precision.
  3. Top-k results returned to the agent.

The multi-query step uses Groq (free), same as the main LLM. If the LLM call
fails (e.g. key not set yet), it silently falls back to single-query mode so
the app still works during setup.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from app.config import get_settings
from app.rag.reranker import rerank
from app.rag.vectorstore import (
    get_document_id_for_source,
    get_chunks_by_document_id,
    get_chunks_by_source,
    get_summary_nodes_by_document_id,
    list_sources,
    similarity_search,
    summary_similarity_search,
)

settings = get_settings()
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

# With cross-encoder re-ranking, we no longer gate on the initial bi-encoder
# score.  ChromaDB can return negative cosine-similarity values for tabular
# data (Excel), which previously caused every chunk to be silently dropped.
# The cross-encoder is the real quality filter now.
MIN_BI_ENCODER_SCORE = None  # disabled — let all candidates through to re-ranker


@dataclass
class RetrievedChunk:
    content: str
    source:  str
    score:   float
    metadata: dict


def _extract_query_date_variants(query: str) -> set[str]:
    variants: set[str] = set()
    lowered = query.lower()

    date_tokens = re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b", lowered)
    for token in date_tokens:
        variants.add(token)
        parsed = None
        for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(token, fmt).date()
                break
            except ValueError:
                continue
        if not parsed:
            continue
        variants.update(
            {
                f"{parsed.year}-{parsed.month:02d}-{parsed.day:02d}",
                f"{parsed.month}/{parsed.day}/{parsed.year}",
                f"{parsed.month}/{parsed.day}/{str(parsed.year)[-2:]}",
            }
        )
    return variants


def _apply_structured_date_intent_filter(query: str, candidates: list[dict]) -> list[dict]:
    """
    For table-row retrieval, prefer rows where date + intent co-occur on a line.
    This reduces confusion between similarly dated fields (e.g., admission vs discharge).
    """
    if not candidates:
        return candidates

    query_l = query.lower()
    admission_intent = any(token in query_l for token in ("admitted", "admission", "admit"))
    discharge_intent = any(token in query_l for token in ("discharged", "discharge"))
    intent_keywords: list[str] = []
    if admission_intent:
        intent_keywords.extend(["admitted", "admission", "admit"])
    if discharge_intent:
        intent_keywords.extend(["discharged", "discharge"])

    date_variants = _extract_query_date_variants(query_l)
    if not intent_keywords or not date_variants:
        return candidates

    condition_phrase = ""
    condition_match = re.search(r"\bfor\s+(.+?)\s+\bon\b", query_l)
    if condition_match:
        condition_phrase = condition_match.group(1).strip(" .,?!")

    def _is_table_row(candidate: dict) -> bool:
        meta = candidate.get("metadata") or {}
        return meta.get("content_type") == "table_row"

    targeted: list[dict] = []
    for c in candidates:
        if not _is_table_row(c):
            continue
        content_l = (c.get("content") or "").lower()
        if condition_phrase and condition_phrase not in content_l:
            continue

        sentence_match = False
        escaped_dates = [re.escape(d) for d in date_variants]
        for line in content_l.splitlines():
            if line.startswith("record:"):
                continue
            if admission_intent and any(
                re.search(
                    rf"\b(?:admitted|admission|admit)\b[^.\n]{{0,120}}(?<!\d){date_token}(?!\d)",
                    line,
                )
                for date_token in escaped_dates
            ):
                sentence_match = True
                break
            if discharge_intent and any(
                re.search(
                    rf"\b(?:discharged|discharge)\b[^.\n]{{0,120}}(?<!\d){date_token}(?!\d)",
                    line,
                )
                for date_token in escaped_dates
            ):
                sentence_match = True
                break

        if sentence_match:
            targeted.append(c)

    if targeted:
        logger.info(
            f"Structured date-intent filter matched {len(targeted)} table row candidates "
            f"for query '{query[:80]}'"
        )
        return targeted

    table_rows = [c for c in candidates if _is_table_row(c)]
    if table_rows and condition_phrase:
        logger.info(
            "Structured date-intent filter found no exact table-row match; "
            "returning explicit no-match notice to prevent hallucinated lookups."
        )
        intent_label = "admission" if admission_intent else "discharge" if discharge_intent else "date-intent"
        sample_source = table_rows[0].get("source", "table")
        notice = {
            "content": (
                "Structured table lookup result: no exact matching row was found for "
                f"{intent_label} date in {{{', '.join(sorted(date_variants))}}} "
                f"with condition containing '{condition_phrase}'."
            ),
            "source": sample_source,
            "score": 0.0,
            "metadata": {
                "source": sample_source,
                "content_type": "structured_notice",
                "node_type": "structured_filter_notice",
            },
        }
        return [notice]
    return candidates


# ── Multi-query generation ────────────────────────────────────────────────────

def _generate_query_variants(query: str, n: int = 3) -> list[str]:
    """
    Use the LLM to rephrase the user's query into n different search queries.
    Returns the original query + generated variants.
    Falls back to [query] alone if the LLM call fails.
    """
    try:
        from langchain_groq import ChatGroq

        load_dotenv(dotenv_path=_ENV_FILE, override=True)
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key or api_key == "your_groq_api_key_here":
            return [query]

        llm = ChatGroq(
            api_key=api_key,
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
            temperature=0.7,
            max_tokens=256,
        )
        prompt = (
            f"Generate {n} different search queries to find relevant documents "
            f"for the following question. Each query should approach the topic "
            f"from a different angle or use different keywords.\n\n"
            f"Original question: {query}\n\n"
            f"Return ONLY the queries, one per line, no numbering or bullets."
        )
        response = llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        # Always include the original query first
        all_queries = [query] + variants[:n]
        logger.info(f"Multi-query: generated {len(all_queries)} search variants")
        return all_queries

    except Exception as e:
        logger.warning(f"Multi-query generation failed, using original query: {e}")
        return [query]


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(query: str, *, user_id: int, k: int | None = None) -> list[RetrievedChunk]:
    """
    Multi-query retrieval with cross-encoder re-ranking.

    1. Generate query variants (original + LLM rephrasings).
    2. Search ChromaDB with each variant, merge + deduplicate.
    3. Re-rank merged results with cross-encoder.
    4. Return top-k.
    """
    k = k or settings.top_k_results

    # Step 1: generate query variants
    queries = _generate_query_variants(query)

    # Step 2a: summary-first retrieval
    summary_hits: list[dict] = _collect_summary_hits(queries, user_id=user_id)

    document_ids: list[str] = []
    for hit in summary_hits:
        document_id = hit["metadata"].get("document_id")
        if document_id and document_id not in document_ids:
            document_ids.append(document_id)
        if len(document_ids) >= max(2, settings.summary_top_k):
            break

    # Step 2b: expand summary parents to raw chunks + run targeted raw searches
    seen_contents: set[str] = set()
    candidates: list[dict] = []

    def _push_raw_candidate(doc, score: float) -> None:
        content_key = (doc.page_content or "")[:200]
        if not content_key or content_key in seen_contents:
            return
        seen_contents.add(content_key)
        candidates.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 4),
            "metadata": doc.metadata,
        })

    # Parent-child expansion from summary hits
    for document_id in document_ids:
        for doc in get_chunks_by_document_id(document_id, user_id=user_id):
            _push_raw_candidate(doc, score=0.0)

    fetch_per_query = max(k, 5)
    scoped_document_ids = document_ids or [None]
    for q in queries:
        for document_id in scoped_document_ids:
            raw_hits = similarity_search(q, user_id=user_id, k=fetch_per_query, document_id=document_id)
            for doc, score in raw_hits:
                _push_raw_candidate(doc, score=score)

    # If summary-scoped search didn't produce raw context, broaden to all docs.
    if not candidates:
        for q in queries:
            raw_hits = similarity_search(q, user_id=user_id, k=fetch_per_query, document_id=None)
            for doc, score in raw_hits:
                _push_raw_candidate(doc, score=score)

    if not candidates:
        # Rare fallback: summary-only document (e.g., OCR/image extraction failed for raw text)
        if not summary_hits:
            return []
        logger.warning("No raw candidates found; falling back to summary-node reranking.")
        reranked_summary = rerank(query, summary_hits, top_k=k)
        return [
            RetrievedChunk(
                content=r["content"],
                source=r["source"],
                score=r.get("rerank_score", r["score"]),
                metadata=r["metadata"],
            )
            for r in reranked_summary
        ]

    logger.info(
        f"Summary-first retrieval collected {len(candidates)} unique candidates "
        f"from {len(queries)} queries and {len(document_ids)} summary-matched documents"
    )

    candidates = _apply_structured_date_intent_filter(query, candidates)

    # Step 3: re-rank RAW candidates with cross-encoder
    reranked = rerank(query, candidates, top_k=k)

    # Step 4: convert to RetrievedChunk (grounded on raw chunks)
    chunks = [
        RetrievedChunk(
            content=r["content"],
            source=r["source"],
            score=r.get("rerank_score", r["score"]),
            metadata=r["metadata"],
        )
        for r in reranked
    ]

    # Inject the most relevant image summary nodes as supplemental visual context
    # when they belong to the same top-ranked documents.
    top_document_ids = {
        c.metadata.get("document_id")
        for c in chunks
        if c.metadata.get("document_id")
    }
    visual_hits = [
        hit for hit in summary_hits
        if (hit["metadata"].get("node_type") == "image_summary")
        and (
            not top_document_ids
            or hit["metadata"].get("document_id") in top_document_ids
        )
    ]
    for hit in visual_hits[:2]:
        chunks.append(
            RetrievedChunk(
                content=hit["content"],
                source=hit["source"],
                score=hit["score"],
                metadata=hit["metadata"],
            )
        )

    if chunks:
        logger.info(
            f"Final top result: score={chunks[0].score:.4f} "
            f"from '{chunks[0].source}'"
        )
    return chunks


def _collect_summary_hits(queries: list[str], *, user_id: int) -> list[dict]:
    best_by_key: dict[tuple[str, str, object], dict] = {}
    for q in queries:
        raw_summary_hits = summary_similarity_search(
            q,
            user_id=user_id,
            k=settings.summary_top_k,
        )
        for doc, score in raw_summary_hits:
            meta = doc.metadata or {}
            key = (
                meta.get("document_id", ""),
                meta.get("node_type", ""),
                meta.get("summary_index", meta.get("image_index", meta.get("table_index"))),
            )
            candidate = {
                "content": doc.page_content,
                "source": meta.get("source", "unknown"),
                "score": round(score, 4),
                "metadata": meta,
            }
            current = best_by_key.get(key)
            if current is None or candidate["score"] > current["score"]:
                best_by_key[key] = candidate

    ordered = sorted(best_by_key.values(), key=lambda h: h["score"], reverse=True)
    return ordered


# ── Document-name retrieval (unchanged) ───────────────────────────────────────

def resolve_document_name(query: str, *, user_id: int) -> str | None:
    """
    If the query mentions a known document by name (exact or partial match),
    return the matching source filename. Returns None if no match found.
    """
    sources = list_sources(user_id=user_id)
    if not sources:
        return None

    query_lower = query.lower()

    for src in sources:
        if src.lower() in query_lower:
            logger.info(f"Document name exact match: '{src}'")
            return src

    for src in sources:
        stem = src.rsplit(".", 1)[0].lower()
        if stem in query_lower:
            logger.info(f"Document stem match: '{src}' via stem '{stem}'")
            return src

        parts = [p for p in stem.replace("-", " ").replace("_", " ").split() if len(p) > 2]
        if len(parts) >= 2:
            matches = sum(1 for p in parts if p in query_lower)
            if matches >= len(parts) * 0.6:
                logger.info(f"Document partial match: '{src}' ({matches}/{len(parts)} words)")
                return src

    return None


def retrieve_full_document(
    source: str,
    *,
    user_id: int,
    document_id: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve ALL chunks for a specific document, in order."""
    docs = (
        get_chunks_by_document_id(document_id, user_id=user_id)
        if document_id
        else get_chunks_by_source(source, user_id=user_id)
    )
    if not docs and document_id:
        docs = get_chunks_by_source(source, user_id=user_id)
    return [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", source),
            score=1.0,
            metadata=doc.metadata,
        )
        for doc in docs
    ]


def retrieve_full_document_by_id(document_id: str, *, user_id: int) -> list[RetrievedChunk]:
    """Retrieve all raw chunks for a specific indexed document ID."""
    docs = get_chunks_by_document_id(document_id, user_id=user_id)
    return [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            score=1.0,
            metadata=doc.metadata,
        )
        for doc in docs
    ]


def resolve_document_id(source: str, *, user_id: int) -> str | None:
    return get_document_id_for_source(source, user_id=user_id)


def retrieve_document_summaries(
    document_id: str,
    *,
    user_id: int,
    node_types: list[str] | None = None,
) -> list[RetrievedChunk]:
    docs = get_summary_nodes_by_document_id(
        document_id,
        user_id=user_id,
        node_types=node_types,
    )
    return [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            score=1.0,
            metadata=doc.metadata,
        )
        for doc in docs
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a clean context block for the LLM."""
    if not chunks:
        return ""
    parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page")
        page_suffix = f" (page {page})" if page else ""
        parts.append(f"Source: {chunk.source}{page_suffix}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)
