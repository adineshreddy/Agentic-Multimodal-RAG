"""
Cross-encoder re-ranker for improving retrieval precision.

Uses a lightweight cross-encoder model that scores (query, passage) pairs
directly — much more accurate than bi-encoder cosine similarity alone.
The model runs locally on CPU and requires no API keys.
"""
from __future__ import annotations

from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logger.info(f"Loading cross-encoder: {model_name}")
    return CrossEncoder(model_name, max_length=512)


def rerank(query: str, passages: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-rank a list of passages using a cross-encoder.

    Args:
        query:    The user's original query.
        passages: List of dicts, each must have a "content" key.
        top_k:    How many results to keep after re-ranking.

    Returns:
        The top_k passages sorted by cross-encoder relevance score (descending),
        each augmented with a "rerank_score" key.
    """
    if not passages:
        return []

    ce = _get_cross_encoder()

    pairs = [(query, p["content"]) for p in passages]
    scores = ce.predict(pairs)

    for p, s in zip(passages, scores):
        p["rerank_score"] = float(s)

    ranked = sorted(passages, key=lambda p: p["rerank_score"], reverse=True)

    logger.info(
        f"Re-ranked {len(passages)} passages → top score: "
        f"{ranked[0]['rerank_score']:.4f}, "
        f"bottom: {ranked[-1]['rerank_score']:.4f}"
    )
    return ranked[:top_k]
