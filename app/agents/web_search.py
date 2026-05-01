"""DuckDuckGo web search — free, no API key required."""
from dataclasses import dataclass

from loguru import logger


@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str


def duckduckgo_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Run a DuckDuckGo text search and return structured results."""
    try:
        # Try new package name first (ddgs), fall back to old name
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
            for r in raw
        ]
    except Exception as exc:
        logger.error(f"Web search failed: {exc}")
        return []


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No web results found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}\nURL: {r.url}\n{r.snippet}")
    return "\n\n---\n\n".join(parts)
