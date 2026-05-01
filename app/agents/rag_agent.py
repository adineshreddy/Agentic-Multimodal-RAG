"""
LangGraph agentic RAG pipeline.

Decision loop:
  1. Retrieve top-k chunks from ChromaDB.
  2. If best chunk score ≥ RAG_SCORE_THRESHOLD → answer from documents.
  3. Otherwise fall back to DuckDuckGo web search.
  4. Return answer + source references.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from app.agents.web_search import duckduckgo_search, format_search_results
from app.config import get_settings
from app.rag.retriever import (
    RetrievedChunk,
    format_context,
    resolve_document_id,
    resolve_document_name,
    retrieve,
    retrieve_document_summaries,
    retrieve_full_document,
)

settings = get_settings()

# Absolute path to project .env
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

# Route to RAG if at least one retrieved chunk exceeds this score.
# Cross-encoder scores range roughly -10 to +10; anything above -3 is
# potentially relevant.  We use a very low threshold so that tabular
# data (Excel) — which always scores lower — still gets routed to RAG.
RAG_SCORE_THRESHOLD = -3.0

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to internal documents.

When answering:
- Base your answer on the provided document context.
- The context may include extracted text, table summaries, and image summaries from the uploaded documents.
- Cite the source document name, e.g. "According to [filename]...".
- If the context does not fully answer the question, say so and give your best answer.
- Be concise, accurate, and direct.
- Never fabricate information not present in the context.
- If the context is insufficient, explicitly say what is missing instead of guessing.
- If the context says "Structured table lookup result: no exact matching row was found...", clearly state that no exact matching row was found.
- Refuse harmful, illegal, or unsafe requests and offer a safer alternative.
- NEVER mention internal details like chunk numbers, chunk indices, relevance scores, or any retrieval metadata. Write naturally as if you read the document directly.

Conversation history (if any) is included for continuity.
"""

DOC_SUMMARY_PROMPT = """You are a helpful AI assistant. The user asked about a specific document.
The document content below may include extracted text and summaries of tables/images from that document.

Instructions:
- Provide a thorough, well-structured summary covering the key topics, arguments, and conclusions.
- If the user asked a specific question about this document, focus on answering it from the full content.
- Cite specific sections or pages when relevant.
- Be comprehensive, but acknowledge if the provided context appears truncated.
- If the context is insufficient, clearly say so and do not invent facts.
- Refuse harmful, illegal, or unsafe requests and offer a safer alternative.
- NEVER mention internal details like chunk numbers, chunk indices, or retrieval metadata. Write naturally as if you read the document directly.
"""

WEB_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question using the web search results below.
Summarize key information, cite source URLs where relevant, and be concise.
Refuse harmful, illegal, or unsafe requests and offer a safer alternative.
"""


class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    query:           str
    user_id:         int
    matched_document: str
    matched_document_id: str
    retrieved_chunks: list[RetrievedChunk]
    web_results:     str
    answer:          str
    sources:         list[dict]
    used_web_search: bool
    token_usage:     dict


def _build_llm() -> ChatGroq:
    # Always reload .env so any edits are picked up without a server restart.
    load_dotenv(dotenv_path=_ENV_FILE, override=True)

    api_key    = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com, "
            "add it to .env as GROQ_API_KEY=gsk_..., then retry."
        )
    return ChatGroq(
        api_key=api_key,
        model=groq_model,
        temperature=0.2,
        max_tokens=2048,
    )


def _extract_token_usage(response) -> dict:
    usage: dict[str, int] = {}

    response_meta = getattr(response, "response_metadata", {}) or {}
    token_usage = response_meta.get("token_usage") or {}
    if isinstance(token_usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = token_usage.get(key)
            if isinstance(value, (int, float)):
                usage[key] = int(value)

    usage_meta = getattr(response, "usage_metadata", {}) or {}
    if isinstance(usage_meta, dict):
        mapping = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }
        for src_key, dst_key in mapping.items():
            value = usage_meta.get(src_key)
            if isinstance(value, (int, float)):
                usage[dst_key] = int(value)

    return usage


def _grounding_support_ratio(answer: str, chunks: list[RetrievedChunk]) -> float:
    """
    Lightweight grounding check:
    ratio of meaningful answer tokens found in retrieved context.
    """
    answer_tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", answer.lower())
        if len(token) >= 4
    ]
    if len(answer_tokens) < 20:
        return 1.0

    context_tokens: set[str] = set()
    for chunk in chunks:
        context_tokens.update(
            token
            for token in re.findall(r"[a-zA-Z0-9']+", chunk.content.lower())
            if len(token) >= 4
        )
    if not context_tokens:
        return 0.0

    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return round(supported / max(1, len(answer_tokens)), 4)


def _apply_grounding_guardrail(answer: str, chunks: list[RetrievedChunk]) -> str:
    ratio = _grounding_support_ratio(answer, chunks)
    if ratio >= settings.grounding_min_ratio:
        return answer

    logger.warning(
        f"Low grounding support detected (ratio={ratio}, threshold={settings.grounding_min_ratio}). "
        "Appending safety disclaimer."
    )
    return (
        f"{answer}\n\n"
        "Note: Parts of this answer may be weakly grounded in the retrieved document context. "
        "For higher confidence, narrow the question or provide a more specific document section."
    )


# ── Graph nodes ────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    user_id = state["user_id"]

    # Check if the user is referring to a specific uploaded document by name
    matched = resolve_document_name(query, user_id=user_id)
    if matched:
        matched_document_id = resolve_document_id(matched, user_id=user_id) or ""
        logger.info(f"Document-name match: '{matched}' — pulling full document")
        chunks = retrieve_full_document(
            matched,
            user_id=user_id,
            document_id=matched_document_id or None,
        )
        if not chunks:
            logger.info(f"No raw chunks found for '{matched}', falling back to summary-first retrieval")
            chunks = retrieve(query, user_id=user_id)
        logger.info(f"Full-document retrieval: {len(chunks)} chunks for '{matched}'")
        return {
            "retrieved_chunks": chunks,
            "matched_document": matched,
            "matched_document_id": matched_document_id,
        }

    chunks = retrieve(query, user_id=user_id)
    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:80]}'")
    return {"retrieved_chunks": chunks, "matched_document": "", "matched_document_id": ""}


def router_node(state: AgentState) -> str:
    if state.get("matched_document"):
        logger.info(f"Routing to doc_summary (matched '{state['matched_document']}')")
        return "doc_summary"

    chunks = state["retrieved_chunks"]
    if chunks and chunks[0].score >= RAG_SCORE_THRESHOLD:
        logger.info(f"Routing to RAG (best score={chunks[0].score:.4f})")
        return "rag_answer"
    logger.info(
        f"Routing to web search "
        f"(best score={'none' if not chunks else f'{chunks[0].score:.4f}'})"
    )
    return "web_search"


def rag_answer_node(state: AgentState) -> dict:
    llm     = _build_llm()
    context = format_context(state["retrieved_chunks"])
    history = state.get("messages", [])

    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
    for msg in history[:-1]:          # exclude the current query (last item)
        messages.append(msg)
    messages.append(
        HumanMessage(
            content=(
                f"Document context:\n\n{context}\n\n"
                f"Question: {state['query']}"
            )
        )
    )

    response = llm.invoke(messages)
    answer = _apply_grounding_guardrail(response.content, state["retrieved_chunks"])
    sources  = [
        {
            "source":  c.source,
            "score":   c.score,
            "excerpt": c.content[:300],
            "page":    c.metadata.get("page", ""),
        }
        for c in state["retrieved_chunks"]
    ]
    return {
        "answer":          answer,
        "sources":         sources,
        "used_web_search": False,
        "token_usage":     _extract_token_usage(response),
        "messages":        [AIMessage(content=response.content)],
    }


def doc_summary_node(state: AgentState) -> dict:
    """Answer questions about a specific document using summaries + raw excerpts."""
    llm = _build_llm()
    chunks = state["retrieved_chunks"]
    docname = state["matched_document"]
    user_id = state["user_id"]
    document_id = state.get("matched_document_id", "")

    summary_chunks: list[RetrievedChunk] = []
    if document_id:
        summary_chunks = retrieve_document_summaries(
            document_id,
            user_id=user_id,
            node_types=["document_summary", "section_summary", "table_summary", "image_summary"],
        )

    document_summary = next(
        (c.content for c in summary_chunks if c.metadata.get("node_type") == "document_summary"),
        "",
    )
    section_summaries = [
        c for c in summary_chunks if c.metadata.get("node_type") == "section_summary"
    ][:6]
    visual_summaries = [
        c for c in summary_chunks if c.metadata.get("node_type") in {"image_summary", "table_summary"}
    ][:4]

    summary_blocks: list[str] = []
    if document_summary:
        summary_blocks.append(f"Stored document summary:\n{document_summary}")
    if section_summaries:
        section_lines = []
        for section in section_summaries:
            title = section.metadata.get("section_title") or "Untitled section"
            section_lines.append(f"- {title}: {section.content}")
        summary_blocks.append("Section summaries:\n" + "\n".join(section_lines))
    if visual_summaries:
        visual_lines = []
        for visual in visual_summaries:
            node_type = visual.metadata.get("node_type", "summary")
            label = "Image" if node_type == "image_summary" else "Table"
            page = visual.metadata.get("page")
            page_suffix = f" (page {page})" if page else ""
            visual_lines.append(f"- {label}{page_suffix}: {visual.content}")
        summary_blocks.append("Visual summaries:\n" + "\n".join(visual_lines))

    # Keep detailed raw excerpts for grounding follow-up questions.
    raw_parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page")
        if page:
            raw_parts.append(f"Page {page}\n{chunk.content}")
        else:
            raw_parts.append(chunk.content)
    raw_context = "\n\n---\n\n".join(raw_parts)

    max_chars = 45_000
    was_truncated = len(raw_context) > max_chars
    if was_truncated:
        raw_context = raw_context[:max_chars] + "\n\n[... document excerpt truncated for length ...]"

    messages = [
        SystemMessage(content=DOC_SUMMARY_PROMPT),
        HumanMessage(
            content=(
                f"Document: {docname}\n"
                f"Stored summaries available: {'yes' if summary_blocks else 'no'}\n"
                f"Raw chunks loaded: {len(chunks)}\n"
                f"Raw context truncated: {'yes' if was_truncated else 'no'}\n\n"
                f"{chr(10).join(summary_blocks) if summary_blocks else 'No stored summaries were available.'}\n\n"
                f"Grounding excerpts:\n\n{raw_context}\n\n"
                f"User question: {state['query']}"
            )
        ),
    ]

    response = llm.invoke(messages)
    answer = _apply_grounding_guardrail(response.content, chunks)
    sources  = [
        {
            "source":  docname,
            "score":   1.0,
            "excerpt": f"Full document ({len(chunks)} chunks)",
            "page":    "",
        }
    ]
    return {
        "answer":          answer,
        "sources":         sources,
        "used_web_search": False,
        "token_usage":     _extract_token_usage(response),
        "messages":        [AIMessage(content=response.content)],
    }


def web_search_node(state: AgentState) -> dict:
    logger.info(f"Web search for: '{state['query'][:80]}'")
    results = duckduckgo_search(state["query"])
    context = format_search_results(results)

    llm      = _build_llm()
    messages = [
        SystemMessage(content=WEB_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Web search results:\n\n{context}\n\nQuestion: {state['query']}"
        ),
    ]
    response = llm.invoke(messages)

    sources = [
        {"source": r.title, "url": r.url, "excerpt": r.snippet[:300]}
        for r in results
    ]
    return {
        "answer":          response.content,
        "web_results":     context,
        "sources":         sources,
        "used_web_search": True,
        "token_usage":     _extract_token_usage(response),
        "messages":        [AIMessage(content=response.content)],
    }


# ── Graph construction ─────────────────────────────────────────────────────────

def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve",    retrieve_node)
    graph.add_node("rag_answer",  rag_answer_node)
    graph.add_node("doc_summary", doc_summary_node)
    graph.add_node("web_search",  web_search_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        router_node,
        {
            "rag_answer":  "rag_answer",
            "doc_summary": "doc_summary",
            "web_search":  "web_search",
        },
    )
    graph.add_edge("rag_answer",  END)
    graph.add_edge("doc_summary", END)
    graph.add_edge("web_search",  END)
    return graph.compile()


_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = _build_graph()
    return _agent_graph


def reset_agent():
    """Force the agent graph to be rebuilt on next call (e.g. after doc deletion)."""
    global _agent_graph
    _agent_graph = None


# ── Public API ─────────────────────────────────────────────────────────────────

def run_agent(query: str, *, user_id: int, conversation_history: list[dict] | None = None) -> dict:
    """
    Run the agentic RAG pipeline.

    Returns:
        {"answer": str, "sources": list[dict], "used_web_search": bool}
    """
    history_messages = []
    for turn in (conversation_history or []):
        if turn["role"] == "user":
            history_messages.append(HumanMessage(content=turn["content"]))
        else:
            history_messages.append(AIMessage(content=turn["content"]))
    history_messages.append(HumanMessage(content=query))

    initial_state: AgentState = {
        "messages":          history_messages,
        "query":             query,
        "user_id":           user_id,
        "matched_document":  "",
        "matched_document_id": "",
        "retrieved_chunks":  [],
        "web_results":       "",
        "answer":            "",
        "sources":           [],
        "used_web_search":   False,
        "token_usage":       {},
    }

    final_state = get_agent().invoke(initial_state)
    return {
        "answer":          final_state["answer"],
        "sources":         final_state["sources"],
        "used_web_search": final_state["used_web_search"],
        "token_usage":     final_state.get("token_usage", {}),
    }
