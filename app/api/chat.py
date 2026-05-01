import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.rag_agent import run_agent
from app.auth.models import User
from app.auth.utils import get_current_user
from app.database import get_db
from app.memory.long_term import (
    delete_session,
    get_session_history,
    list_sessions,
    load_recent_history,
    load_session_recent_history,
    set_session_title,
    save_turn,
)
from loguru import logger

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    include_long_term_memory: bool = True


class SourceRef(BaseModel):
    source: str
    score: float | None = None
    url: str | None = None
    excerpt: str | None = None
    page: int | str | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    session_id: str
    used_web_search: bool
    latency_ms: float
    token_usage: TokenUsage | None = None


class SessionTitleUpdate(BaseModel):
    title: str


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session_id = payload.session_id or str(uuid.uuid4())
    start = time.perf_counter()

    # Short-term memory: active session window.
    session_history = load_session_recent_history(
        db,
        user_id=current_user.id,
        session_id=session_id,
        limit_turns=8,
    )

    # Long-term memory: recent turns from prior sessions.
    history: list[dict] = session_history
    if payload.include_long_term_memory:
        prior_history = load_recent_history(
            db,
            user_id=current_user.id,
            limit=12,
            exclude_session_id=session_id,
        )
        history = prior_history + session_history

    logger.info(
        f"[user={current_user.username}] [session={session_id}] "
        f"Query: '{payload.message[:80]}'"
    )

    try:
        result = run_agent(
            query=payload.message,
            user_id=current_user.id,
            conversation_history=history,
        )
    except ValueError as exc:
        # Configuration errors (missing API key, etc.) — return 422 with clear message
        logger.error(f"Configuration error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        err = str(exc)
        logger.error(f"Agent error: {err}")
        # Surface API auth errors clearly
        if "401" in err or "invalid_api_key" in err or "Invalid API Key" in err:
            raise HTTPException(
                status_code=422,
                detail="Groq API key is invalid or not set. "
                       "Add GROQ_API_KEY=gsk_... to your .env file and restart the server.",
            )
        raise HTTPException(status_code=500, detail=f"Agent error: {err}")

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    usage = result.get("token_usage") or {}
    logger.info(
        f"[session={session_id}] Answered in {latency_ms}ms | web={result['used_web_search']} | "
        f"tokens={usage.get('total_tokens', 'n/a')}"
    )

    # Persist turn to long-term memory
    sources_json = json.dumps(result["sources"])
    save_turn(
        db,
        user_id=current_user.id,
        session_id=session_id,
        user_message=payload.message,
        assistant_message=result["answer"],
        sources=sources_json,
    )

    sources = [
        SourceRef(
            source=s.get("source", ""),
            score=s.get("score"),
            url=s.get("url"),
            excerpt=s.get("excerpt"),
            page=s.get("page"),
        )
        for s in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        used_web_search=result["used_web_search"],
        latency_ms=latency_ms,
        token_usage=TokenUsage(**usage) if usage else None,
    )


@router.get("/history")
def get_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return recent conversation history for the current user."""
    history = load_recent_history(db, user_id=current_user.id, limit=limit)
    return {"history": history, "user": current_user.username}


@router.get("/sessions")
def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all chat sessions for the current user (most recent first)."""
    sessions = list_sessions(db, user_id=current_user.id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all messages for a specific chat session."""
    messages = get_session_history(
        db, user_id=current_user.id, session_id=session_id
    )
    return {"session_id": session_id, "messages": messages}


@router.delete("/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat session and all its messages."""
    count = delete_session(db, user_id=current_user.id, session_id=session_id)
    logger.info(f"Deleted session {session_id} ({count} messages)")
    return {"deleted": count, "session_id": session_id}


@router.patch("/sessions/{session_id}/title")
def rename_chat_session(
    session_id: str,
    payload: SessionTitleUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename a chat session for the current user."""
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if len(title) > 120:
        raise HTTPException(status_code=400, detail="Title is too long (max 120 chars)")

    updated = set_session_title(
        db,
        user_id=current_user.id,
        session_id=session_id,
        title=title,
    )
    return {"session_id": session_id, "title": updated}
