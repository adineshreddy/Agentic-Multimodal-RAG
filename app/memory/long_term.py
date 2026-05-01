"""
Long-term memory: SQLite-backed conversation history.

Persists every conversation turn so context from previous sessions can be
surfaced to the LLM on next login.
"""
import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from app.database import Base


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(10), nullable=False)      # "user" | "assistant"
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)          # JSON-encoded source list
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSessionMeta(Base):
    __tablename__ = "chat_session_meta"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    title = Column(String(120), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def save_turn(
    db: Session,
    *,
    user_id: int,
    session_id: str,
    user_message: str,
    assistant_message: str,
    sources: str | None = None,
) -> None:
    """Persist a single user→assistant turn to SQLite."""
    db.add(ConversationHistory(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=user_message,
    ))
    db.add(ConversationHistory(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        sources=sources,
    ))
    db.commit()


def load_recent_history(
    db: Session,
    *,
    user_id: int,
    limit: int = 20,
    exclude_session_id: str | None = None,
) -> list[dict]:
    """
    Return the most recent conversation turns for a user across all sessions.
    Used to inject prior-session context into the LLM system prompt.
    """
    query = (
        db.query(ConversationHistory)
        .filter(ConversationHistory.user_id == user_id)
    )
    if exclude_session_id:
        query = query.filter(ConversationHistory.session_id != exclude_session_id)

    rows = (
        query
        .order_by(ConversationHistory.id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()  # chronological order
    return [{"role": r.role, "content": r.content} for r in rows]


def load_session_recent_history(
    db: Session,
    *,
    user_id: int,
    session_id: str,
    limit_turns: int = 8,
) -> list[dict]:
    """
    Return recent history for the active chat session only.

    This is the short-term memory window used to keep continuity inside the
    current conversation.
    """
    message_limit = max(2, limit_turns * 2)
    rows = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.user_id == user_id,
            ConversationHistory.session_id == session_id,
        )
        .order_by(ConversationHistory.id.desc())
        .limit(message_limit)
        .all()
    )
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]


def delete_history_for_source(db: Session, *, user_id: int, source_filename: str) -> int:
    """
    Delete conversation turns that reference a source filename.

    Match is done against assistant rows' `sources`, then both the assistant row
    and its paired user row are removed so no partial turns are left behind.
    Returns total rows deleted.
    """
    rows = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.user_id == user_id,
        )
        .order_by(ConversationHistory.session_id.asc(), ConversationHistory.id.asc())
        .all()
    )

    def _row_references_source(row_sources: str | None) -> bool:
        if not row_sources:
            return False
        try:
            parsed = json.loads(row_sources)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("source") == source_filename:
                        return True
                    if isinstance(item, str) and item == source_filename:
                        return True
        except Exception:
            # Backward-compat fallback for unexpected legacy source payloads.
            return source_filename in row_sources
        return False

    ids_to_delete: set[int] = set()
    last_user_id_by_session: dict[str, int] = {}
    for row in rows:
        if row.role == "user":
            last_user_id_by_session[row.session_id] = row.id
            continue
        if row.role != "assistant" or not _row_references_source(row.sources):
            continue

        ids_to_delete.add(row.id)
        paired_user_id = last_user_id_by_session.get(row.session_id)
        if paired_user_id is not None:
            ids_to_delete.add(paired_user_id)

    if not ids_to_delete:
        return 0

    rows_to_delete = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.user_id == user_id,
            ConversationHistory.id.in_(ids_to_delete),
        )
        .all()
    )
    for row in rows_to_delete:
        db.delete(row)
    db.flush()

    # Remove metadata rows for sessions that no longer have any messages.
    session_ids = {row.session_id for row in rows_to_delete}
    for session_id in session_ids:
        has_history = (
            db.query(ConversationHistory.id)
            .filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.session_id == session_id,
            )
            .first()
            is not None
        )
        if not has_history:
            db.query(ChatSessionMeta).filter(
                ChatSessionMeta.user_id == user_id,
                ChatSessionMeta.session_id == session_id,
            ).delete()

    db.commit()
    return len(rows_to_delete)


def get_session_history(
    db: Session,
    *,
    user_id: int,
    session_id: str,
) -> list[dict]:
    """Return all turns for a specific session."""
    rows = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.user_id == user_id,
            ConversationHistory.session_id == session_id,
        )
        .order_by(ConversationHistory.created_at.asc())
        .all()
    )
    return [
        {"role": r.role, "content": r.content, "sources": r.sources}
        for r in rows
    ]


def list_sessions(db: Session, *, user_id: int) -> list[dict]:
    """
    Return a list of chat sessions for a user, most recent first.
    Each entry has session_id, title (first user message), and last_active.
    """
    from sqlalchemy import func

    # Get distinct session_ids with their last activity time
    session_rows = (
        db.query(
            ConversationHistory.session_id,
            func.max(ConversationHistory.created_at).label("last_active"),
        )
        .filter(ConversationHistory.user_id == user_id)
        .group_by(ConversationHistory.session_id)
        .order_by(func.max(ConversationHistory.created_at).desc())
        .all()
    )

    session_ids = [row.session_id for row in session_rows]
    meta_rows = (
        db.query(ChatSessionMeta.session_id, ChatSessionMeta.title)
        .filter(
            ChatSessionMeta.user_id == user_id,
            ChatSessionMeta.session_id.in_(session_ids),
        )
        .all()
    ) if session_ids else []
    custom_titles = {sid: title for sid, title in meta_rows}

    sessions = []
    for row in session_rows:
        # Get the first user message in this session for the title
        first_msg = (
            db.query(ConversationHistory.content)
            .filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.session_id == row.session_id,
                ConversationHistory.role == "user",
            )
            .order_by(ConversationHistory.created_at.asc())
            .first()
        )
        content = first_msg[0] if first_msg else "New chat"
        title = custom_titles.get(row.session_id)
        if not title:
            title = content.strip().split("\n", 1)[0][:40]
            if len(content) > 40:
                title += "…"
        sessions.append({
            "session_id": row.session_id,
            "title": title,
            "last_active": row.last_active.isoformat() if row.last_active else "",
        })
    return sessions


def delete_session(db: Session, *, user_id: int, session_id: str) -> int:
    """Delete all conversation turns for a specific session. Returns deleted count."""
    rows = (
        db.query(ConversationHistory)
        .filter(
            ConversationHistory.user_id == user_id,
            ConversationHistory.session_id == session_id,
        )
        .all()
    )
    count = len(rows)
    for row in rows:
        db.delete(row)
    db.query(ChatSessionMeta).filter(
        ChatSessionMeta.user_id == user_id,
        ChatSessionMeta.session_id == session_id,
    ).delete()
    db.commit()
    return count


def set_session_title(
    db: Session,
    *,
    user_id: int,
    session_id: str,
    title: str,
) -> str:
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Title cannot be empty")
    if len(clean_title) > 120:
        clean_title = clean_title[:120]

    row = (
        db.query(ChatSessionMeta)
        .filter(
            ChatSessionMeta.user_id == user_id,
            ChatSessionMeta.session_id == session_id,
        )
        .first()
    )
    if row:
        row.title = clean_title
    else:
        db.add(ChatSessionMeta(
            user_id=user_id,
            session_id=session_id,
            title=clean_title,
        ))
    db.commit()
    return clean_title
