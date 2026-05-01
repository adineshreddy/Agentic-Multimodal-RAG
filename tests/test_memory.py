"""
Tests for long-term memory behavior.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.memory.long_term import (
    ChatSessionMeta,
    ConversationHistory,
    delete_history_for_source,
    load_recent_history,
    load_session_recent_history,
)


def _build_test_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def test_delete_history_for_source_removes_full_turn_and_session_meta():
    db = _build_test_session()
    try:
        # Session s1 has two turns; only first one references target.pdf.
        db.add_all(
            [
                ConversationHistory(user_id=1, session_id="s1", role="user", content="q1"),
                ConversationHistory(
                    user_id=1,
                    session_id="s1",
                    role="assistant",
                    content="a1",
                    sources='[{"source": "target.pdf"}]',
                ),
                ConversationHistory(user_id=1, session_id="s1", role="user", content="q2"),
                ConversationHistory(
                    user_id=1,
                    session_id="s1",
                    role="assistant",
                    content="a2",
                    sources='[{"source": "other.pdf"}]',
                ),
            ]
        )

        # Session s2 has one turn with legacy non-JSON sources.
        db.add_all(
            [
                ConversationHistory(user_id=1, session_id="s2", role="user", content="q3"),
                ConversationHistory(
                    user_id=1,
                    session_id="s2",
                    role="assistant",
                    content="a3",
                    sources="legacy source list: target.pdf",
                ),
            ]
        )

        # Another user's matching source should be untouched.
        db.add_all(
            [
                ConversationHistory(user_id=2, session_id="u2", role="user", content="u2-q1"),
                ConversationHistory(
                    user_id=2,
                    session_id="u2",
                    role="assistant",
                    content="u2-a1",
                    sources='[{"source": "target.pdf"}]',
                ),
            ]
        )

        db.add_all(
            [
                ChatSessionMeta(user_id=1, session_id="s1", title="session one"),
                ChatSessionMeta(user_id=1, session_id="s2", title="session two"),
            ]
        )
        db.commit()

        removed = delete_history_for_source(db, user_id=1, source_filename="target.pdf")
        assert removed == 4

        remaining_user1 = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.user_id == 1)
            .order_by(ConversationHistory.id.asc())
            .all()
        )
        assert [(r.session_id, r.role, r.content) for r in remaining_user1] == [
            ("s1", "user", "q2"),
            ("s1", "assistant", "a2"),
        ]

        remaining_user2 = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.user_id == 2)
            .order_by(ConversationHistory.id.asc())
            .all()
        )
        assert len(remaining_user2) == 2

        meta_rows = (
            db.query(ChatSessionMeta)
            .filter(ChatSessionMeta.user_id == 1)
            .order_by(ChatSessionMeta.session_id.asc())
            .all()
        )
        assert [m.session_id for m in meta_rows] == ["s1"]
    finally:
        db.close()


def test_load_recent_history_can_exclude_active_session():
    db = _build_test_session()
    try:
        db.add_all(
            [
                ConversationHistory(user_id=1, session_id="s1", role="user", content="u1"),
                ConversationHistory(user_id=1, session_id="s1", role="assistant", content="a1"),
                ConversationHistory(user_id=1, session_id="s2", role="user", content="u2"),
                ConversationHistory(user_id=1, session_id="s2", role="assistant", content="a2"),
            ]
        )
        db.commit()

        all_history = load_recent_history(db, user_id=1, limit=10)
        assert len(all_history) == 4

        prior_only = load_recent_history(db, user_id=1, limit=10, exclude_session_id="s2")
        assert len(prior_only) == 2
        assert [m["content"] for m in prior_only] == ["u1", "a1"]
    finally:
        db.close()


def test_load_session_recent_history_returns_short_term_window():
    db = _build_test_session()
    try:
        # 3 turns in session s1 = 6 messages.
        for idx in range(1, 4):
            db.add(ConversationHistory(user_id=1, session_id="s1", role="user", content=f"u{idx}"))
            db.add(ConversationHistory(user_id=1, session_id="s1", role="assistant", content=f"a{idx}"))
        # Noise from another session should be excluded.
        db.add(ConversationHistory(user_id=1, session_id="s2", role="user", content="other"))
        db.commit()

        window = load_session_recent_history(db, user_id=1, session_id="s1", limit_turns=2)
        assert len(window) == 4
        assert [m["content"] for m in window] == ["u2", "a2", "u3", "a3"]
    finally:
        db.close()
