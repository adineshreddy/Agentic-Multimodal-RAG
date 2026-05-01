"""
Tests for retriever, agent routing logic, auth utilities, and memory.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_retrieve_summary_first_expands_to_raw_and_appends_visual_context():
    from app.rag.retriever import retrieve

    image_summary = Document(
        page_content="Bar chart shows Q4 revenue peak.",
        metadata={
            "source": "report.pdf",
            "document_id": "doc-1",
            "node_type": "image_summary",
            "summary_index": 1,
            "page": 4,
        },
    )
    raw_chunk = Document(
        page_content="Revenue grew 22% in Q4 with enterprise expansion.",
        metadata={
            "source": "report.pdf",
            "document_id": "doc-1",
            "node_type": "raw_chunk",
            "chunk_index": 0,
            "page": 4,
        },
    )

    with patch("app.rag.retriever._generate_query_variants", return_value=["q"]), \
         patch("app.rag.retriever.summary_similarity_search", return_value=[(image_summary, 0.88)]), \
         patch("app.rag.retriever.get_chunks_by_document_id", return_value=[raw_chunk]), \
         patch("app.rag.retriever.similarity_search", return_value=[]), \
         patch(
             "app.rag.retriever.rerank",
             side_effect=lambda _q, passages, top_k=5: [
                 {**passages[0], "rerank_score": 1.25}
             ],
         ):
        chunks = retrieve("How did Q4 perform?", user_id=1, k=3)

    assert len(chunks) == 2
    assert chunks[0].metadata["node_type"] == "raw_chunk"
    assert chunks[1].metadata["node_type"] == "image_summary"


def test_retrieve_falls_back_to_summary_when_no_raw_available():
    from app.rag.retriever import retrieve

    doc_summary = Document(
        page_content="The paper introduces two methods for OCR post-processing.",
        metadata={
            "source": "paper.pdf",
            "document_id": "doc-2",
            "node_type": "document_summary",
            "summary_index": 0,
        },
    )

    with patch("app.rag.retriever._generate_query_variants", return_value=["q"]), \
         patch("app.rag.retriever.summary_similarity_search", return_value=[(doc_summary, 0.79)]), \
         patch("app.rag.retriever.get_chunks_by_document_id", return_value=[]), \
         patch("app.rag.retriever.similarity_search", return_value=[]), \
         patch(
             "app.rag.retriever.rerank",
             side_effect=lambda _q, passages, top_k=5: [
                 {**passages[0], "rerank_score": 0.91}
             ],
         ):
        chunks = retrieve("Summarize the paper", user_id=2, k=2)

    assert len(chunks) == 1
    assert chunks[0].metadata["node_type"] == "document_summary"


def test_structured_date_intent_filter_prefers_admission_line_match():
    from app.rag.retriever import _apply_structured_date_intent_filter

    query = "what is the name of patient who was admitted for Arthritis on 7/14/20?"
    candidates = [
        {
            "content": (
                "Medical Condition: Arthritis\n"
                "Facts: Patient Alice was admitted for Arthritis on 2020-07-14."
            ),
            "source": "healthcare.xlsx",
            "score": 0.0,
            "metadata": {"content_type": "table_row"},
        },
        {
            "content": (
                "Medical Condition: Arthritis\n"
                "Facts: Patient Bob was admitted for Arthritis on 2020-06-23. "
                "Patient Bob was discharged on 2020-07-14."
            ),
            "source": "healthcare.xlsx",
            "score": 0.0,
            "metadata": {"content_type": "table_row"},
        },
    ]

    filtered = _apply_structured_date_intent_filter(query, candidates)
    assert len(filtered) == 1
    assert "Alice" in filtered[0]["content"]


def test_structured_date_intent_filter_returns_notice_when_no_exact_match():
    from app.rag.retriever import _apply_structured_date_intent_filter

    query = "who was admitted for Arthritis on 7/14/20?"
    candidates = [
        {
            "content": (
                "Medical Condition: Arthritis\n"
                "Facts: Patient Bob was admitted for Arthritis on 2020-06-23. "
                "Patient Bob was discharged on 2020-07-14."
            ),
            "source": "healthcare.xlsx",
            "score": 0.0,
            "metadata": {"content_type": "table_row"},
        }
    ]

    filtered = _apply_structured_date_intent_filter(query, candidates)
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["content_type"] == "structured_notice"
    assert "no exact matching row was found" in filtered[0]["content"].lower()


def test_format_context_is_clean_and_hides_internal_scores():
    from app.rag.retriever import RetrievedChunk, format_context

    chunks = [
        RetrievedChunk(
            content="AI learns from data.",
            source="ai.txt",
            score=0.9,
            metadata={"page": 2},
        )
    ]
    context = format_context(chunks)
    assert "Source: ai.txt (page 2)" in context
    assert "relevance:" not in context
    assert "[1]" not in context


def test_retrieve_full_document_prefers_document_id_path():
    from app.rag.retriever import retrieve_full_document

    by_id_docs = [
        Document(
            page_content="Chunk from doc id.",
            metadata={"source": "report.pdf", "chunk_index": 0},
        )
    ]
    with patch("app.rag.retriever.get_chunks_by_document_id", return_value=by_id_docs) as by_id, \
         patch("app.rag.retriever.get_chunks_by_source", return_value=[]) as by_source:
        chunks = retrieve_full_document(
            "report.pdf",
            user_id=10,
            document_id="doc-10",
        )

    assert len(chunks) == 1
    assert chunks[0].source == "report.pdf"
    by_id.assert_called_once()
    by_source.assert_not_called()


def test_agent_routes_to_doc_summary_when_document_matched():
    from app.agents.rag_agent import router_node

    state = {
        "messages": [],
        "query": "Summarize report.pdf",
        "user_id": 1,
        "matched_document": "report.pdf",
        "matched_document_id": "doc-1",
        "retrieved_chunks": [],
        "web_results": "",
        "answer": "",
        "sources": [],
        "used_web_search": False,
    }
    assert router_node(state) == "doc_summary"


def test_agent_routes_to_rag_when_chunks_found():
    from app.agents.rag_agent import router_node

    state = {
        "messages": [],
        "query": "What is RAG?",
        "user_id": 1,
        "matched_document": "",
        "matched_document_id": "",
        "retrieved_chunks": [MagicMock(score=0.5)],
        "web_results": "",
        "answer": "",
        "sources": [],
        "used_web_search": False,
    }
    assert router_node(state) == "rag_answer"


def test_agent_routes_to_web_when_no_chunks():
    from app.agents.rag_agent import router_node

    state = {
        "messages": [],
        "query": "Latest news about Mars?",
        "user_id": 1,
        "matched_document": "",
        "matched_document_id": "",
        "retrieved_chunks": [],
        "web_results": "",
        "answer": "",
        "sources": [],
        "used_web_search": False,
    }
    assert router_node(state) == "web_search"


def test_password_hashing_and_verification():
    from app.auth.utils import hash_password, verify_password

    plain = "supersecret123"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_jwt_encode_decode():
    from app.auth.utils import create_access_token, decode_token

    payload = {"sub": "alice", "user_id": 42}
    token = create_access_token(payload)
    decoded = decode_token(token)
    assert decoded["sub"] == "alice"
    assert decoded["user_id"] == 42


def test_short_term_memory_creation():
    from app.memory.short_term import create_session_memory

    mem = create_session_memory(k=5)
    mem.add_turn("Hello", "Hi there!")
    history = mem.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there!"


def test_long_term_memory_save_and_load():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.memory.long_term import load_recent_history, save_turn

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    db = session()

    save_turn(
        db,
        user_id=1,
        session_id="sess-abc",
        user_message="What is ML?",
        assistant_message="ML is machine learning.",
        sources="[]",
    )

    history = load_recent_history(db, user_id=1, limit=10)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert "ML" in history[0]["content"]
    assert history[1]["role"] == "assistant"
    db.close()
