"""
Tests for FastAPI routes using TestClient.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app

# ── In-memory DB override ──────────────────────────────────────────────────────
TEST_DB_URL = "sqlite://"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


Base.metadata.create_all(bind=engine)
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _register_and_login(username="testuser", password="testpass123"):
    client.post(
        "/auth/register",
        json={"username": username, "email": f"{username}@test.com", "password": password},
    )
    resp = client.post("/auth/login", json={"username": username, "password": password})
    return resp.json()["access_token"]


# ── Auth tests ─────────────────────────────────────────────────────────────────

def test_register_success():
    resp = client.post(
        "/auth/register",
        json={"username": "alice", "email": "alice@test.com", "password": "password123"},
    )
    assert resp.status_code == 201
    assert resp.json()["username"] == "alice"


def test_register_duplicate_username():
    client.post(
        "/auth/register",
        json={"username": "bob", "email": "bob@test.com", "password": "password123"},
    )
    resp = client.post(
        "/auth/register",
        json={"username": "bob", "email": "bob2@test.com", "password": "password123"},
    )
    assert resp.status_code == 400
    assert "taken" in resp.json()["detail"].lower()


def test_login_success():
    client.post(
        "/auth/register",
        json={"username": "carol", "email": "carol@test.com", "password": "pass1234"},
    )
    resp = client.post("/auth/login", json={"username": "carol", "password": "pass1234"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password():
    client.post(
        "/auth/register",
        json={"username": "dave", "email": "dave@test.com", "password": "rightpass"},
    )
    resp = client.post("/auth/login", json={"username": "dave", "password": "wrongpass"})
    assert resp.status_code == 401


# ── Documents tests ────────────────────────────────────────────────────────────

def test_list_documents_requires_auth():
    resp = client.get("/documents")
    assert resp.status_code == 401


def test_list_documents_authenticated():
    token = _register_and_login("docuser")
    with patch("app.api.documents.list_sources", return_value=["file1.txt", "file2.pdf"]):
        resp = client.get("/documents", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["total"] == 2


def test_upload_unsupported_file_type():
    token = _register_and_login("uploaduser")
    resp = client.post(
        "/documents/upload",
        files={"file": ("test.xyz", b"binary content", "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 415


def test_delete_document_performs_full_cleanup():
    token = _register_and_login("deleteuser")
    with (
        patch("app.api.documents.delete_source", return_value=3) as delete_source_mock,
        patch("app.api.documents.remove_multimodal_assets", return_value=1) as remove_assets_mock,
        patch("app.api.documents.delete_history_for_source", return_value=4) as delete_history_mock,
        patch("app.api.documents.clear_vectorstore_caches") as clear_cache_mock,
        patch("app.api.documents.reset_agent") as reset_agent_mock,
        patch("app.api.documents.os.path.exists", return_value=True),
        patch("app.api.documents.os.remove") as remove_file_mock,
    ):
        resp = client.delete(
            "/documents/sample.pdf",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_removed"] == 3
    assert body["history_rows_removed"] == 4
    assert body["asset_dirs_removed"] == 1

    delete_source_mock.assert_called_once()
    remove_assets_mock.assert_called_once()
    delete_history_mock.assert_called_once()
    clear_cache_mock.assert_called_once()
    reset_agent_mock.assert_called_once()
    remove_file_mock.assert_called_once()


def test_chat_includes_token_usage_from_agent():
    token = _register_and_login("chatusage")

    fake_result = {
        "answer": "Here is the answer.",
        "sources": [{"source": "doc.pdf", "score": 0.91, "excerpt": "Snippet", "page": 2}],
        "used_web_search": False,
        "token_usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }

    with patch("app.api.chat.run_agent", return_value=fake_result):
        resp = client.post(
            "/chat",
            json={"message": "Summarize the document", "session_id": "sess-chat-1"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Here is the answer."
    assert body["token_usage"]["total_tokens"] == 70
    assert body["sources"][0]["page"] == 2


def test_chat_uses_active_session_short_term_and_optional_long_term():
    token = _register_and_login("chatmemory")

    captured = {}

    def _fake_run_agent(*, query, user_id, conversation_history):
        captured["query"] = query
        captured["user_id"] = user_id
        captured["history"] = conversation_history
        return {
            "answer": "ok",
            "sources": [],
            "used_web_search": False,
            "token_usage": {},
        }

    with (
        patch("app.api.chat.load_session_recent_history", return_value=[{"role": "user", "content": "s-msg"}]) as short_mem,
        patch("app.api.chat.load_recent_history", return_value=[{"role": "assistant", "content": "lt-msg"}]) as long_mem,
        patch("app.api.chat.run_agent", side_effect=_fake_run_agent),
    ):
        resp = client.post(
            "/chat",
            json={
                "message": "hello",
                "session_id": "sess-chat-2",
                "include_long_term_memory": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    short_mem.assert_called_once()
    long_mem.assert_called_once()
    assert captured["history"] == [
        {"role": "assistant", "content": "lt-msg"},
        {"role": "user", "content": "s-msg"},
    ]

    # Now disable long-term memory and ensure only active session context is passed.
    with (
        patch("app.api.chat.load_session_recent_history", return_value=[{"role": "assistant", "content": "active"}]),
        patch("app.api.chat.load_recent_history") as long_mem_disabled,
        patch("app.api.chat.run_agent", side_effect=_fake_run_agent),
    ):
        resp = client.post(
            "/chat",
            json={
                "message": "hello again",
                "session_id": "sess-chat-2",
                "include_long_term_memory": False,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    long_mem_disabled.assert_not_called()
    assert captured["history"] == [{"role": "assistant", "content": "active"}]


# ── Health endpoint ────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "RAG" in resp.json()["service"]
