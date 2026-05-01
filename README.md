# Agentic Multimodal RAG System

Production-style, local-first RAG chatbot with:
- FastAPI backend + Streamlit frontend
- JWT auth + optional Google OAuth
- Multimodal document ingestion (PDF, DOCX images, tabular data)
- LangGraph agent routing (documents vs web search)
- ChromaDB vector storage (raw + summary collections)
- Voice-to-text via faster-whisper

## What Is Implemented

- User authentication:
  - Username/password login + registration
  - Google OAuth flow (`/auth/google/login` + `/auth/google/callback`)
- Document ingestion:
  - Supported formats: `PDF`, `DOCX`, `DOC`, `TXT`, `MD`, `XLSX`, `XLS`, `CSV`
  - User-scoped uploads under `data/documents/user_<id>/`
  - Full delete path removes vector nodes, local file, multimodal assets, and related chat history references
- Multimodal/structured ingestion:
  - PDF text extraction with PyMuPDF, optional hi-res unstructured OCR/table/image path
  - DOCX embedded image extraction + vision summaries
  - Excel/CSV row-aware parsing with date normalization variants and derived row facts
- Retrieval/agent:
  - Summary-first retrieval + parent raw-chunk expansion
  - Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - Structured date-intent filter for table queries (prevents admission/discharge mismatch)
  - Web fallback via DuckDuckGo
- Memory:
  - Active-session short-term window
  - Optional cross-session long-term context
- Observability:
  - Response latency + token usage surfaced to frontend
  - Grounding support guardrail appends low-confidence note when needed

## Architecture (Current)

```text
Browser (Streamlit UI, :8501)
  ├─ Auth (JWT / Google OAuth)
  ├─ Upload / Delete documents
  ├─ Chat + Voice input
  ▼
FastAPI API (:8000)
  ├─ /auth         -> SQLite users + JWT + Google OAuth callback redirect
  ├─ /documents    -> ingestion pipeline + Chroma raw/summary collections
  ├─ /chat         -> LangGraph agent + memory + source references
  └─ /voice        -> faster-whisper transcription

Ingestion pipeline
  ├─ PDF   -> PyMuPDF (fast path) or unstructured hi_res (OCR/tables/images)
  ├─ DOCX  -> text + embedded image summaries
  ├─ XLSX/XLS/CSV -> row-structured docs + date variants + derived facts
  └─ TXT/MD -> text loader

Retrieval pipeline
  1) Query expansion (multi-query)
  2) Summary similarity search (summary collection)
  3) Parent raw-chunk expansion (raw collection)
  4) Structured table/date intent filtering when applicable
  5) Cross-encoder rerank
  6) RAG answer or web-search fallback
```

## Tech Stack

- Backend: FastAPI, Uvicorn, SQLAlchemy
- Frontend: Streamlit
- LLM: Groq (`GROQ_MODEL`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Vector DB: ChromaDB (persisted to disk)
- OCR/Multimodal: `unstructured[pdf]`, Tesseract, Poppler, libmagic
- Voice: `faster-whisper` (CPU int8)

## Prerequisites

- Python 3.11+
- Groq API key
- System packages:
  - `ffmpeg` (voice transcription)
  - `tesseract` + `poppler` + `libmagic` (multimodal/OCR PDF path)
- Optional: Docker + Docker Compose

Install system dependencies:

```bash
# macOS
brew install ffmpeg tesseract poppler libmagic

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr poppler-utils libmagic1
```

## Environment Setup

```bash
cp .env.example .env
```

Minimum required:

```env
GROQ_API_KEY=gsk_...
SECRET_KEY=<long-random-secret>
```

Generate a strong secret:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Optional but recommended:
- `HF_TOKEN` for faster authenticated Hugging Face downloads
- Google OAuth vars (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `FRONTEND_URL`)

## Run Locally

### 1) Create venv and install

```bash
cd /Users/dinesh/rag/agentic-multimodal-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Start frontend (new terminal)

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```

### 4) Open URLs

- App: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Optional one-command local start

```bash
chmod +x start.sh
./start.sh
```

## Run With Docker

### 1) Prepare env

```bash
cd /Users/dinesh/rag/agentic-multimodal-system
cp .env.example .env
# Edit .env and set GROQ_API_KEY + SECRET_KEY (+ OAuth keys if needed)
```

### 2) Build and run

```bash
docker compose up --build
```

### 3) Open URLs

- App: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### 4) Useful Docker commands

```bash
docker compose logs -f api
docker compose logs -f frontend
docker compose down
```

## Google OAuth Notes (Local + Docker)

Google Cloud OAuth client must include:

- Authorized redirect URI:
  - `http://localhost:8000/auth/google/callback`
- Authorized JavaScript origins:
  - `http://localhost:8000`
  - `http://localhost:8501`

Docker detail:
- Frontend talks to backend internally using `API_BASE=http://api:8000`
- Browser OAuth link uses `BROWSER_API_BASE=http://localhost:8000`

## Usage Flow

1. Register/login (or Google login).
2. Upload documents from the Upload page.
3. Ask questions in chat (text or mic).
4. Review source references, latency, and token usage.
5. Delete documents when needed (cleanup is end-to-end).

## API Endpoints

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/register` | No | Register user |
| POST | `/auth/login` | No | Username/password login |
| GET | `/auth/google/login` | No | Start Google OAuth |
| GET | `/auth/google/callback` | No | Google OAuth callback |
| POST | `/chat` | Yes | Ask question, get answer + sources + token usage |
| GET | `/chat/history` | Yes | Recent cross-session message history |
| GET | `/chat/sessions` | Yes | List chat sessions |
| GET | `/chat/sessions/{session_id}` | Yes | Get messages in one session |
| PATCH | `/chat/sessions/{session_id}/title` | Yes | Rename session |
| DELETE | `/chat/sessions/{session_id}` | Yes | Delete session |
| POST | `/documents/upload` | Yes | Upload + ingest document |
| GET | `/documents` | Yes | List indexed sources |
| DELETE | `/documents/{filename}` | Yes | Full document cleanup |
| POST | `/voice/transcribe` | Yes | Transcribe audio (`wav/mp3/webm/ogg/m4a/flac`) |
| GET | `/health` | No | Health check |

## Project Structure

```text
app/
  agents/
    rag_agent.py
    web_search.py
  api/
    chat.py
    documents.py
    voice.py
  auth/
    models.py
    routes.py
    schemas.py
    utils.py
  memory/
    long_term.py
    short_term.py
  rag/
    embeddings.py
    ingestion.py
    reranker.py
    retriever.py
    vectorstore.py
  config.py
  database.py
  hf_runtime.py
  main.py
frontend/
  streamlit_app.py
tests/
Dockerfile
docker-compose.yml
start.sh
```

## Testing

Run all tests:

```bash
pytest -q
```

Current suite includes API, ingestion, memory, retrieval, and runtime config tests.

## Troubleshooting

- Upload says server unreachable:
  - Ensure API is running on `:8000`.
  - In Docker, check `docker compose logs -f api`.
- First-time model downloads are slow:
  - Expected for embeddings/reranker/whisper/model caches.
- Vision model error/deprecation:
  - Use `VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct` (already set in `.env.example`).
- Groq rate limit (`429`):
  - Wait for quota reset or use a key with available tokens.
- Google OAuth fails:
  - Verify exact redirect URI/origins in Google Console.

