# Agentic Multimodal RAG System — Architecture

A production-style, local-first RAG chatbot. This document is the demo script: every section maps to something concrete in the code.

---

## 1. Low-Level Architecture (Block Diagram)

Every box is a real module. Arrows are real function calls.

```text
┌──────────────────────────────── CLIENT ────────────────────────────────┐
│  Streamlit UI  (frontend/streamlit_app.py)                             │
│  ┌────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────────────┐   │
│  │ Login  │  │ Upload/Delete│  │ Chat box  │  │ Mic recorder       │   │
│  └───┬────┘  └──────┬───────┘  └─────┬─────┘  └──────────┬─────────┘   │
└──────┼─────────────┼────────────────┼───────────────────┼──────────────┘
       │ JWT token   │ multipart      │ JSON              │ audio blob
       ▼             ▼                ▼                   ▼
┌──────────────────────────── FASTAPI BACKEND ───────────────────────────┐
│                       app/main.py  +  CORS + latency middleware        │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ auth/routes  │  │ api/documents│  │ api/chat     │  │ api/voice  │  │
│  │ /register    │  │ /upload      │  │ POST /chat   │  │ /transcribe│  │
│  │ /login       │  │ /list        │  │ /sessions    │  │            │  │
│  │ /google/*    │  │ /delete      │  │ /history     │  │            │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                 │                 │                │         │
│         ▼                 ▼                 ▼                ▼         │
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────┐  ┌──────┐  │
│  │ auth/utils   │  │ rag/ingestion      │  │ agents/      │  │faster│  │
│  │ • bcrypt     │  │ • _load_pdf        │  │ rag_agent    │  │whisp │  │
│  │ • JWT encode │  │ • _load_docx       │  │ (LangGraph)  │  │ (CPU │  │
│  │ • get_user   │  │ • _load_excel/csv  │  │              │  │ int8)│  │
│  └──────────────┘  │ • _load_text       │  └──────┬───────┘  └──────┘  │
│                    │ • Chunking         │         │                    │
│                    │ • Summary builder  │         │                    │
│                    │ • Vision summary   │         │                    │
│                    └─────────┬──────────┘         │                    │
└──────────────────────────────┼────────────────────┼────────────────────┘
                               │                    │
                ┌──────────────┴────────┐  ┌────────┴──────────────────────┐
                ▼                       ▼  ▼                               │
┌─────────────────────────────────────────────────────────────────────────┐│
│                     RAG PIPELINE  (app/rag/)                            ││
│                                                                         ││
│  rag/embeddings.py      rag/vectorstore.py        rag/retriever.py      ││
│  ┌────────────────┐     ┌─────────────────────┐   ┌──────────────────┐  ││
│  │ MiniLM L6 v2   │ ──► │ ChromaDB            │   │ retrieve()       │  ││
│  │ 384-dim        │     │  ┌───────────────┐  │   │  1 multi-query   │  ││
│  │ HF local       │     │  │ raw collection│  │   │  2 summary search│  ││
│  └────────────────┘     │  └───────────────┘  │   │  3 parent expand │  ││
│                         │  ┌───────────────┐  │◄──┤  4 date filter   │  ││
│                         │  │summary collec.│  │   │  5 cross-encoder │  ││
│                         │  └───────────────┘  │   │  6 grounding     │  ││
│                         │  metadata filter:   │   └────────┬─────────┘  ││
│                         │   user_id, doc_id,  │            │            ││
│                         │   node_type         │            ▼            ││
│                         └─────────────────────┘   rag/reranker.py       ││
│                                                   ms-marco MiniLM       ││
└─────────────────────────────────────────────────────────────────────────┘│
                                                            │              │
                            ┌───────────────────────────────┘              │
                            ▼                                              │
┌─────────────────────────────────────────────────────────────────────────┐│
│                    AGENT GRAPH  (app/agents/rag_agent.py)               ││
│                                                                         ││
│           ┌──────────────┐                                              ││
│           │ retrieve_node│  resolve_document_name + retrieve()          ││
│           └──────┬───────┘                                              ││
│                  ▼                                                      ││
│           ┌──────────────┐    matched doc?  →  doc_summary_node         ││
│           │ router_node  │    score ≥ -3.0  →  rag_answer_node          ││
│           └──┬───────┬───┘    else          →  web_search_node          ││
│              │       │       │                                          ││
│              ▼       ▼       ▼                                          ││
│   doc_summary  rag_answer  web_search ─► duckduckgo_search()            ││
│        │           │           │                                        ││
│        └───────────┴───────────┘                                        ││
│                    ▼                                                    ││
│             grounding guardrail                                         ││
│             (_grounding_support_ratio)                                  ││
│                    │                                                    ││
│                    ▼                                                    ││
│             ChatGroq.invoke()  →  answer + token_usage                  ││
└─────────────────────────────────────────────────────────────────────────┘│
                            │                                              │
                            ▼                                              │
┌─────────────────────────────────────────────────────────────────────────┐│
│                   MEMORY  (app/memory/)                                 ││
│                                                                         ││
│  short_term.py                       long_term.py (SQLite)              ││
│  ┌─────────────────────┐             ┌──────────────────────────────┐   ││
│  │ Sliding window      │             │ conversation_history table   │   ││
│  │ (in active session) │             │ chat_session_meta table      │   ││
│  └─────────────────────┘             │ • save_turn                  │   ││
│                                      │ • load_session_recent_history│   ││
│                                      │ • load_recent_history        │   ││
│                                      │ • delete_history_for_source  │   ││
│                                      └──────────────────────────────┘   ││
└─────────────────────────────────────────────────────────────────────────┘│
                                                                          │
┌──────────── PERSISTENCE ────────────┐  ┌──────── EXTERNAL SERVICES ─────┘
│  vectorstore/    ChromaDB files     │  │  Groq API (LLM + vision)
│  app.db          SQLite             │  │  Google OAuth
│  data/documents/user_<id>/   files  │  │  DuckDuckGo (web fallback)
│  └ .multimodal_assets/<doc_id>/imgs │  │  HuggingFace Hub (model dl)
└─────────────────────────────────────┘  └─────────────────────────────────
```

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Uvicorn | Async, OpenAPI-native |
| Frontend | Streamlit | Single-file UI, fast iteration |
| LLM | Groq `llama-3.3-70b-versatile` | High tokens/sec |
| Vision LLM | Groq `llama-4-scout-17b-16e-instruct` | Image summarization |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, CPU, 384-dim |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | High-precision reranking |
| Vector DB | ChromaDB (persisted) | Embedded, no extra service |
| RDBMS | SQLite + SQLAlchemy | Users, sessions, history |
| Voice | `faster-whisper` (CPU int8) | Local STT |
| Multimodal | `unstructured[pdf]`, Tesseract, Poppler | OCR, tables, images |
| Auth | JWT + Google OAuth | Standards |
| Orchestration | LangGraph | Conditional graph routing |

---

## 3. Chunking — How Documents Become Searchable

Code: `app/rag/ingestion.py`

### 3.1 Why chunk at all?

LLM context windows are limited and embedding quality degrades on very long passages. We need **small, semantically coherent passages** that an embedding model can vectorize and the LLM can reason over.

### 3.2 Per-format chunking strategy

We use **format-aware chunking** — one strategy does not fit all data shapes.

| Format | Loader | Chunk granularity | Notes |
|---|---|---|---|
| PDF (text) | PyMuPDF (`fitz`) | one Document per page → recursive split | fast text path |
| PDF (rich) | `unstructured` `hi_res` | "by_title" chunks (max 4000 chars) | preserves tables + image blocks |
| DOCX | `Docx2txtLoader` + `zipfile` for images | recursive split | embedded images become `image_summary` nodes |
| TXT/MD | `TextLoader` | recursive split | |
| XLSX | `openpyxl` row iteration | **one Document per row** | each row becomes "Column: Value" sentences |
| CSV | `csv.reader` | **one Document per row** | dialect auto-detected |

### 3.3 The text chunker

For prose-style content (PDF text, DOCX, TXT), a `RecursiveCharacterTextSplitter` is used:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # ≥800 chars (overrides smaller .env default)
    chunk_overlap=150,     # ≥150 chars overlap so sentences aren't cut
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Why these numbers?
- **800 chars** ≈ a couple of paragraphs — long enough to carry meaning, short enough for the embedder to encode tightly.
- **150 char overlap** preserves cross-chunk context (e.g. a sentence that straddles a chunk boundary still appears in both).
- **Separator priority** — split on paragraphs first, sentences next, words last. We never split mid-word.

### 3.4 Per-chunk metadata stamp

Every chunk is tagged so retrieval can filter precisely:

```python
chunk.metadata = {
    "source":       filename,
    "user_id":      user_id,           # tenancy boundary
    "document_id":  uuid5(user_id:filename),  # stable across re-uploads
    "node_type":    "raw_chunk",       # vs "document_summary" etc.
    "chunk_index":  i,
    "total_chunks": N,
    "page":         page_number,       # PDFs
    "content_type": "text" | "table" | "table_row" | "image",
    "extraction_method": "pymupdf" | "unstructured_hi_res" | "docx_media" | ...,
}
```

This metadata is what lets us do **per-user isolation**, **document-scoped search**, and **content-type filtering** at query time.

### 3.5 Tabular chunking is special

For Excel/CSV, splitting by characters would destroy the row structure. Instead each row becomes its own Document:

```text
Sheet: HospitalRecords
Row: 47
Patient Name: Jane Doe
Medical Condition: Pneumonia
Admission Date: 2024-03-04 (3/4/2024, 3/4/24)   ← multiple date variants
Discharge Date: 2024-03-09 (3/9/2024, 3/9/24)
Record: Patient Name=Jane Doe | Medical Condition=Pneumonia | ...
Facts: Patient Jane Doe was admitted for Pneumonia on 2024-03-04.
       Patient Jane Doe was discharged on 2024-03-09.
```

Three things happen here:
1. **Date normalization** — every parseable date is rewritten in 5 formats so `"3/4/24"` and `"2024-03-04"` both match.
2. **Derived facts** — full natural-language sentences are appended so the embedding model captures the *meaning*, not just the columns.
3. **Compact + verbose** representations of the same row maximize recall.

---

## 4. Summary Index — What It's For

Code: `app/rag/ingestion.py` (builders) + `app/rag/vectorstore.py` (collection).

ChromaDB stores **two collections**:

| Collection | Holds | Purpose |
|---|---|---|
| `rag_documents_raw` | Raw chunks (the source of truth) | Final grounding context for the LLM |
| `rag_documents_summary` | LLM-generated summary nodes | Cheap, semantically dense first-hop search |

### 4.1 Four kinds of summary nodes

| `node_type` | One per | Generated by | Why |
|---|---|---|---|
| `document_summary` | Whole document | Groq LLM (350 tokens) | "What is this document about?" |
| `section_summary` | Heading section (optional) | Groq LLM | Mid-grain semantic anchor |
| `table_summary` | Each detected table | Groq LLM | Tables don't embed well as raw text |
| `image_summary` | Each embedded image | Groq **vision LLM** | Makes images *searchable as text* |

A scanned chart of "patient admissions by month" never carries that text — but the vision model produces a sentence like *"Bar chart showing monthly hospital admissions, peaking in March 2024 at 142"*, and that sentence goes into the index. **That's how images become first-class RAG content.**

### 4.2 Why have a summary collection at all?

Two reasons:
1. **Routing** — searching summaries first tells us *which document* is most relevant. Then we can pull all the raw chunks for *that* document and rerank them.
2. **Coverage** — a long document's raw chunks are noisy; a single summary node is dense and concrete, so it's much more likely to match a vague query like *"what's the legal compliance angle of this report?"*.

This is the **Multi-Vector / Parent-Document Retriever** pattern.

### 4.3 Linking summaries to parents

Every summary node carries `parent_id = document_id`. After we get a top-k summary hit, we call `get_chunks_by_document_id()` to load every raw chunk for that document — those chunks are what the LLM finally sees.

---

## 5. Query Optimization — How Retrieval Actually Runs

Code: `app/rag/retriever.py`, function `retrieve()`.

The retrieval pipeline is **6 stages**. Each one fixes a real failure mode of naive RAG.

```text
              user query
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 1. MULTI-QUERY EXPANSION     │  LLM rephrases the query into
   │    _generate_query_variants  │  3 alternative phrasings.
   └──────────────────────────────┘  Recall ↑ (different keywords).
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 2. SUMMARY-FIRST SEARCH      │  similarity_search on the SUMMARY
   │    _collect_summary_hits     │  collection for each variant.
   └──────────────────────────────┘  Identifies relevant *documents*.
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 3. PARENT EXPANSION          │  For each matched document:
   │    get_chunks_by_document_id │  load ALL raw chunks (ordered).
   └──────────────────────────────┘  + targeted raw search per variant.
                  │                  Precision ↑ on grounded text.
                  ▼
   ┌──────────────────────────────┐
   │ 4. STRUCTURED INTENT FILTER  │  For tabular queries with a date
   │    _apply_structured_date_   │  + intent ("admitted on 3/4/24"),
   │    intent_filter             │  keep only rows where date and
   └──────────────────────────────┘  intent co-occur on a line.
                  │                  Stops admit/discharge confusion.
                  ▼
   ┌──────────────────────────────┐
   │ 5. CROSS-ENCODER RERANK      │  ms-marco MiniLM scores every
   │    rerank()                  │  (query, passage) pair directly.
   └──────────────────────────────┘  This is the real precision filter.
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 6. IMAGE SUMMARY INJECTION   │  If the top docs have related
   │    + GROUNDING GUARDRAIL     │  image_summary nodes, append the
   └──────────────────────────────┘  best 2 as visual context.
                  │                  Then check answer-vs-context
                  ▼                  overlap and append a low-conf
              top-k chunks           note if too low.
```

### Why each stage matters

1. **Multi-query**: a user asks *"what was the merger reasoning?"* but the doc says *"rationale for the acquisition"*. One query misses it; three find it.
2. **Summary-first**: avoids reranking thousands of raw chunks. We narrow to the right *document* in milliseconds.
3. **Parent expansion**: never answer from a summary alone — always grounded in the raw text.
4. **Structured intent filter**: a hospital row contains 4 dates (admit, discharge, surgery, follow-up). Vector similarity treats them all as "dates", so we add a regex-level co-occurrence check to disambiguate.
5. **Cross-encoder rerank**: bi-encoder cosine is fast but coarse. The cross-encoder reads query+passage *together* and is dramatically more accurate. We rely on it as the final precision filter — the bi-encoder score threshold is intentionally disabled.
6. **Grounding guardrail**: cheap token-overlap check between the answer and the retrieved context. If too few answer tokens are supported, append a "may be weakly grounded" disclaimer to the user.

### Routing decision (after retrieval)

`router_node` in `agents/rag_agent.py:218`:

| Condition | Goes to | Behavior |
|---|---|---|
| Query mentions a known doc name | `doc_summary_node` | Loads full doc + all summaries |
| Best reranked score ≥ -3.0 | `rag_answer_node` | Standard RAG answer with citations |
| Otherwise | `web_search_node` | DuckDuckGo fallback |

---

## 6. Memory Model

| Layer | Where | Scope | Use |
|---|---|---|---|
| Short-term | SQLite query — last 8 turns of *current* session | One session | Continuity within active chat |
| Long-term | SQLite query — last 12 messages from *other* sessions | All sessions | Context after re-login |

Both are appended in front of the user query before invoking the LLM, so the model sees a coherent conversation history.

Tables (`memory/long_term.py`):
- `conversation_history` — every user/assistant message, with sources JSON.
- `chat_session_meta` — session titles.

---

## 7. Authentication

`app/auth/`:
- **Username/password**: bcrypt hashing, JWT (`HS256`, 24h).
- **Google OAuth**: `Flow.from_client_config(...)` → `/auth/google/login` → callback links/creates `User`, redirects to Streamlit with the JWT.
- Every protected route uses `Depends(get_current_user)`; every Chroma query filters on `user_id` — that's the hard tenancy boundary.

---

## 8. Observability & Guardrails

- **Latency**: HTTP middleware logs every request; `/chat` returns `latency_ms` to the UI.
- **Token usage**: extracted from Groq `response_metadata.token_usage` and surfaced in the UI.
- **Grounding ratio**: fraction of answer tokens (≥4 chars) found in retrieved context. Below `0.12`, append a low-confidence note.
- **System prompts** explicitly forbid: inventing facts, exposing retrieval metadata, harmful/unsafe responses.

---

## 9. Data Lifecycle (Delete is real)

When a document is deleted (`/documents/{filename}`):
1. Delete all raw + summary nodes from ChromaDB.
2. Delete the file from disk.
3. Delete `.multimodal_assets/<doc_id>/` (extracted images).
4. Clear vectorstore caches and reset the LangGraph agent.
5. Delete `conversation_history` rows whose `sources` JSON references this filename, plus their paired user message.

No orphaned chunks. No orphaned chat references.

---

## 10. Configuration Highlights

`app/config.py` (`pydantic-settings`, no caching — `.env` edits picked up live):

```env
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5
SUMMARY_TOP_K=3
ENABLE_SUMMARY_INDEX=true
ENABLE_IMAGE_SUMMARIES=true
GROUNDING_MIN_RATIO=0.12
WHISPER_MODEL=base
```

---

## 11. Deployment

- **Local**: `uvicorn app.main:app` + `streamlit run frontend/streamlit_app.py` (or `./start.sh`).
- **Docker** (`docker-compose.yml`): two services — `api` (FastAPI) and `frontend` (Streamlit). Volumes persist `vectorstore/`, `data/`, and `app.db`.

---

## 12. Design Decisions Worth Mentioning

- **Two Chroma collections** — summary as router, raw as ground truth.
- **Cross-encoder is the real precision filter** — bi-encoder threshold intentionally disabled.
- **Tabular rows are atomic Documents** with date variants + derived facts — embeddings work on natural language, not column names.
- **LangGraph over `if/else`** — explicit nodes/edges; trivial to add a "clarify" node or a SQL-tool node later.
- **Local-first** — embeddings, reranker, whisper all run on CPU. Only the LLM and web search are external.
- **No `Settings` cache** — `.env` edits reflected without restart (helpful in demos).
