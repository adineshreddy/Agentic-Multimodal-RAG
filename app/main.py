import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.auth.routes import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.voice import router as voice_router
from app.config import get_settings
from app.database import init_db
from app.hf_runtime import configure_hf_runtime


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_hf_runtime()
    logger.info("Initialising database…")
    init_db()
    logger.info("Database ready.")

    # ── Confirm API keys are present ───────────────────────────────────────────
    groq = os.environ.get("GROQ_API_KEY", "")
    gid  = os.environ.get("GOOGLE_CLIENT_ID", "")
    if groq and groq != "your_groq_api_key_here":
        logger.info(f"GROQ_API_KEY loaded ✓")
    else:
        logger.warning("GROQ_API_KEY missing — chat will fail. Check .env and restart.")
    if gid and gid != "your_google_client_id_here":
        logger.info("GOOGLE_CLIENT_ID loaded ✓")
    else:
        logger.info("GOOGLE_CLIENT_ID not set — Google login disabled.")

    yield
    logger.info("Shutdown complete.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Multimodal RAG API",
    description=(
        "A multimodal RAG chatbot with document ingestion, web search fallback, "
        "voice input, JWT auth, and persistent memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request latency logging ────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{elapsed}ms]")
    return response


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(voice_router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Agentic Multimodal RAG API"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
