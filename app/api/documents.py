import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.models import User
from app.auth.utils import get_current_user
from app.config import get_settings
from app.database import get_db
from app.agents.rag_agent import reset_agent
from app.memory.long_term import delete_history_for_source
from app.rag.ingestion import SUPPORTED_EXTENSIONS, ingest_file, remove_multimodal_assets
from app.rag.vectorstore import clear_vectorstore_caches, delete_source, list_sources
from loguru import logger

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])


class IngestResult(BaseModel):
    file: str
    chunks: int | None = None
    ids: int | None = None
    error: str | None = None
    warning: str | None = None


class DocumentsResponse(BaseModel):
    sources: list[str]
    total: int


def _user_documents_dir(user_id: int) -> str:
    return os.path.join(settings.documents_path, f"user_{user_id}")


@router.post("/upload", response_model=IngestResult)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload a document, save it locally, and ingest it into ChromaDB."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    user_documents_dir = _user_documents_dir(current_user.id)
    os.makedirs(user_documents_dir, exist_ok=True)
    save_path = os.path.join(user_documents_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[user={current_user.username}] Uploaded '{file.filename}'")

    try:
        result = ingest_file(save_path, user_id=current_user.id)
        return IngestResult(**result)
    except Exception as exc:
        logger.error(f"Ingestion failed for '{file.filename}': {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")


@router.get("", response_model=DocumentsResponse)
def list_documents(current_user: User = Depends(get_current_user)):
    """List all document sources currently indexed in ChromaDB."""
    sources = list_sources(user_id=current_user.id)
    return DocumentsResponse(sources=sources, total=len(sources))


@router.delete("/{filename}")
def delete_document(
    filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Fully remove a document:
    1. Delete all chunks from ChromaDB vectorstore
    2. Delete the physical file from disk
    3. Clear the in-memory vectorstore cache (so next query reconnects cleanly)
    4. Remove conversation history rows that referenced this document
    """
    # 1. Remove chunks from ChromaDB
    chunks_removed = delete_source(filename, user_id=current_user.id)
    logger.info(f"[user={current_user.username}] Removed {chunks_removed} chunks for '{filename}'")

    # 2. Delete local file
    local_path = os.path.join(_user_documents_dir(current_user.id), filename)
    if os.path.exists(local_path):
        os.remove(local_path)
        logger.info(f"[user={current_user.username}] Deleted file '{filename}' from disk")

    # 2b. Delete persisted multimodal extraction assets for this source
    assets_removed = remove_multimodal_assets(local_path, user_id=current_user.id)
    if assets_removed:
        logger.info(f"[user={current_user.username}] Removed multimodal assets for '{filename}'")

    # 3. Clear caches so next operation gets a fresh state
    clear_vectorstore_caches()
    reset_agent()

    # 4. Remove conversation history rows that cite this document
    history_removed = delete_history_for_source(
        db,
        user_id=current_user.id,
        source_filename=filename,
    )
    if history_removed:
        logger.info(f"[user={current_user.username}] Removed {history_removed} history rows "
                    f"referencing '{filename}'")

    return {
        "message": f"'{filename}' fully removed.",
        "chunks_removed": chunks_removed,
        "asset_dirs_removed": assets_removed,
        "history_rows_removed": history_removed,
    }
