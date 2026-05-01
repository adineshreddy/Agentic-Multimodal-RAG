import io
import tempfile
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.auth.models import User
from app.auth.utils import get_current_user
from app.config import get_settings
from loguru import logger

settings = get_settings()
router = APIRouter(prefix="/voice", tags=["Voice"])

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper model '{settings.whisper_model}'...")
        _whisper_model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8",  # CPU-friendly quantisation
        )
        logger.info("Whisper model loaded.")
    return _whisper_model


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    duration_s: float | None = None


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    _: User = Depends(get_current_user),
):
    """
    Transcribe an uploaded audio file (wav/mp3/webm/ogg) using faster-whisper.
    Returns the detected transcript and language.
    """
    allowed = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".flac"}
    ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else ".wav"
    if ext not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported audio format: {ext}")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    model = _get_whisper()

    # Write to a temp file because faster-whisper needs a file path
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"Transcribed {len(audio_bytes)//1024}KB audio: '{transcript[:60]}...'")
        return TranscribeResponse(
            transcript=transcript,
            language=info.language,
            duration_s=round(info.duration, 2),
        )
    except Exception as exc:
        logger.error(f"Transcription error: {exc}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")
    finally:
        os.unlink(tmp_path)
