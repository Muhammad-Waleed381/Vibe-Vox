"""FastAPI app for VibeVox MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import asyncio
import base64
import io
import json
import os
import time
import uuid

import httpx
import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

from dialogue_detection import has_dialogue
from document_parser import parse_document
from groq_client import analyze_text_groq, load_groq_config
from ingestion import ingest_text
from style_prompt_compiler import compile_style_prompt


APP_DIR = Path(__file__).parent
INDEX_PATH = APP_DIR / "index.html"
EXPORT_DIR = APP_DIR.parent / "exports"

DEFAULT_TTS_PORT = 5000
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_SPEAKER = "Male_Narrator"
DEFAULT_CROSSFADE_MS = 50

app = FastAPI(title="VibeVox MVP")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/parse_document")
async def parse_document_upload(file: UploadFile = File(...)):
    """Parse uploaded document and return extracted text."""
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        text = parse_document(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    text_length = len(text)
    word_count = len(text.split())

    return {
        "filename": file.filename,
        "text": text,
        "text_length": text_length,
        "word_count": word_count,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="UI not found")
    return INDEX_PATH.read_text(encoding="utf-8")


def _build_tts_url(port: int) -> str:
    return f"http://localhost:{port}/tts"


async def _fetch_tts_audio(
    text: str,
    style_prompt: str,
    speaker: str,
    port: int,
    reference_audio: str | None = None,
    reference_text: str | None = None,
) -> bytes:
    url = _build_tts_url(port)
    payload = {
        "text": text,
        "style_prompt": style_prompt,
        "speaker": speaker,
    }
    if reference_audio:
        payload["reference_audio"] = reference_audio
    if reference_text:
        payload["reference_text"] = reference_text

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        await response.aread()
        if response.status_code != 200:
            detail = response.text
            raise HTTPException(
                status_code=502,
                detail=f"TTS server error: {detail}",
            )
        return response.content


# ... (WavStitcher class remains unchanged)


class SpeakRequest(BaseModel):
    text: str | None = None
    tts_port: int = DEFAULT_TTS_PORT
    groq_model: str = DEFAULT_GROQ_MODEL
    speaker: str = DEFAULT_SPEAKER
    reference_audio: str | None = None # Base64 encoded WAV
    reference_text: str | None = None


# ... (Export logic unchanged for now, focusing on stream first)


@app.post("/api/speak_stream")
async def speak_stream(
    text: str | None = None,
    tts_port: int = DEFAULT_TTS_PORT,
    groq_model: str = DEFAULT_GROQ_MODEL,
    speaker: str = DEFAULT_SPEAKER,
    reference_audio: str | None = None,
    reference_text: str | None = None,
    body: SpeakRequest | None = Body(None),
):
    if body is not None:
        text = body.text
        tts_port = body.tts_port
        groq_model = body.groq_model
        speaker = body.speaker
        reference_audio = body.reference_audio
        reference_text = body.reference_text

    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        groq_config = load_groq_config(groq_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    chunks = ingest_text(text or "", max_tokens=150)

    async def producer(queue: asyncio.Queue) -> None:
        try:
            for chunk in chunks:
                auto_character = has_dialogue(chunk.text)
                emotion, intensity, _raw = await asyncio.to_thread(
                    analyze_text_groq,
                    chunk.text,
                    groq_config,
                )
                style = compile_style_prompt(
                    emotion,
                    intensity,
                    speaker=speaker,
                    character_mode=auto_character,
                )
                await queue.put(
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "emotion": emotion,
                        "intensity": intensity,
                        "style_prompt": style.prompt,
                        # Pass reference audio/text to every chunk
                        "reference_audio": reference_audio,
                        "reference_text": reference_text,
                    }
                )
        finally:
            await queue.put(None)

    async def event_stream():
        queue: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=2)
        producer_task = asyncio.create_task(producer(queue))

        def encode_event(payload: dict) -> bytes:
            return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

        try:
            yield encode_event(
                {
                    "type": "meta",
                    "chunk_count": len(chunks),
                    "tts_port": tts_port,
                    "speaker": speaker,
                }
            )

            while True:
                item = await queue.get()
                if item is None:
                    break
                audio_bytes = await _fetch_tts_audio(
                    item["text"],
                    item["style_prompt"],
                    speaker,
                    tts_port,
                    reference_audio=item.get("reference_audio"),
                    reference_text=item.get("reference_text"),
                )
                yield encode_event(
                    {
                        "type": "chunk",
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "emotion": item["emotion"],
                        "intensity": item["intensity"],
                        "style_prompt": item["style_prompt"],
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                        "audio_mime": "audio/wav",
                    }
                )

            yield encode_event({"type": "done"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield encode_event({"type": "error", "error": str(exc)})
        finally:
            producer_task.cancel()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/export")
async def export_audiobook(request: ExportRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        _ = load_groq_config(request.groq_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    chunks = ingest_text(request.text, max_tokens=150)
    if not chunks:
        raise HTTPException(status_code=500, detail="No chunks generated from input text")

    job_id = uuid.uuid4().hex
    EXPORT_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "chunk_total": len(chunks),
        "chunk_done": 0,
        "progress": 0.0,
        "created_at": time.time(),
        "updated_at": time.time(),
        "output_path": None,
        "error": None,
    }

    task = asyncio.create_task(_run_export_job(job_id, request))
    EXPORT_TASKS[job_id] = task

    return _public_export_status(EXPORT_JOBS[job_id])


@app.get("/api/export/{job_id}")
async def export_status(job_id: str):
    job = EXPORT_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    return _public_export_status(job)


@app.get("/api/export/{job_id}/download")
async def export_download(job_id: str):
    job = EXPORT_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    if job["status"] != "done" or not job.get("output_path"):
        raise HTTPException(status_code=409, detail="Export is not ready")

    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="Export file not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VIBEVOX_HOST", "0.0.0.0")
    port = int(os.environ.get("VIBEVOX_PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
