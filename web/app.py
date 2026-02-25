"""FastAPI app for VibeVox MVP."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
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

MAX_TEXT_CHARS = 120_000
MAX_REFERENCE_AUDIO_B64_CHARS = 20_000_000
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".text", ".pdf", ".epub"}

APP_START_TIME = time.time()

EXPORT_JOBS: dict[str, dict[str, Any]] = {}
EXPORT_TASKS: dict[str, asyncio.Task] = {}

app = FastAPI(title="VibeVox MVP")


class AnalyzeRequest(BaseModel):
    text: str
    groq_model: str = DEFAULT_GROQ_MODEL


class SpeakRequest(BaseModel):
    text: str | None = None
    tts_port: int = DEFAULT_TTS_PORT
    groq_model: str = DEFAULT_GROQ_MODEL
    speaker: str = DEFAULT_SPEAKER
    reference_audio: str | None = None
    reference_text: str | None = None


class ExportRequest(BaseModel):
    text: str
    tts_port: int = DEFAULT_TTS_PORT
    groq_model: str = DEFAULT_GROQ_MODEL
    speaker: str = DEFAULT_SPEAKER
    reference_audio: str | None = None
    reference_text: str | None = None
    crossfade_ms: int = DEFAULT_CROSSFADE_MS


def _validate_text(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(cleaned) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text is too large (max {MAX_TEXT_CHARS} characters)",
        )
    return cleaned


def _validate_tts_port(port: int) -> int:
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise HTTPException(status_code=400, detail="tts_port must be between 1 and 65535")
    return port


def _validate_reference_audio(reference_audio: str | None) -> str | None:
    if not reference_audio:
        return None
    if len(reference_audio) > MAX_REFERENCE_AUDIO_B64_CHARS:
        raise HTTPException(status_code=413, detail="Reference audio is too large")
    return reference_audio


def _public_export_status(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "chunk_total": job["chunk_total"],
        "chunk_done": job["chunk_done"],
        "progress": round(float(job["progress"]), 4),
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "download_url": f"/api/export/{job['job_id']}/download" if job.get("output_path") else None,
    }


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
    payload: dict[str, Any] = {
        "text": text,
        "style_prompt": style_prompt,
        "speaker": speaker,
    }
    if reference_audio:
        payload["reference_audio"] = reference_audio
    if reference_text:
        payload["reference_text"] = reference_text

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=20.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        await response.aread()
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"TTS server error: {response.text}")
        return response.content


async def _run_export_job(job_id: str, request: ExportRequest) -> None:
    job = EXPORT_JOBS[job_id]
    try:
        job["status"] = "running"
        job["updated_at"] = time.time()

        groq_config = load_groq_config(request.groq_model)
        chunks = ingest_text(request.text, max_tokens=150)
        if not chunks:
            raise RuntimeError("No chunks generated from input text")

        stitched_audio: AudioSegment | None = None
        crossfade_ms = max(0, min(int(request.crossfade_ms), 500))

        for idx, chunk in enumerate(chunks):
            auto_character = has_dialogue(chunk.text)
            emotion, intensity, _raw = await asyncio.to_thread(
                analyze_text_groq,
                chunk.text,
                groq_config,
            )
            style = compile_style_prompt(
                emotion,
                intensity,
                speaker=request.speaker,
                character_mode=auto_character,
            )

            audio_bytes = await _fetch_tts_audio(
                chunk.text,
                style.prompt,
                request.speaker,
                request.tts_port,
                reference_audio=request.reference_audio,
                reference_text=request.reference_text,
            )
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            if stitched_audio is None:
                stitched_audio = segment
            else:
                effective_crossfade = min(crossfade_ms, len(stitched_audio), len(segment))
                stitched_audio = stitched_audio.append(segment, crossfade=effective_crossfade)

            job["chunk_done"] = idx + 1
            job["progress"] = (idx + 1) / len(chunks)
            job["updated_at"] = time.time()

        if stitched_audio is None:
            raise RuntimeError("No audio generated")

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / f"{job_id}.wav"
        stitched_audio.export(output_path, format="wav")

        job["status"] = "done"
        job["progress"] = 1.0
        job["output_path"] = str(output_path)
        job["updated_at"] = time.time()
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["updated_at"] = time.time()
        raise
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["updated_at"] = time.time()
    finally:
        EXPORT_TASKS.pop(job_id, None)


@app.get("/")
def index() -> HTMLResponse:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="UI not found")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - APP_START_TIME, 2),
        "active_export_jobs": sum(1 for j in EXPORT_JOBS.values() if j["status"] in {"queued", "running"}),
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "default_tts_port": DEFAULT_TTS_PORT,
        "default_groq_model": DEFAULT_GROQ_MODEL,
        "default_speaker": DEFAULT_SPEAKER,
        "default_crossfade_ms": DEFAULT_CROSSFADE_MS,
        "limits": {
            "max_text_chars": MAX_TEXT_CHARS,
            "max_reference_audio_b64_chars": MAX_REFERENCE_AUDIO_B64_CHARS,
            "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        },
    }


@app.post("/api/analyze_text")
async def analyze_text_emotions(request: AnalyzeRequest = Body(...)):
    text = _validate_text(request.text)

    try:
        groq_config = load_groq_config(request.groq_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    chunks = ingest_text(text, max_tokens=150)
    results = []
    for chunk in chunks:
        emotion, intensity, _ = analyze_text_groq(chunk.text, groq_config)
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                "emotion": emotion,
                "intensity": intensity,
            }
        )

    emotion_counts: dict[str, int] = {}
    for item in results:
        emotion = item["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    return {
        "chunks": results,
        "total_chunks": len(chunks),
        "emotion_summary": emotion_counts,
    }


@app.post("/api/parse_document")
async def parse_document_upload(file: UploadFile = File(...)):
    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported document type")

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Document is too large")

    try:
        text = parse_document(content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "filename": filename,
        "text": text,
        "text_length": len(text),
        "word_count": len(text.split()),
    }


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

    text = _validate_text(text)
    _validate_tts_port(tts_port)
    _validate_reference_audio(reference_audio)

    try:
        groq_config = load_groq_config(groq_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    chunks = ingest_text(text, max_tokens=150)

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
                        "reference_audio": reference_audio,
                        "reference_text": reference_text,
                    }
                )
        finally:
            await queue.put(None)

    async def event_stream():
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=2)
        producer_task = asyncio.create_task(producer(queue))

        def encode_event(payload: dict[str, Any]) -> bytes:
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
async def export_audiobook(request: ExportRequest = Body(...)):
    request.text = _validate_text(request.text)
    request.tts_port = _validate_tts_port(request.tts_port)
    request.reference_audio = _validate_reference_audio(request.reference_audio)

    try:
        load_groq_config(request.groq_model)
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


@app.delete("/api/export/{job_id}")
async def cancel_export(job_id: str):
    job = EXPORT_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Export job not found")
    task = EXPORT_TASKS.get(job_id)
    if task and not task.done():
        task.cancel()
    job["status"] = "cancelled"
    job["updated_at"] = time.time()
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
