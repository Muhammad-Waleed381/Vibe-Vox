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
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

from dialogue_detection import has_dialogue
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
) -> bytes:
    url = _build_tts_url(port)
    payload = {
        "text": text,
        "style_prompt": style_prompt,
        "speaker": speaker,
    }

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


def _decode_wav(audio_bytes: bytes) -> AudioSegment:
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")


def _match_segment_format(segment: AudioSegment, reference: AudioSegment) -> AudioSegment:
    if segment.frame_rate != reference.frame_rate:
        segment = segment.set_frame_rate(reference.frame_rate)
    if segment.sample_width != reference.sample_width:
        segment = segment.set_sample_width(reference.sample_width)
    if segment.channels != reference.channels:
        segment = segment.set_channels(reference.channels)
    return segment


def _stitch_wav_segments(segments: list[bytes], crossfade_ms: int) -> bytes:
    if len(segments) == 1:
        return segments[0]
    combined = _decode_wav(segments[0])
    for raw in segments[1:]:
        next_segment = _decode_wav(raw)
        next_segment = _match_segment_format(next_segment, combined)
        fade_ms = min(crossfade_ms, len(combined), len(next_segment))
        combined = combined.append(next_segment, crossfade=fade_ms)
    output = io.BytesIO()
    combined.export(output, format="wav")
    return output.getvalue()


def _match_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("Expected samples to be 2D (frames, channels)")
    if samples.shape[1] == channels:
        return samples
    if channels == 1:
        return samples.mean(axis=1, keepdims=True)
    if samples.shape[1] == 1:
        return np.repeat(samples, channels, axis=1)
    raise ValueError(f"Unsupported channel conversion: {samples.shape[1]} -> {channels}")


def _resample_samples(samples: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return samples
    from math import gcd

    try:
        from scipy.signal import resample_poly
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Sample rate mismatch and scipy is unavailable for resampling"
        ) from exc

    g = gcd(from_sr, to_sr)
    up = to_sr // g
    down = from_sr // g
    resampled = resample_poly(samples, up, down, axis=0)
    return resampled.astype(np.float32, copy=False)


class WavStitcher:
    def __init__(self, out_path: Path, crossfade_ms: int):
        self.out_path = out_path
        self.crossfade_ms = max(0, int(crossfade_ms))
        self._target_sr: int | None = None
        self._channels: int | None = None
        self._pending: np.ndarray | None = None
        self._tail_samples: int = 0
        self._file: sf.SoundFile | None = None

    def add_wav_bytes(self, audio_bytes: bytes) -> None:
        samples, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True,
        )
        self.add_samples(samples, sr)

    def add_samples(self, samples: np.ndarray, sr: int) -> None:
        if self._file is None:
            self._target_sr = sr
            self._channels = samples.shape[1]
            self._tail_samples = int((self._target_sr * self.crossfade_ms) / 1000.0)
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = sf.SoundFile(
                str(self.out_path),
                mode="w",
                samplerate=self._target_sr,
                channels=self._channels,
                format="WAV",
                subtype="PCM_16",
            )
        else:
            if self._target_sr is None or self._channels is None:
                raise RuntimeError("WavStitcher is in an invalid state")
            if sr != self._target_sr:
                samples = _resample_samples(samples, sr, self._target_sr)
            if samples.shape[1] != self._channels:
                samples = _match_channels(samples, self._channels)

        self._append_with_crossfade(samples)

    def _append_with_crossfade(self, segment: np.ndarray) -> None:
        if self._file is None:
            raise RuntimeError("Output file is not initialized")

        if self._tail_samples <= 0:
            self._file.write(segment)
            return

        if self._pending is None:
            combined = segment
        else:
            overlap = min(self._tail_samples, len(self._pending), len(segment))
            if overlap <= 0:
                combined = np.concatenate([self._pending, segment], axis=0)
            else:
                extra_pending = self._pending[:-overlap]
                prev_overlap = self._pending[-overlap:]
                next_overlap = segment[:overlap]
                t = np.linspace(
                    0.0,
                    1.0,
                    overlap,
                    endpoint=False,
                    dtype=np.float32,
                )[:, None]
                mixed = prev_overlap * (1.0 - t) + next_overlap * t
                combined = np.concatenate(
                    [extra_pending, mixed, segment[overlap:]],
                    axis=0,
                )

        if len(combined) <= self._tail_samples:
            self._pending = combined
            return

        self._file.write(combined[:-self._tail_samples])
        self._pending = combined[-self._tail_samples:]

    def finalize(self) -> None:
        if self._file is None:
            return
        if self._pending is not None and len(self._pending):
            self._file.write(self._pending)
        self._pending = None
        self._file.close()
        self._file = None


EXPORT_JOBS: dict[str, dict] = {}
EXPORT_TASKS: dict[str, asyncio.Task] = {}


class ExportRequest(BaseModel):
    text: str
    tts_port: int = DEFAULT_TTS_PORT
    groq_model: str = DEFAULT_GROQ_MODEL
    speaker: str = DEFAULT_SPEAKER
    crossfade_ms: int = DEFAULT_CROSSFADE_MS


class SpeakRequest(BaseModel):
    text: str
    tts_port: int = DEFAULT_TTS_PORT
    groq_model: str = DEFAULT_GROQ_MODEL
    speaker: str = DEFAULT_SPEAKER


def _public_export_status(job: dict) -> dict:
    payload = {
        "job_id": job["job_id"],
        "status": job["status"],
        "chunk_total": job["chunk_total"],
        "chunk_done": job["chunk_done"],
        "progress": job["progress"],
        "error": job.get("error"),
    }
    if job["status"] == "done":
        payload["download_url"] = f"/api/export/{job['job_id']}/download"
    return payload


async def _run_export_job(job_id: str, request: ExportRequest) -> None:
    job = EXPORT_JOBS[job_id]
    job.update({"status": "running", "updated_at": time.time()})

    try:
        groq_config = load_groq_config(request.groq_model)
    except RuntimeError as exc:
        job.update({"status": "error", "error": str(exc), "updated_at": time.time()})
        return

    chunks = ingest_text(request.text, max_tokens=150)
    if not chunks:
        job.update(
            {
                "status": "error",
                "error": "No chunks generated from input text",
                "updated_at": time.time(),
            }
        )
        return

    out_path = EXPORT_DIR / f"vibevox_{job_id}.wav"
    stitcher = WavStitcher(out_path, crossfade_ms=request.crossfade_ms)

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
                    speaker=request.speaker,
                    character_mode=auto_character,
                )
                await queue.put((style.prompt, chunk.text))
        finally:
            await queue.put(None)

    queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue(maxsize=2)
    producer_task = asyncio.create_task(producer(queue))
    job["chunk_total"] = len(chunks)
    job["updated_at"] = time.time()
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            style_prompt, chunk_text = item
            audio_bytes = await _fetch_tts_audio(
                chunk_text,
                style_prompt,
                request.speaker,
                request.tts_port,
            )
            stitcher.add_wav_bytes(audio_bytes)

            job["chunk_done"] += 1
            job["progress"] = job["chunk_done"] / job["chunk_total"]
            job["updated_at"] = time.time()
    except Exception as exc:
        job.update({"status": "error", "error": str(exc), "updated_at": time.time()})
        try:
            stitcher.finalize()
        finally:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
        return
    finally:
        producer_task.cancel()

    stitcher.finalize()
    job.update(
        {
            "status": "done",
            "output_path": str(out_path),
            "updated_at": time.time(),
        }
    )


@app.post("/api/speak")
async def speak(
    text: str,
    tts_port: int = DEFAULT_TTS_PORT,
    groq_model: str = DEFAULT_GROQ_MODEL,
    speaker: str = DEFAULT_SPEAKER,
    audio_mime: str = "audio/wav",
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

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
                await queue.put((style.prompt, chunk.text))
        finally:
            await queue.put(None)

    queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue(maxsize=2)
    producer_task = asyncio.create_task(producer(queue))
    segments: list[bytes] = []
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            style_prompt, chunk_text = item
            audio_bytes = await _fetch_tts_audio(
                chunk_text,
                style_prompt,
                speaker,
                tts_port,
            )
            segments.append(audio_bytes)
    finally:
        producer_task.cancel()

    if not segments:
        raise HTTPException(status_code=500, detail="No audio segments generated")

    stitched_audio = _stitch_wav_segments(segments, crossfade_ms=DEFAULT_CROSSFADE_MS)
    return StreamingResponse(
        io.BytesIO(stitched_audio),
        media_type=audio_mime,
    )


@app.post("/api/speak_stream")
async def speak_stream(
    text: str | None = None,
    tts_port: int = DEFAULT_TTS_PORT,
    groq_model: str = DEFAULT_GROQ_MODEL,
    speaker: str = DEFAULT_SPEAKER,
    body: SpeakRequest | None = Body(None),
):
    if body is not None:
        text = body.text
        tts_port = body.tts_port
        groq_model = body.groq_model
        speaker = body.speaker

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
