"""FastAPI app for VibeVox MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import asyncio
import io
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydub import AudioSegment

from dialogue_detection import has_dialogue
from groq_client import analyze_text_groq, load_groq_config
from ingestion import ingest_text
from style_prompt_compiler import compile_style_prompt


APP_DIR = Path(__file__).parent
INDEX_PATH = APP_DIR / "index.html"

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


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VIBEVOX_HOST", "0.0.0.0")
    port = int(os.environ.get("VIBEVOX_PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
