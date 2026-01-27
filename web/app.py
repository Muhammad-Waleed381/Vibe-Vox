"""FastAPI app for VibeVox MVP."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, Optional
import os
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from dialogue_detection import has_dialogue
from groq_client import analyze_text_groq, load_groq_config
from ingestion import ingest_text
from style_prompt_compiler import compile_style_prompt


APP_DIR = Path(__file__).parent
INDEX_PATH = APP_DIR / "index.html"

DEFAULT_TTS_PORT = 5000
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_SPEAKER = "Male_Narrator"

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


async def _tts_stream(
    text: str,
    style_prompt: str,
    speaker: str,
    port: int,
) -> AsyncGenerator[bytes, None]:
    url = _build_tts_url(port)
    payload = {
        "text": text,
        "style_prompt": style_prompt,
        "speaker": speaker,
    }

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                detail = await response.aread()
                raise HTTPException(
                    status_code=502,
                    detail=f"TTS server error: {detail.decode('utf-8', 'ignore')}",
                )
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


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

    async def pipeline_stream() -> AsyncGenerator[bytes, None]:
        queue: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue(maxsize=2)
        producer_task = asyncio.create_task(producer(queue))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                style_prompt, chunk_text = item
                async for audio_chunk in _tts_stream(
                    chunk_text,
                    style_prompt,
                    speaker,
                    tts_port,
                ):
                    yield audio_chunk
        finally:
            producer_task.cancel()

    return StreamingResponse(
        pipeline_stream(),
        media_type=audio_mime,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VIBEVOX_HOST", "0.0.0.0")
    port = int(os.environ.get("VIBEVOX_PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
