"""FastAPI app for VibeVox MVP."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse


APP_DIR = Path(__file__).parent
INDEX_PATH = APP_DIR / "index.html"

DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_MODEL = "qwen3-tts"
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


def _build_ollama_url(port: int) -> str:
    return f"http://localhost:{port}/api/generate"


async def _ollama_stream(
    prompt: str,
    port: int,
    model: str,
) -> AsyncGenerator[bytes, None]:
    url = _build_ollama_url(port)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                detail = await response.aread()
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama error: {detail.decode('utf-8', 'ignore')}",
                )
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


@app.post("/api/speak")
async def speak(
    text: str,
    ollama_port: int = DEFAULT_OLLAMA_PORT,
    model: str = DEFAULT_OLLAMA_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    speaker: str = DEFAULT_SPEAKER,
    audio_mime: str = "audio/mpeg",
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        from analysis import analyze_text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis import failed: {exc}")

    try:
        results = analyze_text(
            text,
            max_tokens=150,
            groq_model=groq_model,
            speaker=speaker,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    prompt_parts = []
    for result in results:
        prompt_parts.append(result.style_prompt.prompt)
        prompt_parts.append(result.chunk.text)
    prompt = "\n\n".join(prompt_parts)

    return StreamingResponse(
        _ollama_stream(prompt, ollama_port, model),
        media_type=audio_mime,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VIBEVOX_HOST", "0.0.0.0")
    port = int(os.environ.get("VIBEVOX_PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
