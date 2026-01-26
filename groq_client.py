"""Groq client wrapper for Emotion/Intensity analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Optional

import httpx


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


@dataclass
class GroqConfig:
    api_key: str
    model: str = DEFAULT_GROQ_MODEL
    timeout_seconds: float = 30.0


def load_groq_config(model: Optional[str] = None) -> GroqConfig:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return GroqConfig(api_key=api_key, model=model or DEFAULT_GROQ_MODEL)


SYSTEM_PROMPT = (
    "You are a narration sentiment analyzer. Return only JSON with keys "
    "Emotion and Intensity. Emotion must be one of: Neutral, Joy, Sadness, "
    "Anger, Fear, Suspense, Surprise, Disgust, Tender, Calm, Awe, Nostalgia, "
    "Boredom, Excitement, Pride, Relief, Hope, Melancholy, Guilt, Frustration, "
    "Curiosity, Determination, Contempt, Serenity, Envy. "
    "Intensity must be one of: Low, Medium, High."
)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def analyze_text_groq(text: str, config: GroqConfig) -> tuple[str, str, str]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {text}\nJSON:"},
        ],
    }

    with httpx.Client(timeout=config.timeout_seconds) as client:
        response = client.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    emotion = parsed.get("Emotion", "Neutral")
    intensity = parsed.get("Intensity", "Medium")
    return emotion, intensity, content
