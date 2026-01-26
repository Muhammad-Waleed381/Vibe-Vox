"""Compile Emotion/Intensity into a TTS style prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


DEFAULT_SPEAKER = "Male_Narrator"


@dataclass
class StylePrompt:
    emotion: str
    intensity: str
    speaker: str
    prompt: str


EMOTION_TEMPLATES: Dict[str, str] = {
    "neutral": "a steady, clear voice with balanced pacing",
    "joy": "a warm, bright voice with an uplifted tone",
    "sadness": "a soft, subdued voice with gentle pacing",
    "anger": "a firm, tense voice with sharp enunciation",
    "fear": "a shaky, cautious voice with tight breath control",
    "suspense": "a low, controlled voice with deliberate pauses",
    "surprise": "a lively voice with quick, rising inflection",
    "disgust": "a restrained voice with a slight edge",
    "tender": "a soft, caring voice with smooth phrasing",
    "calm": "a relaxed voice with slow, even cadence",
}

INTENSITY_MODIFIERS: Dict[str, str] = {
    "low": "subtle",
    "medium": "moderate",
    "high": "strong",
}

EMOTION_ALIASES: Dict[str, str] = {
    "positive": "joy",
    "negative": "sadness",
    "neutral": "neutral",
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "scared": "fear",
    "fearful": "fear",
}


def normalize_emotion(emotion: str) -> str:
    key = emotion.strip().lower()
    return EMOTION_ALIASES.get(key, key)


def normalize_intensity(intensity: str) -> str:
    key = intensity.strip().lower()
    if key in {"low", "medium", "high"}:
        return key
    if key in {"weak", "light"}:
        return "low"
    if key in {"strong", "intense"}:
        return "high"
    return "medium"


def format_speaker(speaker: str) -> str:
    return speaker.replace("_", " ").strip()


def compile_style_prompt(
    emotion: str,
    intensity: str,
    speaker: str = DEFAULT_SPEAKER,
) -> StylePrompt:
    norm_emotion = normalize_emotion(emotion)
    norm_intensity = normalize_intensity(intensity)
    base = EMOTION_TEMPLATES.get(norm_emotion, EMOTION_TEMPLATES["neutral"])
    modifier = INTENSITY_MODIFIERS.get(norm_intensity, "moderate")
    speaker_text = format_speaker(speaker)
    prompt = (
        f"A {speaker_text} voice, {modifier} in intensity, {base}. "
        "Maintain consistent speaker identity."
    )
    return StylePrompt(
        emotion=norm_emotion,
        intensity=norm_intensity,
        speaker=speaker,
        prompt=prompt,
    )
