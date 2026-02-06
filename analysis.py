"""Analysis stage for Emotion/Intensity classification powered by Groq."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ingestion import ingest_text
from semantic_chunker import Chunk, DEFAULT_TOKENIZER_MODEL
from style_prompt_compiler import DEFAULT_SPEAKER, StylePrompt, compile_style_prompt
from dialogue_detection import has_dialogue
from groq_client import (
    DEFAULT_GROQ_MODEL,
    GroqConfig,
    analyze_text_groq,
    load_groq_config,
)


@dataclass
class AnalysisResult:
    chunk: Chunk
    emotion: str
    intensity: str
    style_prompt: StylePrompt
    raw_output: str


def analyze_chunk(
    chunk: Chunk,
    groq_config: GroqConfig,
    speaker: str = DEFAULT_SPEAKER,
    character_mode: bool = False,
) -> AnalysisResult:
    emotion, intensity, output_text = analyze_text_groq(chunk.text, groq_config)
    style_prompt = compile_style_prompt(
        emotion,
        intensity,
        speaker=speaker,
        character_mode=character_mode,
    )
    return AnalysisResult(
        chunk=chunk,
        emotion=emotion,
        intensity=intensity,
        style_prompt=style_prompt,
        raw_output=output_text,
    )


def analyze_text(
    text: str,
    min_tokens: int = 40,
    max_tokens: int = 150,
    character_mode: bool = False,
    groq_model: str = DEFAULT_GROQ_MODEL,
    speaker: str = DEFAULT_SPEAKER,
) -> List[AnalysisResult]:
    chunks = ingest_text(
        text,
        tokenizer_model=DEFAULT_TOKENIZER_MODEL,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    groq_config = load_groq_config(groq_model)

    results: List[AnalysisResult] = []
    for chunk in chunks:
        auto_character = has_dialogue(chunk.text)
        results.append(
            analyze_chunk(
                chunk,
                groq_config,
                speaker=speaker,
                character_mode=character_mode or auto_character,
            )
        )
    return results
