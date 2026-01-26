"""Analysis stage for Emotion/Intensity classification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ingestion import ingest_text
from semantic_chunker import Chunk, DEFAULT_TOKENIZER_MODEL
from style_prompt_compiler import StylePrompt, compile_style_prompt
from dialogue_detection import has_dialogue
from groq_client import GroqConfig, analyze_text_groq, load_groq_config


ANALYSIS_MODEL_ID = DEFAULT_TOKENIZER_MODEL


@dataclass
class AnalysisResult:
    chunk: Chunk
    emotion: str
    intensity: str
    style_prompt: StylePrompt
    raw_output: str


PROMPT_TEMPLATE = (
    "You are a sentiment analyzer for narration. "
    "Return a JSON object with keys Emotion and Intensity. "
    "Use short labels like Suspense, Joy, Sadness, Anger, Neutral for Emotion "
    "and Low, Medium, High for Intensity.\n\n"
    "Text: {text}\n"
    "JSON:"
)


def load_analysis_model(model_id: str = ANALYSIS_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def analyze_chunk(
    chunk: Chunk,
    model,
    tokenizer,
    max_new_tokens: int = 64,
    character_mode: bool = False,
    groq_config: GroqConfig | None = None,
) -> AnalysisResult:
    output_text = ""
    if groq_config:
        emotion, intensity, output_text = analyze_text_groq(chunk.text, groq_config)
    else:
        prompt = PROMPT_TEMPLATE.format(text=chunk.text)
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        data = _extract_json(output_text) or {}
        emotion = data.get("Emotion", "Neutral")
        intensity = data.get("Intensity", "Medium")
    style_prompt = compile_style_prompt(
        emotion,
        intensity,
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
    model_id: str = ANALYSIS_MODEL_ID,
    min_tokens: int = 40,
    max_tokens: int = 150,
    character_mode: bool = False,
    model=None,
    tokenizer=None,
    groq_model: str | None = None,
) -> List[AnalysisResult]:
    chunks = ingest_text(
        text,
        tokenizer_model=model_id,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )

    groq_config = None
    if groq_model is not None:
        groq_config = load_groq_config(groq_model)
    elif model is None or tokenizer is None:
        model, tokenizer = load_analysis_model(model_id)

    results: List[AnalysisResult] = []
    for chunk in chunks:
        auto_character = has_dialogue(chunk.text)
        results.append(
            analyze_chunk(
                chunk,
                model,
                tokenizer,
                character_mode=character_mode or auto_character,
                groq_config=groq_config,
            )
        )
    return results
