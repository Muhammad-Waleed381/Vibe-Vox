"""Compile Emotion/Intensity into a TTS style prompt or instruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


DEFAULT_SPEAKER = "Male_Narrator"
SUPPORTED_MODES = ["voice_clone", "custom_voice", "voice_design"]


@dataclass
class StylePrompt:
    emotion: str
    intensity: str
    speaker: str
    prompt: str


EMOTION_TEMPLATES: Dict[str, str] = {
    "neutral": (
        "a steady, neutral voice with clear articulation, even pacing, and minimal inflection"
    ),
    "awe": (
        "a hushed, expansive voice with lifted resonance, slow pacing, and a sense of wonder"
    ),
    "nostalgia": (
        "a warm, reflective voice with gentle pacing, softened consonants, and a distant tone"
    ),
    "boredom": (
        "a flat, low-energy voice with slow pacing, reduced pitch variation, and soft volume"
    ),
    "joy": (
        "a warm, bright voice with a light smile in the tone, buoyant pacing, and open vowels"
    ),
    "excitement": (
        "an energetic voice with quick pacing, elevated pitch, and lively emphasis"
    ),
    "pride": (
        "a confident, lifted voice with steady pacing, strong resonance, and clear emphasis"
    ),
    "relief": (
        "a softened voice with a gentle exhale, easing pace, and relaxed tone"
    ),
    "hope": (
        "a bright, forward voice with gentle lift, steady pacing, and open tone"
    ),
    "sadness": (
        "a soft, subdued voice with slower pacing, gentle falls in pitch, and restrained energy"
    ),
    "melancholy": (
        "a wistful voice with slow pacing, warm resonance, and lingering vowels"
    ),
    "guilt": (
        "a subdued voice with lowered energy, quieter volume, and careful phrasing"
    ),
    "anger": (
        "a firm, tense voice with clipped phrasing, sharp consonants, and controlled intensity"
    ),
    "frustration": (
        "a tight, tense voice with uneven pacing, clipped phrasing, and restrained force"
    ),
    "fear": (
        "a cautious, breathy voice with slight tremor, tighter phrasing, and alert pacing"
    ),
    "suspense": (
        "a low, controlled voice with deliberate pauses, narrowed tone, and measured cadence"
    ),
    "surprise": (
        "a lively voice with quick, rising inflection, widened tone, and sudden emphasis"
    ),
    "curiosity": (
        "a lightly lifted voice with inquisitive inflection, moderate pace, and soft emphasis"
    ),
    "disgust": (
        "a restrained voice with slight nasal edge, compressed phrasing, and subtle tension"
    ),
    "contempt": (
        "a cool, clipped voice with dry delivery, slight edge, and restrained emphasis"
    ),
    "tender": (
        "a soft, caring voice with smooth phrasing, gentle warmth, and rounded vowels"
    ),
    "determination": (
        "a focused voice with steady pacing, firm articulation, and forward energy"
    ),
    "calm": (
        "a relaxed voice with slow, even cadence, low arousal, and smooth transitions"
    ),
    "serenity": (
        "a tranquil, flowing voice with slow cadence, softened edges, and steady breath"
    ),
    "envy": (
        "a tight, measured voice with restrained tone, subtle tension, and careful phrasing"
    ),
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
    "excited": "excitement",
    "sad": "sadness",
    "melancholic": "melancholy",
    "angry": "anger",
    "mad": "anger",
    "scared": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "tense": "suspense",
    "curious": "curiosity",
    "hopeful": "hope",
    "proud": "pride",
    "relieved": "relief",
    "determined": "determination",
    "bored": "boredom",
    "serene": "serenity",
    "nostalgic": "nostalgia",
}

NARRATION_BASE = (
    "Narration style for long-form book reading: maintain clarity, consistent speaker identity, "
    "and smooth pacing. Avoid exaggerated character acting unless explicitly required."
)

CHARACTER_MODE_BASE = (
    "Character mode: allow stronger emotional color and expressive delivery, while keeping "
    "speech intelligible and speaker identity stable."
)

INTENSITY_RULES: Dict[str, str] = {
    "low": "keep dynamics subtle, soften emphasis, and maintain steady pacing",
    "medium": "use moderate emphasis with natural variation and clear articulation",
    "high": "increase emphasis and contrast while keeping narration controlled and intelligible",
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
    cleaned = speaker.replace("_", " ").strip()
    return cleaned or DEFAULT_SPEAKER.replace("_", " ")


INSTRUCT_TEMPLATES: Dict[str, str] = {
    "neutral": "Speak with a neutral, steady tone.",
    "awe": "Speak with awe and wonder.",
    "nostalgia": "Speak with warm nostalgia.",
    "boredom": "Speak in a bored, flat tone.",
    "joy": "Speak with joy and brightness.",
    "excitement": "Speak with excitement and energy.",
    "pride": "Speak with pride and confidence.",
    "relief": "Speak with relief and ease.",
    "hope": "Speak with hopefulness.",
    "sadness": "Speak with sadness and melancholy.",
    "melancholy": "Speak with a wistful, melancholy tone.",
    "guilt": "Speak with guilt and hesitation.",
    "anger": "Speak with anger and intensity.",
    "frustration": "Speak with frustration and tension.",
    "fear": "Speak with fear and caution.",
    "suspense": "Speak with suspense and tension.",
    "surprise": "Speak with surprise and sudden emphasis.",
    "curiosity": "Speak with curiosity and intrigue.",
    "disgust": "Speak with disgust and restraint.",
    "contempt": "Speak with contempt and coolness.",
    "tender": "Speak with tenderness and care.",
    "determination": "Speak with determination and focus.",
    "calm": "Speak calmly and evenly.",
    "serenity": "Speak with serenity and peace.",
    "envy": "Speak with envy and restraint.",
}

INSTRUCT_INTENSITY: Dict[str, str] = {
    "low": "mild",
    "medium": "moderate",
    "high": "strong",
}


def compile_instruction(
    emotion: str,
    intensity: str,
    mode: str = "voice_design",
) -> str:
    """Return a mode-appropriate instruction string.

    voice_design  → full descriptive paragraph  (for generate_voice_design)
    custom_voice  → short direct sentence       (for generate_custom_voice instruct)
    voice_clone   → empty string                (no instruction needed)
    """
    norm_emotion = normalize_emotion(emotion)
    norm_intensity = normalize_intensity(intensity)

    if mode == "custom_voice":
        modifier = INSTRUCT_INTENSITY.get(norm_intensity, "moderate")
        base = INSTRUCT_TEMPLATES.get(norm_emotion, INSTRUCT_TEMPLATES["neutral"])
        return f"{modifier.capitalize()}: {base}"
    if mode == "voice_design":
        return compile_style_prompt(emotion, intensity).prompt
    return ""


def compile_style_prompt(
    emotion: str,
    intensity: str,
    speaker: str = DEFAULT_SPEAKER,
    character_mode: bool = False,
) -> StylePrompt:
    norm_emotion = normalize_emotion(emotion)
    norm_intensity = normalize_intensity(intensity)
    base = EMOTION_TEMPLATES.get(norm_emotion, EMOTION_TEMPLATES["neutral"])
    modifier = INTENSITY_MODIFIERS.get(norm_intensity, "moderate")
    intensity_rule = INTENSITY_RULES.get(norm_intensity, INTENSITY_RULES["medium"])
    speaker_text = format_speaker(speaker)
    mode_base = CHARACTER_MODE_BASE if character_mode else NARRATION_BASE
    prompt = (
        f"A {speaker_text} voice, {modifier} in intensity, {base}. "
        f"{intensity_rule}. "
        f"{mode_base}"
    )
    return StylePrompt(
        emotion=norm_emotion,
        intensity=norm_intensity,
        speaker=speaker,
        prompt=prompt,
    )
