# VibeVox-Core 🎙️⚡

> **Adaptive Text-to-Speech Engine with Semantic Sentiment Modulation**

![Project Status](https://img.shields.io/badge/Status-Active_Development-electricorange?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Qwen3_TTS-navy?style=for-the-badge)
![Pipeline](https://img.shields.io/badge/Architecture-Async_Pipeline-blue?style=for-the-badge)

## 📖 Overview
**VibeVox-Core** is an open-source audio synthesis engine that moves beyond static Text-to-Speech. Unlike standard TTS systems that read an entire document in a monotone "reading voice," VibeVox analyzes the **semantic sentiment** of the text stream and dynamically modulates the **prosody, pitch, and tone** of the voice model in real-time.

It utilizes a producer-consumer architecture to chain a lightweight analysis LLM (Qwen2.5-0.5B) with a high-fidelity diffusion TTS model (Qwen3-TTS), ensuring the voice sounds "Scared" when the text is scary, and "Elated" when the text is happy—without breaking the latency budget.

## 🛠️ The "Why" (Engineering Challenges)
Building this required solving three core ML Engineering problems:
1.  **Prompt Compilation:** Mapping discrete sentiment classification labels (Positive/Negative/Neutral) into rich natural language prompts ("A shaky, whispering voice...") that Qwen3-TTS's 'Voice Design' mode can interpret.
2.  **Identity Persistence:** Ensuring that while the *emotion* changes, the *speaker identity* remains constant across inference steps (preventing "Voice Drift").
3.  **Latency Hiding:** Implementing a threaded buffer system where text analysis happens $N+1$ steps ahead of audio generation to achieve near-instant playback.

## 🏗️ Architecture

1.  **Ingestion:** Text is semantically chunked (sentence/paragraph level).
2.  **The Brain (Analysis):** `Qwen2.5-0.5B-Instruct` analyzes the chunk for two vectors:
    * *Emotion:* (e.g., "Suspense")
    * *Intensity:* (e.g., "High")
3.  **The Compiler:** A mapping layer converts vectors into a **Style Prompt**.
    * *Input:* `{"Emotion": "Suspense", "Speaker": "Male_Narrator"}`
    * *Output Prompt:* "A deep male voice speaking slowly and quietly, as if revealing a secret."
4.  **The Mouth (Synthesis):** `Qwen3-TTS` generates the audio segment.
5.  **The Stream:** Audio is stitched via `pydub` (cross-faded to remove popping) and streamed to the output device.

## 💻 Tech Stack

* **Core Inference:** `PyTorch`, `HuggingFace Transformers`
* **Models:**
    * TTS: `Qwen/Qwen3-TTS` (Voice Design Mode)
    * NLP: `Qwen/Qwen2.5-0.5B-Instruct` (Quantized INT8)
* **Audio Processing:** `FFmpeg`, `Librosa`, `PyAudio`
* **Optimization:** `BitsAndBytes` (4-bit quantization), `Accelerate`

## 🚀 Quick Start

### Prerequisites
* NVIDIA GPU (6GB+ VRAM recommended) or Mac M-Series (MPS).
* Python 3.10+
* FFmpeg installed on system path.

### Installation
```bash
git clone https://github.com/Muhammad-Waleed381/Vibe-Vox.git
cd Vibe-Vox
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## 🔎 Semantic Chunking (MVP)

Use the ingestion pipeline to split text into semantic chunks with
model-accurate token budgets.

```python
from ingestion import ingest_text

text = "The hallway was quiet. He paused, listening for any sound."
chunks = ingest_text(text, max_tokens=150)

for chunk in chunks:
    print(chunk.chunk_id, chunk.token_count, chunk.text)
```

## 🧠 Analysis Stage (Emotion + Intensity)

Run the analysis LLM on each chunk to get Emotion/Intensity labels.

```python
from analysis import analyze_text

text = "The hallway was quiet. He paused, listening for any sound."
results = analyze_text(text, max_tokens=150)

for result in results:
    print(result.emotion, result.intensity, result.style_prompt.prompt)
```

## 🎛️ Style Prompt Compiler

Map Emotion/Intensity to a TTS style prompt for Voice Design mode.

```python
from style_prompt_compiler import compile_style_prompt

style = compile_style_prompt("Suspense", "High", speaker="Male_Narrator")
print(style.prompt)
```
