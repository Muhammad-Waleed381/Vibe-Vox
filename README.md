# VibeVox: AI Voice Cloning & Emotion Engine

> **Clone Your Voice. Perform Any Emotion. Read Anything.**

![Status](https://img.shields.io/badge/Status-Beta-purple?style=for-the-badge)
![Model](https://img.shields.io/badge/Core-Qwen3_TTS-navy?style=for-the-badge)
![Capability](https://img.shields.io/badge/Voice-Cloning_%2B_Emotion-crimson?style=for-the-badge)

VibeVox is an advanced AI Web Application for professional-grade voice synthesis. It combines Qwen3-TTS with emotion-aware orchestration to offer **three synthesis modes**:

| Mode | Model | What it does |
|---|---|---|
| **Clone** | Qwen3-TTS Base | Zero-shot voice cloning from 3s reference audio |
| **Custom Voice** | Qwen3-TTS CustomVoice | 9 pre-built premium speakers with instruction-based emotion |
| **Voice Design** | Qwen3-TTS VoiceDesign | Describe any voice in natural language — creates it on the fly |

## Features

### 1. Zero-Shot Voice Cloning
Upload a 3-second clip + transcript. The Base model clones it instantly — no training. Reuse the same voice across an entire audiobook via **prompt caching** (build once, reuse for all chunks).

### 2. Emotion-Aware Performance
Every chunk is analyzed by Llama 3.1 (Groq) for emotion + intensity. The compiler generates the right instruction for the active mode:
- **Clone** → reference audio carries the voice (no instruction needed)
- **Custom Voice** → short instruction: *"Strong: Speak with anger and intensity."*
- **Voice Design** → full descriptive paragraph: *"A firm, tense voice with clipped phrasing..."*

### 3. 9 Premium Speakers (CustomVoice)
Select from pre-built voices — no reference audio required:

| Speaker | Description | Native Language |
|---|---|---|
| Vivian | Bright young female | Chinese |
| Serena | Warm gentle female | Chinese |
| Uncle Fu | Seasoned male, low mellow | Chinese |
| Dylan | Youthful Beijing male | Chinese (Beijing) |
| Eric | Lively Chengdu male | Chinese (Sichuan) |
| Ryan | Dynamic male | English |
| Aiden | Sunny American male | English |
| Ono Anna | Playful female | Japanese |
| Sohee | Warm female | Korean |

### 4. Document → Audiobook
Upload PDF, EPUB, or TXT. VibeVox chunks, analyzes, and stitches the full document into a seamless WAV export with crossfade.

## Architecture

```
┌─ User ──────────────────────────────────────┐
│  Mode: Clone | CustomVoice | VoiceDesign     │
│  Text + Reference Audio (optional)           │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Orchestrator (web/app.py)                   │
│  • Chunks text (ingestion.py)                │
│  • Analyzes emotion (groq_client.py)         │
│  • Compiles instruction (style_prompt..py)   │
│  • Builds clone prompt once if cloning       │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  TTS Server (tts_server.py)                  │
│                                              │
│  ┌──────────────┐  ┌───────────────────┐    │
│  │ Base Model   │  │ CustomVoice Model │    │
│  │ (voice clone)│  │ (9 speakers)      │    │
│  └──────┬───────┘  └────────┬──────────┘    │
│         │                   │                │
│  ┌──────┴───────┐           │                │
│  │VoiceDesign   │           │                │
│  │(novel voices)│           │                │
│  └──────────────┘           │                │
│         │                   │                │
│         └───────┬───────────┘                │
│                 ▼                            │
│          Audio WAV stream                    │
└─────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites
- **GPU:** 12GB+ VRAM (all 3 models) or 6GB (1-2 models with `MODEL_SIZE=0.6B`)
- **Python:** 3.10+
- **Disk:** 10-15GB free (model weights)
- **API:** Groq API Key (for emotion analysis)

### Installation

```bash
python -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### Configuration

```bash
export GROQ_API_KEY='your-key-here'

# Optional: model size (default: 1.7B)
export MODEL_SIZE=0.6B

# Optional: which models to load (default: base + design)
export LOAD_BASE_MODEL=true
export LOAD_CUSTOM_VOICE_MODEL=false
export LOAD_DESIGN_MODEL=true
```

### Launch

```bash
./start_vibevox.sh
```

Access the Web UI at `http://localhost:8000`.

### Mode Examples

Select a mode in the UI, then click **Live Clone & Play**:

- **Clone** — upload reference audio + transcript → speaks in that voice
- **Custom Voice** — pick a speaker, optionally type *"Very happy"* → speaker obeys the emotion
- **Voice Design** — type *"a raspy goblin voice, high-pitched with a lisp"* → novel voice created on the spot

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | Service health, uptime, loaded models |
| `GET /api/config` | Available modes, ports, limits |
| `GET /api/speakers` | Available CustomVoice speakers |
| `POST /api/analyze_text` | Returns chunk-level emotion analysis |
| `POST /api/parse_document` | Extracts text from PDF/EPUB/TXT |
| `POST /api/speak_stream` | Streams synthesized audio (NDJSON) |
| `POST /api/export` | Background audiobook export job |
| `GET /api/export/{id}` | Export job status + progress |
| `DELETE /api/export/{id}` | Cancel export |

### `POST /api/speak_stream`

```json
{
  "mode": "custom_voice",
  "text": "Hello world",
  "speaker": "Ryan",
  "style_prompt": "Speak with excitement.",
  "tts_port": 5000
}
```

Returns `application/x-ndjson` stream with audio chunks as base64 WAV.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `1.7B` | Model size: `0.6B` (lighter, faster) or `1.7B` (quality) |
| `LOAD_BASE_MODEL` | `true` | Load the voice clone model |
| `LOAD_CUSTOM_VOICE_MODEL` | `false` | Load the CustomVoice model |
| `LOAD_DESIGN_MODEL` | `true` | Load the VoiceDesign model |
| `TTS_PORT` | `5000` | TTS server port |
| `VIBEVOX_PORT` | `8000` | Web UI server port |
| `GROQ_API_KEY` | — | Required for emotion analysis |

## Learn More

See [AGENTS.md](AGENTS.md) for a detailed breakdown of the intelligent agents powering VibeVox.
