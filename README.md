# VibeVox-Core 🎙️⚡

> **Adaptive Text-to-Speech Engine with Semantic Sentiment Modulation**

![Project Status](https://img.shields.io/badge/Status-Active_Development-electricorange?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Qwen3_TTS-navy?style=for-the-badge)
![Pipeline](https://img.shields.io/badge/Architecture-Async_Pipeline-blue?style=for-the-badge)


## 📖 Overview
**VibeVox-Core** is an open-source audio synthesis engine that moves beyond static Text-to-Speech. Unlike standard TTS systems that read an entire document in a monotone "reading voice," VibeVox analyzes the **semantic sentiment** of the text stream and dynamically modulates the **prosody, pitch, and tone** of the voice model in real-time.

It utilizes a producer-consumer architecture to chain a Groq-hosted analysis LLM with a high-fidelity diffusion TTS model (Qwen3-TTS), ensuring the voice sounds "Scared" when the text is scary, and "Elated" when the text is happy—without breaking the latency budget.

## 🛠️ The "Why" (Engineering Challenges)
Building this required solving three core ML Engineering problems:
1.  **Prompt Compilation:** Mapping discrete sentiment classification labels (Positive/Negative/Neutral) into rich natural language prompts ("A shaky, whispering voice...") that Qwen3-TTS's 'Voice Design' mode can interpret.
2.  **Identity Persistence:** Ensuring that while the *emotion* changes, the *speaker identity* remains constant across inference steps (preventing "Voice Drift").
3.  **Latency Hiding:** Implementing a threaded buffer system where text analysis happens $N+1$ steps ahead of audio generation to achieve near-instant playback.

## 🏗️ Architecture

1.  **Ingestion:** Text is semantically chunked (sentence/paragraph level).
2.  **The Brain (Analysis):** A Groq-hosted LLM analyzes the chunk for two vectors:
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
    * NLP: Groq-hosted LLM (defaults to `llama-3.1-8b-instant`)
* **Audio Processing:** `FFmpeg`, `Librosa`, `PyAudio`
* **Optimization:** `BitsAndBytes` (4-bit quantization), `Accelerate`

## 🚀 Installation & Setup

### Step 1: Install Dependencies

**Important:** Flash Attention requires significant compilation time. Install in order:

```bash
# Activate your virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention (this will take 5-15 minutes)
pip install flash-attn --no-build-isolation

# Install remaining dependencies
pip install -r requirements.txt

# Download NLTK data (if not already done)
python -m nltk.downloader punkt
```

**Note:** If you have less than 96GB RAM and many CPU cores, use:
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Step 2: Set Environment Variables

```bash
# Required: Your Groq API key for sentiment analysis
export GROQ_API_KEY='your-groq-api-key-here'

# Optional: Custom ports
export TTS_PORT=5000      # Default TTS server port
export VIBEVOX_PORT=8000  # Default web server port
```

### Step 3: First-Time Model Download

The first time you run the TTS server, it will download the Qwen3-TTS model (~3.5GB). This happens automatically but may take 5-10 minutes depending on your internet speed.

## 🎮 Running VibeVox

### Option 1: Use the Startup Script (Recommended)

```bash
./start_vibevox.sh
```

This will:
- ✅ Start the TTS server on port 5000
- ✅ Start the web API on port 8000
- ✅ Show status and log locations
- ✅ Keep both running in the background

### Option 2: Run Servers Manually

**Terminal 1 - TTS Server:**
```bash
python tts_server.py
```

**Terminal 2 - Web Server:**
```bash
python -m web.app
```

## 🧪 Testing the System

### 1. Check TTS Server Health
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "device": "cuda:0"
}
```

### 2. Test Direct TTS Synthesis

```bash
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The hallway was quiet. He paused, listening for any sound.",
    "style_prompt": "A deep male voice speaking slowly and quietly, as if revealing a secret.",
    "language": "English"
  }' \
  --output test_audio.wav
```

Then play the audio:
```bash
aplay test_audio.wav  # Linux
# or
afplay test_audio.wav  # macOS
```

### 3. Test Full Pipeline (Analysis + TTS)

```bash
curl -X POST "http://localhost:8000/api/speak?text=The+haunted+mansion+loomed+before+them.+Sarah+felt+her+heart+racing." \
  --output full_pipeline_test.wav
```

### 4. Live Playback (Streamed)

The Web UI uses a streaming endpoint that starts playback as soon as the first chunk is ready.

```bash
curl -N -X POST http://localhost:8000/api/speak_stream \
  -H "Content-Type: application/json" \
  -d '{"text":"The hallway was quiet. He paused, listening for any sound.","tts_port":5000,"speaker":"Male_Narrator","groq_model":"llama-3.1-8b-instant"}'
```

### 5. Export a Full Audiobook (WAV)

Exports run as an async job and write a stitched WAV to `exports/`.

```bash
curl -X POST http://localhost:8000/api/export \
  -H "Content-Type: application/json" \
  -d '{"text":"Your long text here","tts_port":5000,"speaker":"Male_Narrator","groq_model":"llama-3.1-8b-instant","crossfade_ms":50}'

curl http://localhost:8000/api/export/<job_id>
curl -L http://localhost:8000/api/export/<job_id>/download --output audiobook.wav
```

## 🔎 Code Examples

### Semantic Chunking (MVP)

Use the ingestion pipeline to split text into semantic chunks with model-accurate token budgets.

```python
from ingestion import ingest_text

text = "The hallway was quiet. He paused, listening for any sound."
chunks = ingest_text(text, max_tokens=150)

for chunk in chunks:
    print(chunk.chunk_id, chunk.token_count, chunk.text)
```

### Analysis Stage (Emotion + Intensity)

Run the analysis LLM on each chunk to get Emotion/Intensity labels.

```python
from analysis import analyze_text

text = "The hallway was quiet. He paused, listening for any sound."
results = analyze_text(text, max_tokens=150)

for result in results:
    print(result.emotion, result.intensity, result.style_prompt.prompt)
```

### Style Prompt Compiler

Map Emotion/Intensity to a TTS style prompt for Voice Design mode.

```python
from style_prompt_compiler import compile_style_prompt

style = compile_style_prompt("Suspense", "High", speaker="Male_Narrator")
print(style.prompt)
```

## 🎛️ Configuration Options

### Model Variants

You can change the TTS model in `tts_server.py`:

```python
# Current (best quality, more VRAM ~8GB)
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Faster alternative (less VRAM ~4GB)
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
```

### GPU/CPU Selection

The server auto-detects CUDA. To force CPU:
```python
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = torch.float32
```

### Attention Mechanism

If Flash Attention fails or you're on older GPUs:
```python
use_flash_attn = False  # Falls back to "eager" attention
```

## 🐛 Troubleshooting

### Issue: "Model download is very slow"

**Solution:** Use ModelScope (faster for some regions):
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./models/qwen3-tts
```

Then update `tts_server.py`:
```python
DEFAULT_MODEL_ID = "./models/qwen3-tts"
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Use the 0.6B model instead of 1.7B
2. Lower batch size in synthesis
3. Use CPU (slower but works)

### Issue: "Flash Attention won't compile"

**Solution:** The server auto-falls back to eager attention. Or install pre-built wheels:
```bash
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Issue: "TTS server returns 502 error"

**Check:**
1. Is TTS server running? `curl http://localhost:5000/health`
2. Check TTS logs: `tail -f logs/tts_server.log`
3. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## 📚 Resources

- [Qwen3-TTS Official Repo](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Documentation](https://qwen.ai/qwen3-tts)
- [Voice Design Examples](https://github.com/QwenLM/Qwen3-TTS/tree/main/examples)
