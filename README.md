# VibeVox-Core 🎙️⚡

> **Adaptive Text-to-Speech Engine with Semantic Sentiment Modulation**

![Project Status](https://img.shields.io/badge/Status-Active_Development-electricorange?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Qwen3_TTS-navy?style=for-the-badge)
![Pipeline](https://img.shields.io/badge/Architecture-Async_Pipeline-blue?style=for-the-badge)

VibeVox-Core is an open-source audio synthesis engine that moves beyond static Text-to-Speech. Unlike standard TTS systems that read an entire document in a monotone "reading voice," VibeVox analyzes the **semantic sentiment** of the text stream and dynamically modulates the **prosody, pitch, and tone** of the voice model in real-time.

It utilizes a producer-consumer architecture to chain a Groq-hosted analysis LLM with a high-fidelity diffusion TTS model (Qwen3-TTS), ensuring the voice sounds "Scared" when the text is scary, and "Elated" when the text is happy—without breaking the latency budget.

## 🚀 Quick Start

### 1. Install Dependencies
**Important:** Flash Attention requires significant compilation time (5-15 mins). Install in this order:

```bash
# Activate your virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Install remaining dependencies
pip install -r requirements.txt

# Download NLTK data (required for chunking)
python -m nltk.downloader punkt
```

**Note:** If you have less than 96GB RAM:
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### 2. Set Environment Variables
You need a Groq API key for the semantic analysis (Llama 3.1).

```bash
export GROQ_API_KEY='your-groq-api-key-here'

# Optional: Custom ports
export TTS_PORT=5000      # Default TTS server port
export VIBEVOX_PORT=8000  # Default web server port
```

### 3. Run VibeVox
Use the provided startup script which handles both the TTS and Web servers:

```bash
./start_vibevox.sh
```

This will:
- ✅ Start the TTS server on port 5000 (first run downloads the ~3.5GB model)
- ✅ Start the web API on port 8000
- ✅ Show status and log locations

## 🎮 Usage

### Web Interface
Open your browser to [http://localhost:8000](http://localhost:8000) to access the VibeVox MVP interface.
- **Live Play:** Enter text and hear it synthesize in real-time.
- **Export:** Build a full audiobook with cross-faded transitions.

### API Endpoints

#### Live Streaming (NDJSON)
```bash
curl -N -X POST http://localhost:8000/api/speak_stream \
  -H "Content-Type: application/json" \
  -d '{"text":"The hallway was quiet...","tts_port":5000,"speaker":"Male_Narrator","groq_model":"llama-3.1-8b-instant"}'
```

#### Export Job (Async)
```bash
curl -X POST http://localhost:8000/api/export \
  -H "Content-Type: application/json" \
  -d '{"text":"Your long text here","crossfade_ms":50}'
```

## 🐛 Troubleshooting

### "Model download is very slow"
Use ModelScope (faster for some regions):
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./models/qwen3-tts
```
Then update `DEFAULT_MODEL_ID` in `tts_server.py`.

### "CUDA out of memory"
1. Use the 0.6B model instead of 1.7B in `tts_server.py`.
2. Lower batch size in synthesis (though currently processed sequentially).
3. Use CPU (slower but works).

### "Flash Attention won't compile"
The server auto-falls back to eager attention. Or install pre-built wheels:
```bash
pip install flash-attn --no-build-isolation --no-cache-dir
```

## 📚 Resources
- **System Architecture:** See [AGENTS.md](AGENTS.md)
- **Qwen3-TTS Official Repo:** [https://github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
