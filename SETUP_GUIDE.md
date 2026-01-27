# Qwen3-TTS Setup & Usage Guide

## 🎯 What Changed?

Your project now has **actual TTS synthesis** using Qwen3-TTS! Here's what was added:

### New Files
1. **`tts_server.py`** - FastAPI server that runs the Qwen3-TTS VoiceDesign model
2. **`start_vibevox.sh`** - Startup script to run both TTS and web servers together

### Updated Files
1. **`requirements.txt`** - Added `qwen-tts`, `soundfile`, and `flash-attn` dependencies

---

## 🚀 Installation & Setup

### Step 1: Install Dependencies

**Important:** Flash Attention requires significant compilation time. Install in order:

```bash
# Activate your virtual environment first
source venv/bin/activate

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

---

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

---

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

This will:
1. Chunk the text semantically
2. Analyze emotion/intensity with Groq
3. Compile style prompts
4. Synthesize with Qwen3-TTS
5. Stream back audio

---

## 📊 How It Works

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Text Input                                              │
│     "The haunted mansion loomed before them..."             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Semantic Chunking (ingestion.py)                        │
│     Splits into sentence/paragraph chunks                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Sentiment Analysis (analysis.py + Groq)                 │
│     Emotion: "Suspense", Intensity: "High"                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Style Prompt Compilation (style_prompt_compiler.py)     │
│     "A Male Narrator voice, strong in intensity,            │
│      a low, controlled voice with deliberate pauses..."     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  5. TTS Synthesis (tts_server.py + Qwen3-TTS)              │
│     generate_voice_design(text, instruct=style_prompt)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Audio Output (WAV stream)                               │
│     Voice matches the emotion/intensity of the text!        │
└─────────────────────────────────────────────────────────────┘
```

### Qwen3-TTS Voice Design Mode

The `generate_voice_design()` function accepts:
- **`text`**: The actual text to speak
- **`instruct`**: Natural language description of HOW to say it (your style prompts!)
- **`language`**: Target language (English, Chinese, Japanese, etc.)

Your style prompts like:
> "A deep male voice speaking slowly and quietly, as if revealing a secret"

Are **exactly** what the VoiceDesign model expects! 🎯

---

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

---

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

---

## 📚 Example Usage

### Python API Usage

```python
from analysis import analyze_text

# Analyze text with Groq
text = "The storm raged outside. Lightning illuminated the dark room."
results = analyze_text(text, groq_model="llama-3.1-8b-instant")

for result in results:
    print(f"Chunk: {result.chunk.text}")
    print(f"Emotion: {result.emotion}")
    print(f"Intensity: {result.intensity}")
    print(f"Style Prompt: {result.style_prompt.prompt}")
    print("---")
```

### Web API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/speak",
    params={
        "text": "Your text here",
        "speaker": "Male_Narrator",
        "groq_model": "llama-3.1-8b-instant"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

---

## 🎯 What Makes This Special

Unlike standard TTS that reads everything monotonously:

- ❌ **Standard TTS:** Same boring voice for everything
- ✅ **VibeVox:** Voice **adapts** to content emotion
  - Scary text → Tense, whispered voice
  - Happy text → Bright, energetic voice
  - Sad text → Soft, subdued voice

This is achieved by:
1. **Smart Analysis:** Groq LLM detects emotion/intensity
2. **Prompt Engineering:** Your style compiler creates rich voice descriptions
3. **Advanced TTS:** Qwen3-TTS VoiceDesign interprets natural language instructions

---

## 📖 Further Reading

- [Qwen3-TTS Official Repo](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Documentation](https://qwen.ai/qwen3-tts)
- [Voice Design Examples](https://github.com/QwenLM/Qwen3-TTS/tree/main/examples)

---

## 🎉 You're All Set!

Your VibeVox system now has:
- ✅ Real TTS synthesis with Qwen3-TTS
- ✅ Dynamic voice modulation based on emotion
- ✅ Proper API integration
- ✅ Production-ready server architecture

Enjoy your adaptive TTS engine! 🎙️⚡
