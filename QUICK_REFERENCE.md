# VibeVox Quick Reference Card

## 🚀 Quick Start Commands

```bash
# 1. First Time Setup
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
export GROQ_API_KEY='your-key'

# 2. Test Installation
python test_installation.py

# 3. Start System
./start_vibevox.sh

# 4. Stop System
./stop_vibevox.sh
```

---

## 🔌 API Endpoints

### TTS Server (Port 5000)
```bash
# Health Check
curl http://localhost:5000/health

# Synthesize Speech
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "style_prompt": "A clear voice", "language": "English"}' \
  --output audio.wav
```

### Web Server (Port 8000)
```bash
# Health Check
curl http://localhost:8000/api/health

# Full Pipeline (Text → Analysis → TTS)
curl -X POST "http://localhost:8000/api/speak?text=Your+text+here" \
  --output output.wav
```

---

## 📂 File Structure

```
Vibe-Vox/
├── tts_server.py          # TTS synthesis server (NEW)
├── web/app.py             # Web API server
├── analysis.py            # Sentiment analysis
├── style_prompt_compiler.py  # Emotion → Voice prompts
├── ingestion.py           # Text chunking
├── requirements.txt       # Dependencies (UPDATED)
├── start_vibevox.sh       # Startup script (NEW)
├── stop_vibevox.sh        # Shutdown script (NEW)
├── test_installation.py   # Installation test (NEW)
├── README.md              # Main docs (UPDATED)
├── SETUP_GUIDE.md         # Detailed setup (NEW)
└── IMPLEMENTATION_CORRECTIONS.md  # Change summary (NEW)
```

---

## 🎛️ Environment Variables

```bash
# Required
export GROQ_API_KEY='your-groq-api-key'

# Optional
export TTS_PORT=5000           # TTS server port
export VIBEVOX_PORT=8000       # Web server port
export TTS_HOST=127.0.0.1      # TTS server host
export VIBEVOX_HOST=0.0.0.0    # Web server host
```

---

## 🧪 Testing

```bash
# Test all dependencies
python test_installation.py

# Test TTS directly
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "style_prompt": "Clear voice", "language": "English"}' \
  --output test.wav && aplay test.wav

# Test full pipeline
curl -X POST "http://localhost:8000/api/speak?text=The+storm+raged" \
  --output full.wav && aplay full.wav
```

---

## 🔧 Common Issues

### Problem: CUDA out of memory
**Solution:** Use the smaller model
```python
# In tts_server.py, line 23
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
```

### Problem: Flash Attention won't install
**Solution:** It will fallback to eager attention (slower but works)
```bash
# Or force install with limited jobs
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Problem: TTS server not responding
**Solution:** Check logs
```bash
tail -f logs/tts_server.log
```

### Problem: Model download is slow
**Solution:** Use ModelScope (faster in some regions)
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/qwen3-tts
```

---

## 📊 System Requirements

- **GPU:** NVIDIA with 8GB+ VRAM (1.7B model) or 4GB+ (0.6B model)
- **RAM:** 16GB+ recommended
- **Disk:** 10GB free (models + cache)
- **CUDA:** 11.8+ or 12.x
- **Python:** 3.10 - 3.12
- **Internet:** Required for first-time model download

---

## 🎯 Model Options

### VoiceDesign (Current)
```python
"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
```
- **Best for:** Natural language voice control
- **VRAM:** ~8GB
- **Use case:** Your adaptive TTS system

### CustomVoice (Alternative)
```python
"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
```
- **Best for:** Preset speakers with instructions
- **VRAM:** ~8GB
- **Speakers:** Aiden, Ryan, Serena, Vivian, etc.

### Base (Voice Cloning)
```python
"Qwen/Qwen3-TTS-12Hz-1.7B-Base"
```
- **Best for:** Cloning specific voices
- **VRAM:** ~8GB
- **Use case:** Character consistency

### 0.6B Variants
```python
"Qwen/Qwen3-TTS-12Hz-0.6B-..."
```
- **Faster:** 2x faster inference
- **VRAM:** ~4GB
- **Trade-off:** Slightly lower quality

---

## 📝 Logs Location

```bash
logs/
├── tts_server.log     # TTS server output
├── web_server.log     # Web server output
├── tts_server.pid     # TTS server process ID
└── web_server.pid     # Web server process ID
```

View logs in real-time:
```bash
tail -f logs/tts_server.log
tail -f logs/web_server.log
```

---

## 🌐 URLs

- **Web UI:** http://localhost:8000/
- **TTS Health:** http://localhost:5000/health
- **API Health:** http://localhost:8000/api/health
- **TTS Endpoint:** http://localhost:5000/tts
- **Speak Endpoint:** http://localhost:8000/api/speak

---

## 💡 Tips

1. **First run downloads ~3.5GB** - be patient!
2. **Flash Attention compilation** takes 5-15 minutes
3. **Use startup script** for easiest management
4. **Check logs** if something goes wrong
5. **Test installation first** before running servers
6. **Keep GROQ_API_KEY** in your shell profile for persistence

---

## 📚 Documentation

- **Quick Start:** `README.md`
- **Detailed Setup:** `SETUP_GUIDE.md`
- **Changes Made:** `IMPLEMENTATION_CORRECTIONS.md`
- **This File:** Quick reference card

---

**Quick Help:**
```bash
# Start everything
./start_vibevox.sh

# Stop everything
./stop_vibevox.sh

# Test everything
python test_installation.py

# View what's running
ps aux | grep 'python.*vox'
```
