# VibeVox: AI Voice Cloning & Emotion Engine 🎙️⚡

> **Clone Your Voice. Perform Any Emotion. Read Anything.**

![Status](https://img.shields.io/badge/Status-Beta-purple?style=for-the-badge)
![Model](https://img.shields.io/badge/Core-Qwen3_TTS-navy?style=for-the-badge)
![Capability](https://img.shields.io/badge/Voice-Cloning_%2B_Emotion-crimson?style=for-the-badge)

VibeVox is an advanced AI Web Application designed to democratize professional-grade voice synthesis. It allows anyone to **clone their own voice** using just a short sample and then use that digital twin to **perform text with deep emotional intelligence**.

We solve the "robotic voice" problem by combining state-of-the-art Voice Cloning (Qwen3-TTS) with a semantic sentiment analysis engine (Llama 3.1) that directs *how* the cloned voice should feel—be it a terrified whisper or a jubilant shout.

## 🌟 Core Features (Vision)

### 1. Zero-Shot Voice Cloning
Upload a 10-second clip of your voice (or any voice you have rights to), and VibeVox instantly creates a high-fidelity digital replica. No training time required.

### 2. Emotion-Aware Performance
Unlike standard TTS that drones on, VibeVox reads the room.
- **Sad Text?** Your cloned voice breaks, pauses, and sighs.
- **Action Scene?** Your cloned voice accelerates, projects, and intensifies.
- **Dialogue?** It detects characters and switches styles automatically.

### 3. Document ➔ Audiobook
Upload a PDF, EPUB, or paste a full novel. VibeVox will process the entire document, applying the appropriate emotional subtext to every paragraph, and generate a seamless audiobook in *your* voice.

## 🏗️ Architecture

The system uses a producer-consumer agent architecture:
1.  **Ingestion:** Breaks text/documents into semantic chunks.
2.  **Director (Analysis):** Llama 3.1 reads the text to determine the emotion (e.g., "Suspense", "Joy").
3.  **Compiler:** Translates the emotion into a complex style prompt for the TTS.
4.  **Synthesizer:** Qwen3-TTS takes your **Reference Audio** and the **Style Prompt** to generate the speech.
5.  **Studio (Web UI):** Streams the audio back to you in real-time or exports a full WAV file.

## 🚀 Getting Started

### Prerequisites
- **GPU:** NVIDIA GPU with 12GB+ VRAM (for dual models) or 8GB (one at a time)
- **Python:** 3.10+
- **Disk:** 15GB free (models + cache)
- **API:** Groq API Key (for the "Director" agent)

### Installation

1.  **Clone & Install:**
    ```bash
    # Create environment
    python -m venv venv
    source venv/bin/activate

    # Install Torch & Flash Attention (Required for speed)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install flash-attn --no-build-isolation
    pip install -r requirements.txt
    python -m nltk.downloader punkt
    ```

2.  **Configure:**
    ```bash
    export GROQ_API_KEY='your-key-here'
    ```

3.  **Launch:**
    ```bash
    ./start_vibevox.sh
    ```
    Access the Web UI at `http://localhost:8000`.

## ⚠️ Current Limitations (Beta)
- **Cloning Interface:** The UI currently relies on text descriptions for voice styling. **Audio upload for cloning is in active development.**
- **Document Support:** PDF upload is planned; currently supports text paste.

## 📚 Learn More
See [AGENTS.md](AGENTS.md) for a detailed breakdown of the intelligent agents powering VibeVox.
