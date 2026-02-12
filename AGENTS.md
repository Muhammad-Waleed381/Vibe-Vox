# VibeVox Agents: The Voice Cloning & Emotion Pipeline

VibeVox is built as a modular system of intelligent agents. These agents work together to enable the core promise of the platform: **taking a user's unique voice identity and infusing it with dynamic, context-aware emotion.**

## 1. The Ingestion Agent (`ingestion.py`)
**"The Reader"**
This agent prepares the input for processing.
- **Current Responsibilities:**
    - Accepts raw text input from the web interface.
    - Performs **Semantic Chunking**: slicing text into meaningful sentences/paragraphs to ensure emotions change naturally with the narrative flow, rather than sentence-by-sentence jerks.
- **Future Capabilities (Planned):**
    - **Document Parsing:** Will ingest PDF/EPUB/TXT files uploaded by the user to allow reading full books in their cloned voice.

## 2. The Analysis Agent (`analysis.py`)
**"The Director"**
This agent reads the script and decides *how* it should be performed.
- **Powered By:** Groq (Llama 3.1 8B).
- **Role:** It performs deep sentiment analysis on every text chunk.
- **Why it matters:** It doesn't just see text; it sees subtext. It instructs the system to speak a happy sentence with "Joy" and a scary one with "Suspense," ensuring the cloned voice acts the part.

## 3. The Compiler Agent (`style_prompt_compiler.py`)
**"The Prompt Engineer"**
This agent merges the "Who" (User's Voice) with the "How" (Emotion).
- **Role:** It acts as the bridge between the high-level emotion tags (e.g., `Emotion="Fear"`) and the low-level TTS constraints.
- **Critical Function:** It constructs the complex natural language prompts required by the Qwen3-TTS engine, effectively saying: *"Use [User's Cloned Voice] but modify it to sound breathy, fast, and terrified."*

## 4. The Synthesis Agent (`tts_server.py`)
**"The Voice Cloner"**
The powerhouse of the system, running the Qwen3-TTS model.
- **Current State:** Generates high-fidelity speech from text prompts ("Voice Design").
- **Target State (Voice Cloning):** This agent is the primary target for upgrade. It needs to accept a **Reference Audio** (the user's voice sample) and perform Zero-Shot Voice Cloning *modulated* by the styling instructions from the Compiler Agent.
- **Goal:** To reproduce the user's timbre and pitch while obeying the emotional direction.

## 5. The Orchestrator Agent (`web/app.py`)
**"The Interface"**
The bridge between the user and the AI pipeline.
- **Role:** Manages the API and Web UI.
- **Workflow:**
    1.  **Input:** User uploads reference audio + text (or document).
    2.  **Stream:** Pushes generated audio chunks back to the browser in real-time.
    3.  **Export:** Stitches the final performance into a download-ready file.
