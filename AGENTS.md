# VibeVox System Agents

The VibeVox architecture is composed of five distinct agents that work together in a producer-consumer pipeline to transform raw text into emotionally resonant speech.

## 1. The Ingestion Agent (`ingestion.py`)
This is the entry point of the pipeline. Its primary responsibility is to break down large blocks of text into smaller, semantically meaningful units that can be processed individually.

- **Role:** Text Pre-processor
- **Key Task:** Semantic Chunking
- **Logic:** It uses Natural Language Processing (NLTK) to identify sentence boundaries and group them into paragraphs, ensuring that no chunk exceeds the context window of the Analysis Agent while preserving narrative flow.

## 2. The Analysis Agent (`analysis.py`)
Serving as the "brain" of the operation, this agent determines *how* the text should be spoken.

- **Role:** Sentiment & Context Analyst
- **Powered By:** Groq (Llama 3.1 8B Instant)
- **Key Task:** Emotion Tagging
- **Output:** It assigns an `Emotion` (e.g., "Suspense", "Joy", "Anger") and an `Intensity` (High/Medium/Low) to each text chunk. It effectively reads between the lines to understand the subtext.

## 3. The Compiler Agent (`style_prompt_compiler.py`)
This agent acts as the translator between the abstract analysis and the concrete needs of the TTS engine.

- **Role:** Prompt Engineer
- **Key Task:** Style Mapping
- **Process:** It takes the structured data from the Analysis Agent (e.g., `Emotion="Fear", Intensity="High"`) and converts it into a rich, natural language description that the TTS model can understand (e.g., *"A shaky, whispering voice, speaking quickly and with urgency"*).

## 4. The Synthesis Agent (`tts_server.py`)
The "mouth" of VibeVox, this is a dedicated heavy-lifting server that runs the generative audio model.

- **Role:** Audio Generator
- **Powered By:** Qwen3-TTS (Voice Design Mode)
- **Key Task:** Text-to-Speech Synthesis
- **Performance:** optimized with Flash Attention 2 for low-latency generation. It maintains a consistent speaker identity ("Voice Anchor") while applying the dynamic style prompts from the Compiler Agent.

## 5. The Orchestrator Agent (`web/app.py`)
This is the user-facing interface that manages the flow of data between all other agents.

- **Role:** System Controller & API Layer
- **Key Task:** Pipeline Management
- **Features:**
    - **Live Streaming:** Pushes audio chunks to the client as soon as they are ready using Server-Sent Events (SSE).
    - **Export Management:** Handles long-running audiobook generation jobs, stitching thousands of audio clips into a single coherent file with cross-fading.
    - **Resource Management:** Ensures the GPU is fed efficiently without overlapping requests causing OOM errors.
