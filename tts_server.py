"""TTS Server for VibeVox using Qwen3-TTS VoiceDesign model."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_LANGUAGE = "English"


class TTSRequest(BaseModel):
    text: str
    style_prompt: str
    speaker: str = "Male_Narrator"
    language: str = DEFAULT_LANGUAGE


class TTSServer:
    """Qwen3-TTS server with voice design capabilities."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        use_flash_attn: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_flash_attn = use_flash_attn
        self.model: Optional[Qwen3TTSModel] = None

    def load_model(self):
        """Load the Qwen3-TTS model."""
        if self.model is not None:
            logger.info("Model already loaded.")
            return

        logger.info(f"Loading Qwen3-TTS model: {self.model_id}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")

        try:
            attn_implementation = "flash_attention_2" if self.use_flash_attn else "eager"
            
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
            logger.info("✓ Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to eager attention if flash_attn fails
            if self.use_flash_attn:
                logger.info("Retrying with eager attention...")
                try:
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.model_id,
                        device_map=self.device,
                        dtype=self.dtype,
                        attn_implementation="eager",
                    )
                    logger.info("✓ Model loaded successfully with eager attention!")
                except Exception as e2:
                    logger.error(f"Failed to load model even with eager attention: {e2}")
                    raise

    def synthesize(
        self,
        text: str,
        style_prompt: str,
        language: str = DEFAULT_LANGUAGE,
    ) -> tuple[bytes, int]:
        """
        Synthesize speech using voice design mode.

        Args:
            text: The text to synthesize
            style_prompt: Natural language description of desired voice characteristics
            language: Target language (default: English)

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Synthesizing: '{text[:50]}...'")
        logger.info(f"Style: '{style_prompt[:80]}...'")

        try:
            # Generate speech using voice design
            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=style_prompt,
            )

            # Convert numpy array to WAV bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wavs[0], sr, format='WAV')
            audio_bytes = audio_buffer.getvalue()

            logger.info(f"✓ Synthesized {len(audio_bytes)} bytes at {sr}Hz")
            return audio_bytes, sr

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise


# Initialize global TTS server
tts_server = TTSServer()
app = FastAPI(title="VibeVox TTS Server")


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    logger.info("Starting VibeVox TTS Server...")
    tts_server.load_model()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": tts_server.model is not None,
        "model_id": tts_server.model_id,
        "device": tts_server.device,
    }


@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text with dynamic voice styling.

    Args:
        request: TTSRequest containing text, style_prompt, speaker, and language

    Returns:
        StreamingResponse with audio/wav content
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if not request.style_prompt.strip():
        raise HTTPException(status_code=400, detail="Style prompt is required")

    try:
        audio_bytes, sample_rate = tts_server.synthesize(
            text=request.text,
            style_prompt=request.style_prompt,
            language=request.language,
        )

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="output.wav"',
                "X-Sample-Rate": str(sample_rate),
            },
        )

    except Exception as e:
        logger.error(f"TTS request failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "5000"))

    logger.info(f"Starting TTS server on {host}:{port}")
    uvicorn.run(
        "tts_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
