"""TTS Server for VibeVox using Qwen3-TTS VoiceDesign model."""

from __future__ import annotations

import base64
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
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_LANGUAGE = "English"


class TTSRequest(BaseModel):
    text: str
    style_prompt: str = ""
    speaker: str = "Male_Narrator"
    language: str = DEFAULT_LANGUAGE
    reference_audio: Optional[str] = None  # Base64 encoded audio
    reference_text: Optional[str] = None   # Transcript of the reference audio


class TTSServer:
    """Qwen3-TTS server with Dua-Model capabilities (Cloning & Design)."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        design_model_id: str = DEFAULT_DESIGN_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        use_flash_attn: bool = True,
    ):
        self.base_model_id = base_model_id
        self.design_model_id = design_model_id
        self.device = device
        self.dtype = dtype
        self.use_flash_attn = use_flash_attn
        self.model_base: Optional[Qwen3TTSModel] = None
        self.model_design: Optional[Qwen3TTSModel] = None

    def _load_single_model(self, model_id: str) -> Qwen3TTSModel:
        """Helper to load a single model instance."""
        attn_implementation = "flash_attention_2" if self.use_flash_attn else "eager"
        try:
            logger.info(f"Loading model: {model_id}...")
            return Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            logger.error(f"Failed to load {model_id} with flash_attn: {e}")
            if self.use_flash_attn:
                logger.info("Retrying with eager attention...")
                return Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation="eager",
                )
            raise e

    def load_model(self):
        """Load Qwen3-TTS models based on configuration."""
        should_load_base = os.environ.get("LOAD_BASE_MODEL", "true").lower() == "true"
        should_load_design = os.environ.get("LOAD_DESIGN_MODEL", "true").lower() == "true"

        if (not should_load_base or self.model_base is not None) and \
           (not should_load_design or self.model_design is not None):
            logger.info("Requested models already loaded.")
            return

        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")

        try:
            # 1. Load Base Model (Cloning)
            if should_load_base:
                if self.model_base is None:
                    self.model_base = self._load_single_model(self.base_model_id)
                    logger.info("✓ Base (Cloning) Model loaded!")
            else:
                logger.info("⏭️  Skipping Base Model (Cloning) to save memory.")

            # 2. Load Design Model (Styling)
            if should_load_design:
                if self.model_design is None:
                    self.model_design = self._load_single_model(self.design_model_id)
                    logger.info("✓ Design (Style) Model loaded!")
            else:
                 logger.info("⏭️  Skipping Design Model (Styling) to save memory.")

        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            raise

    def synthesize(
        self,
        text: str,
        style_prompt: str = "",
        language: str = DEFAULT_LANGUAGE,
        reference_audio: bytes | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, int]:
        """
        Synthesize speech using either Voice Cloning (Base) or Voice Design.

        Args:
            text: The text to synthesize
            style_prompt: Optional styling instruction
            language: Target language
            reference_audio: Bytes of the reference audio file
            reference_text: Transcript of the reference audio

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        if self.model_base is None and self.model_design is None:
            raise RuntimeError("No models loaded. Check configuration.")

        logger.info(f"Synthesizing: '{text[:50]}...'")

        try:
            if reference_audio:
                # --- Voice Cloning Mode (Base Model) ---
                if self.model_base is None:
                    raise HTTPException(status_code=400, detail="Voice Cloning (Base Model) is disabled on this server to save memory.")
                
                logger.info("✓ Mode: Voice Cloning (Base Model)")
                audio_io = io.BytesIO(reference_audio)
                wav, sr = sf.read(audio_io)
                wavs, output_sr = self.model_base.generate(
                    text=text,
                    language=language,
                    prompt_wavs=[wav],
                    prompt_text=reference_text,
                )
            else:
                # --- Voice Design Mode (Design Model) ---
                if self.model_design is None:
                    raise HTTPException(status_code=400, detail="Voice Design (Style Model) is disabled on this server to save memory.")

                logger.info("✓ Mode: Voice Design (Instruct Model)")
                if not style_prompt:
                    raise ValueError("Style prompt required for Voice Design mode")
                
                logger.info(f"Style: '{style_prompt[:80]}...'")
                wavs, output_sr = self.model_design.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=style_prompt,
                )

            # Convert numpy array to WAV bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wavs[0], output_sr, format='WAV')
            audio_bytes = audio_buffer.getvalue()

            logger.info(f"✓ Synthesized {len(audio_bytes)} bytes at {output_sr}Hz")
            return audio_bytes, output_sr

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise


# Initialize global TTS server
tts_server = TTSServer()
app = FastAPI(title="VibeVox TTS Server")


@app.on_event("startup")
async def startup_event():
    """Load the models on startup."""
    logger.info("Starting VibeVox TTS Server...")
    tts_server.load_model()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": {
            "base": tts_server.model_base is not None,
            "design": tts_server.model_design is not None,
        },
        "model_ids": {
            "base": tts_server.base_model_id,
            "design": tts_server.design_model_id,
        },
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

    # Decode reference audio if present
    reference_audio_bytes = None
    if request.reference_audio:
        try:
            reference_audio_bytes = base64.b64decode(request.reference_audio)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio")
    
    # If no reference audio, we MUST have a style prompt for the Design model
    if not reference_audio_bytes and not request.style_prompt.strip():
        raise HTTPException(status_code=400, detail="Style prompt is required when not using Voice Cloning")

    try:
        audio_bytes, sample_rate = tts_server.synthesize(
            text=request.text,
            style_prompt=request.style_prompt,
            language=request.language,
            reference_audio=reference_audio_bytes,
            reference_text=request.reference_text,
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
